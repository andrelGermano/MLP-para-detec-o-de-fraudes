import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from rede_mlp import criar_modelo, CAMADAS_ESCONDIDAS, ATIVACAO, DROPOUT
from preprocessamento import carregar_dados, separar_dados, gerar_folds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TAXA_APRENDIZADO = 0.01
MAX_EPOCAS = 200
PACIENCIA = 10
LIMIAR = 0.5


def transformar_em_tensores(X, y):
    # y vira coluna (N, 1) porque é o formato que a BCELoss espera
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
    return X_t, y_t


def criar_dataloader(X_t, y_t, batch_size):
    # batch_size define a estratégia de gradiente descendente
    #  none  -> usa todos os exemplos de uma vez (batch GD)
    #  1     -> um exemplo por vez (stochastic GD)
    #  k     -> lotes de tamanho k (mini-batch GD)
    dataset = TensorDataset(X_t, y_t)
    if batch_size is None:
        batch_size = len(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def calcular_metricas(modelo, X_t, y_t, criterio):
    """Avalia o modelo sobre um conjunto e devolve perda, acurácia e F1.
       Usado a cada época para montar o histórico de convergência."""

    # desliga o comportamento de sorteio de dropout
    modelo.eval()

    # desliga o cálculo de gradientes pq estamos só avaliando, não treinando
    with torch.no_grad():
        probs = modelo(X_t.to(DEVICE))

        # compara as chances previstas com a resposta certa e devolve um número que
        # diz o quanto a rede errou 
        perda = criterio(probs, y_t.to(DEVICE)).item()

        # se a chance for >= LIMIAR vira 1 (fraude), senão vira 0 (normal)
        preds = (probs.cpu().numpy() >= LIMIAR).astype(int).ravel()

    # lista com as respostas certas
    y_true = y_t.cpu().numpy().astype(int).ravel()

    return {
        'loss': perda,
        'accuracy': accuracy_score(y_true, preds),
        'f1': f1_score(y_true, preds, zero_division=0),
    }


def treinar(modelo, fold, batch_size=None, lr=TAXA_APRENDIZADO,
            max_epocas=MAX_EPOCAS, paciencia=PACIENCIA):
    """Treina um modelo em um fold. Retorna o modelo treinado, o histórico
    por época e o tempo de treinamento."""
    modelo = modelo.to(DEVICE)

    # define a função de perda como BCE
    criterio = nn.BCELoss()

    # define o otimizador como SGD (gradiente descendente estocástico)
    otimizador = optim.SGD(modelo.parameters(), lr=lr)

    X_tr, y_tr = transformar_em_tensores(fold['X_treino'], fold['y_treino'])
    X_val, y_val = transformar_em_tensores(fold['X_val'], fold['y_val'])
    loader = criar_dataloader(X_tr, y_tr, batch_size)

    historico = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
    }

    # variáveis de controle do early stopping.
    melhor_val_loss = float('inf')
    melhor_estado = None
    epocas_sem_melhora = 0

    inicio = time.perf_counter()
    for _ in range(max_epocas):

        modelo.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            # zera gradientes da época anterior
            otimizador.zero_grad()
            # forward
            saida = modelo(X_batch)
            # calcula o erro (BCE)
            perda = criterio(saida, y_batch)
            # backpropagation: calcula gradientes
            perda.backward()
            # atualiza os pesos
            otimizador.step()

        # métricas de treino e validação de 1 época
        m_tr = calcular_metricas(modelo, X_tr, y_tr, criterio)
        m_val = calcular_metricas(modelo, X_val, y_val, criterio)
        historico['train_loss'].append(m_tr['loss'])
        historico['val_loss'].append(m_val['loss'])
        historico['train_acc'].append(m_tr['accuracy'])
        historico['val_acc'].append(m_val['accuracy'])
        historico['train_f1'].append(m_tr['f1'])
        historico['val_f1'].append(m_val['f1'])

        # early stopping: guarda o melhor modelo e para se estagnar
        if m_val['loss'] < melhor_val_loss:
            melhor_val_loss = m_val['loss']
            melhor_estado = {k: v.clone() for k, v in modelo.state_dict().items()}
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                break

    tempo = time.perf_counter() - inicio

    # restaura os pesos da melhor época
    if melhor_estado is not None:
        modelo.load_state_dict(melhor_estado)

    return modelo, historico, tempo


def avaliar(modelo, X, y):
    """Avaliação completa de um modelo treinado: todas as métricas + matriz de confusão.
    Usada na validação de cada fold e no teste final."""

    X_t, y_t = transformar_em_tensores(X, y)
    modelo = modelo.to(DEVICE)
    modelo.eval()

    with torch.no_grad():
        probs = modelo(X_t.to(DEVICE)).cpu().numpy()
    preds = (probs >= LIMIAR).astype(int).ravel()
    y_true = y_t.cpu().numpy().astype(int).ravel()
    return {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'matriz_confusao': confusion_matrix(y_true, preds),
    }


def treinar_validacao_cruzada(folds, camadas_escondidas=CAMADAS_ESCONDIDAS,
                              ativacao=ATIVACAO, dropout=DROPOUT,
                              batch_size=None, lr=TAXA_APRENDIZADO,
                              max_epocas=MAX_EPOCAS, paciencia=PACIENCIA, seed=42):
    """Treina e avalia uma mesma configuração nos 5 folds da validação cruzada.
    Retorna uma lista de métricas (uma por fold) e os históricos de cada fold."""

    resultados = []
    historicos = []

    # repete todo o processo de treino e avaliação para cada um dos 5 folds
    for fold in folds:
        # quantas colunas (features) entram na rede (vem do pré-processamento)
        dim_entrada = fold['X_treino'].shape[1]

        # cria um modelo novo para cada fold, para começar do zero (sem reaproveitar
        # o que foi aprendido no fold anterior)
        modelo = criar_modelo(
            dim_entrada, camadas_escondidas=camadas_escondidas,
            ativacao=ativacao, dropout=dropout, seed=seed,
        )

        # treina nesse fold. devolve o modelo já treinado, o histórico e o tempo gasto
        modelo, historico, tempo = treinar(
            modelo, fold, batch_size=batch_size, lr=lr,
            max_epocas=max_epocas, paciencia=paciencia,
        )

        # mede o desempenho na parte de validação do fold (dados que não foram treinados)
        metricas = avaliar(modelo, fold['X_val'], fold['y_val'])


        metricas['tempo'] = tempo

        resultados.append(metricas)
        historicos.append(historico)

    return resultados, historicos


def resumir_resultados(resultados):
    """Calcula a média das métricas numéricas ao longo dos folds"""
    chaves = ['accuracy', 'precision', 'recall', 'f1', 'tempo']
    return {chave: float(np.mean([r[chave] for r in resultados])) for chave in chaves}


if __name__ == '__main__':
    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)
    folds = gerar_folds(X_treino, y_treino)

    print("Treinando configuração base (Mini-batch GD, batch=32)...")
    resultados, historicos = treinar_validacao_cruzada(folds, batch_size=32, seed=42)

    for i, r in enumerate(resultados, start=1):
        print(f"  Fold {i}: acc={r['accuracy']:.4f} | f1={r['f1']:.4f} | tempo={r['tempo']:.1f}s")

    media = resumir_resultados(resultados)
    print("\nMédia dos folds:")
    for chave, valor in media.items():
        print(f"  {chave}: {valor:.4f}")
