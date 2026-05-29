import os

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessamento import carregar_dados, separar_dados, criar_pipeline, RANDOM_STATE
from rede_mlp import criar_modelo
from treinamento import treinar, avaliar

SEED = 42                # semente fixa para o modelo final ser reprodutível
VAL_SIZE = 0.2           # fração do treino reservada para validação

PASTA_RESULTADOS = 'resultados'
ARQUIVO_RANKING_GD = os.path.join(PASTA_RESULTADOS, 'ranking_gd.csv')


def converter_batch_size(valor):
    if str(valor) == 'full':
        return None
    return int(valor)


def ler_config_final():
    melhor = pd.read_csv(ARQUIVO_RANKING_GD).iloc[0]
    return {
        'camadas_escondidas': [int(melhor['neuronios'])],
        'lr': float(melhor['lr']),
        'batch_size': converter_batch_size(melhor['batch_size']),
        'estrategia': melhor['estrategia'],
    }


def treinar_modelo_final():
    config = ler_config_final()
    X, y = carregar_dados()

    # 1) divisão em teste e treino
    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)

    # 2) reserva uma validação para o early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_treino, y_treino,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_treino,
    )

    # 3) pré-processamento ajustado SÓ no treino e aplicado em validação e teste
    pipeline = criar_pipeline()
    X_tr_proc = pipeline.fit_transform(X_tr)
    X_val_proc = pipeline.transform(X_val)
    X_teste_proc = pipeline.transform(X_teste)

    # 4) treino do modelo final com a melhor config
    modelo = criar_modelo(
        X_tr_proc.shape[1],
        camadas_escondidas=config['camadas_escondidas'],
        seed=SEED,
    )
    fold = {
        'X_treino': X_tr_proc, 'y_treino': y_tr.values,
        'X_val': X_val_proc, 'y_val': y_val.values,
    }
    modelo, historico, tempo = treinar(
        modelo,
        fold,
        batch_size=config['batch_size'],
        lr=config['lr'],
    )

    # 5) avaliação final no conjunto de teste 
    metricas = avaliar(modelo, X_teste_proc, y_teste.values)

    # 6) salva o histórico de convergência para os gráficos
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    pd.DataFrame(historico).to_csv(
        os.path.join(PASTA_RESULTADOS, 'modelo_final_historico.csv'),
        index_label='epoca',
    )
    pd.DataFrame([{k: v for k, v in metricas.items() if k != 'matriz_confusao'}]).to_csv(
        os.path.join(PASTA_RESULTADOS, 'modelo_final_metricas.csv'),
        index=False,
    )
    pd.DataFrame(
        metricas['matriz_confusao'],
        index=['real_normal', 'real_fraude'],
        columns=['pred_normal', 'pred_fraude'],
    ).to_csv(os.path.join(PASTA_RESULTADOS, 'modelo_final_matriz_confusao.csv'))

    return modelo, metricas, historico, tempo, config


if __name__ == '__main__':
    modelo, metricas, historico, tempo, config = treinar_modelo_final()

    print("=== MODELO FINAL ===")
    print(
        f"Topologia: entrada -> {config['camadas_escondidas']} -> 1 "
        f"| lr={config['lr']} | estrategia={config['estrategia']} "
        f"| batch={config['batch_size'] if config['batch_size'] is not None else 'full'}"
    )
    print(f"Épocas treinadas: {len(historico['train_loss'])} | tempo: {tempo:.1f}s")

    print("\n--- Métricas no conjunto de TESTE (20% reservado) ---")
    print(f"Accuracy : {metricas['accuracy']:.4f}")
    print(f"Precision: {metricas['precision']:.4f}")
    print(f"Recall   : {metricas['recall']:.4f}")
    print(f"F1-score : {metricas['f1']:.4f}")

    print("\nMatriz de confusão [[VN, FP], [FN, VP]]:")
    print(metricas['matriz_confusao'])
