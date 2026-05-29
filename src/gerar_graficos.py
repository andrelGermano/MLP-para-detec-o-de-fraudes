import os
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

PASTA_RESULTADOS = 'resultados'
PASTA_GRAFICOS = os.path.join(PASTA_RESULTADOS, 'graficos')

ARQUIVOS_FATORES = [
    os.path.join(PASTA_RESULTADOS, 'experimentos_fatores.csv'),
    os.path.join(PASTA_RESULTADOS, 'experimentos_fatores_novos.csv'),
]
ARQUIVO_RANKING_CONFIG = os.path.join(PASTA_RESULTADOS, 'ranking_configuracoes.csv')
ARQUIVO_EXPERIMENTOS_GD = os.path.join(PASTA_RESULTADOS, 'experimentos_gd.csv')
ARQUIVO_RANKING_GD = os.path.join(PASTA_RESULTADOS, 'ranking_gd.csv')
ARQUIVO_HISTORICO_FINAL = os.path.join(PASTA_RESULTADOS, 'modelo_final_historico.csv')
ARQUIVO_MATRIZ_FINAL = os.path.join(PASTA_RESULTADOS, 'modelo_final_matriz_confusao.csv')


def configurar_estilo():
    plt.rcParams.update({
        'figure.dpi': 130,
        'savefig.dpi': 180,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'font.size': 10,
    })


def salvar_figura(nome):
    caminho = os.path.join(PASTA_GRAFICOS, nome)
    plt.tight_layout()
    plt.savefig(caminho, bbox_inches='tight')
    plt.close()
    print(f"salvo: {caminho}")


def carregar_fatores():
    dfs = [pd.read_csv(caminho) for caminho in ARQUIVOS_FATORES if os.path.exists(caminho)]
    if not dfs:
        raise FileNotFoundError("Nenhum CSV de experimentos de fatores encontrado.")
    return pd.concat(dfs, ignore_index=True)


def grafico_f1_por_lr_e_neuronios(df):
    resumo = (
        df.groupby(['neuronios', 'lr'])['f1']
        .mean()
        .reset_index()
        .sort_values('lr')
    )

    plt.figure(figsize=(8, 4.5))
    for neuronios, grupo in resumo.groupby('neuronios'):
        plt.plot(
            grupo['lr'].astype(str),
            grupo['f1'],
            marker='o',
            linewidth=2,
            label=f'{neuronios} neurônios',
        )
    plt.title('F1 médio por taxa de aprendizado')
    plt.xlabel('Learning rate')
    plt.ylabel('F1 médio')
    plt.ylim(0, 1)
    plt.legend(title='Camada escondida')
    salvar_figura('fatores_f1_por_lr.png')


def grafico_heatmap_f1(df):
    tabela = df.groupby(['neuronios', 'lr'])['f1'].mean().unstack('lr').sort_index()
    valores = tabela.values

    plt.figure(figsize=(8, 4.2))
    plt.imshow(valores, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    plt.colorbar(label='F1 médio')
    plt.xticks(range(len(tabela.columns)), [str(c) for c in tabela.columns])
    plt.yticks(range(len(tabela.index)), [str(i) for i in tabela.index])
    plt.xlabel('Learning rate')
    plt.ylabel('Neurônios')
    plt.title('Mapa de F1 médio por configuração')

    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            plt.text(j, i, f'{valores[i, j]:.3f}', ha='center', va='center', color='black')

    salvar_figura('fatores_heatmap_f1.png')


def grafico_boxplot_lr(df):
    lrs = sorted(df['lr'].unique())
    dados = [df.loc[df['lr'] == lr, 'f1'] for lr in lrs]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(dados, tick_labels=[str(lr) for lr in lrs], showmeans=True)
    plt.title('Variabilidade do F1 por taxa de aprendizado')
    plt.xlabel('Learning rate')
    plt.ylabel('F1')
    plt.ylim(0, 1)
    salvar_figura('fatores_boxplot_f1_por_lr.png')


def grafico_top_configuracoes():
    if not os.path.exists(ARQUIVO_RANKING_CONFIG):
        return

    ranking = pd.read_csv(ARQUIVO_RANKING_CONFIG).head(10)
    labels = [f"{int(r.neuronios)}n | lr={r.lr:g}" for _, r in ranking.iterrows()]

    plt.figure(figsize=(9, 5))
    y = np.arange(len(ranking))
    plt.barh(y, ranking['f1'], color='#3b82f6')
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel('F1 médio')
    plt.title('Top 10 configurações por F1')
    plt.xlim(0, 1)
    for i, valor in enumerate(ranking['f1']):
        plt.text(valor + 0.01, i, f'{valor:.3f}', va='center')
    salvar_figura('ranking_top10_configuracoes.png')


def grafico_gd_metricas():
    if not os.path.exists(ARQUIVO_RANKING_GD):
        return

    ranking = pd.read_csv(ARQUIVO_RANKING_GD)
    metricas = ['accuracy', 'precision', 'recall', 'f1']
    labels = ranking['estrategia'].tolist()
    x = np.arange(len(labels))
    largura = 0.2

    plt.figure(figsize=(9, 5))
    for idx, metrica in enumerate(metricas):
        plt.bar(x + (idx - 1.5) * largura, ranking[metrica], width=largura, label=metrica)
    plt.xticks(x, labels, rotation=15, ha='right')
    plt.ylabel('Valor médio')
    plt.ylim(0, 1)
    plt.title('Comparação de métricas por estratégia de GD')
    plt.legend()
    salvar_figura('gd_metricas_por_estrategia.png')


def grafico_gd_tempo():
    if not os.path.exists(ARQUIVO_RANKING_GD):
        return

    ranking = pd.read_csv(ARQUIVO_RANKING_GD)

    plt.figure(figsize=(7, 4.5))
    plt.bar(ranking['estrategia'], ranking['tempo'], color='#14b8a6')
    plt.ylabel('Tempo médio (s)')
    plt.title('Tempo médio por estratégia de GD')
    plt.xticks(rotation=15, ha='right')
    for i, valor in enumerate(ranking['tempo']):
        plt.text(i, valor + 0.05, f'{valor:.2f}s', ha='center')
    salvar_figura('gd_tempo_por_estrategia.png')


def grafico_gd_boxplot():
    if not os.path.exists(ARQUIVO_EXPERIMENTOS_GD):
        return

    df = pd.read_csv(ARQUIVO_EXPERIMENTOS_GD)
    estrategias = df['estrategia'].drop_duplicates().tolist()
    dados = [df.loc[df['estrategia'] == estrategia, 'f1'] for estrategia in estrategias]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(dados, tick_labels=estrategias, showmeans=True)
    plt.title('Variabilidade do F1 por estratégia de GD')
    plt.xlabel('Estratégia')
    plt.ylabel('F1')
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha='right')
    salvar_figura('gd_boxplot_f1.png')


def grafico_historico_modelo_final():
    if not os.path.exists(ARQUIVO_HISTORICO_FINAL):
        return

    historico = pd.read_csv(ARQUIVO_HISTORICO_FINAL)

    plt.figure(figsize=(8, 4.5))
    plt.plot(historico['epoca'], historico['train_acc'], label='accuracy treino', linewidth=2)
    plt.plot(historico['epoca'], historico['val_acc'], label='accuracy validação', linewidth=2)
    plt.plot(historico['epoca'], historico['train_f1'], label='F1 treino', linewidth=2)
    plt.plot(historico['epoca'], historico['val_f1'], label='F1 validação', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Métrica')
    plt.ylim(0, 1)
    plt.title('Evolução de accuracy e F1 no modelo final')
    plt.legend()
    salvar_figura('modelo_final_accuracy_f1.png')

    plt.figure(figsize=(8, 4.5))
    plt.plot(historico['epoca'], historico['train_loss'], label='loss treino', linewidth=2)
    plt.plot(historico['epoca'], historico['val_loss'], label='loss validação', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('BCE loss')
    plt.title('Evolução da perda no modelo final')
    plt.legend()
    salvar_figura('modelo_final_loss.png')


def grafico_matriz_confusao():
    if not os.path.exists(ARQUIVO_MATRIZ_FINAL):
        return

    matriz = pd.read_csv(ARQUIVO_MATRIZ_FINAL, index_col=0)
    valores = matriz.values

    plt.figure(figsize=(5, 4.5))
    plt.imshow(valores, cmap='Blues')
    plt.colorbar(label='Quantidade')
    plt.xticks(range(2), ['Pred. normal', 'Pred. fraude'])
    plt.yticks(range(2), ['Real normal', 'Real fraude'])
    plt.title('Matriz de confusão do modelo final')

    limite = valores.max() / 2
    for i in range(2):
        for j in range(2):
            cor = 'white' if valores[i, j] > limite else 'black'
            plt.text(j, i, str(valores[i, j]), ha='center', va='center', color=cor, fontsize=13)

    salvar_figura('modelo_final_matriz_confusao.png')


def main():
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    configurar_estilo()

    fatores = carregar_fatores()
    grafico_f1_por_lr_e_neuronios(fatores)
    grafico_heatmap_f1(fatores)
    grafico_boxplot_lr(fatores)
    grafico_top_configuracoes()
    grafico_gd_metricas()
    grafico_gd_tempo()
    grafico_gd_boxplot()
    grafico_historico_modelo_final()
    grafico_matriz_confusao()

    print(f"\nGráficos salvos em: {PASTA_GRAFICOS}")


if __name__ == '__main__':
    main()
