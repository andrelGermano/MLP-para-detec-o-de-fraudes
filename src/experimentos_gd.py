import os

import pandas as pd

from preprocessamento import carregar_dados, separar_dados, gerar_folds
from treinamento import treinar_validacao_cruzada, resumir_resultados


ESTRATEGIAS = [
    ('Batch', None),
    ('Mini-batch 64', 64),
    ('Mini-batch 32', 32),
    ('Stochastic', 1),
]
SEMENTES = list(range(1, 11))

PASTA_RESULTADOS = 'resultados'
ARQUIVO_RANKING = os.path.join(PASTA_RESULTADOS, 'ranking_configuracoes.csv')
ARQUIVO_SAIDA = os.path.join(PASTA_RESULTADOS, 'experimentos_gd.csv')
ARQUIVO_RANKING_GD = os.path.join(PASTA_RESULTADOS, 'ranking_gd.csv')


def ler_melhor_config():
    if not os.path.exists(ARQUIVO_RANKING):
        raise FileNotFoundError(
            f"{ARQUIVO_RANKING} não encontrado. Rode selecionar_melhor.py primeiro."
        )
    melhor = pd.read_csv(ARQUIVO_RANKING).iloc[0]
    camadas = [int(melhor['neuronios'])]
    lr = float(melhor['lr'])
    return camadas, lr


def executar_experimentos():

    camadas_fixas, lr_fixa = ler_melhor_config()
    print(f"Melhor config do item 5: {camadas_fixas[0]} neurônios, lr={lr_fixa}\n")

    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)
    folds = gerar_folds(X_treino, y_treino)

    linhas = []
    for nome, batch_size in ESTRATEGIAS:
        for seed in SEMENTES:
            resultados, _ = treinar_validacao_cruzada(
                folds,
                camadas_escondidas=camadas_fixas,
                batch_size=batch_size,
                lr=lr_fixa,
                seed=seed,
            )

            media = resumir_resultados(resultados)
            linhas.append({
                'neuronios': camadas_fixas[0],
                'lr': lr_fixa,
                'estrategia': nome,
                'batch_size': batch_size if batch_size is not None else 'full',
                'seed': seed,
                **media,
            })

            print(f"{nome:<14} seed={seed:>2} "
                  f"| f1={media['f1']:.4f} acc={media['accuracy']:.4f} "
                  f"tempo={media['tempo']:.1f}s")

    df = pd.DataFrame(linhas)
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    df.to_csv(ARQUIVO_SAIDA, index=False)
    return df


def gerar_ranking_gd(df):
    metricas = ['accuracy', 'precision', 'recall', 'f1', 'tempo']
    
    ranking = (
        df.groupby(['neuronios', 'lr', 'estrategia', 'batch_size'], sort=False)[metricas]
        .mean()
        .reset_index()
        .sort_values(['f1', 'tempo'], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking.to_csv(ARQUIVO_RANKING_GD, index=False)
    return ranking


if __name__ == '__main__':
    df = executar_experimentos()

    print("\n=== Média por estratégia (sobre as 10 execuções) ===")
    ranking = gerar_ranking_gd(df)
    print(ranking.round(4).to_string(index=False))
    print(f"\nResultados salvos em: {ARQUIVO_SAIDA}")
    print(f"Ranking salvo em: {ARQUIVO_RANKING_GD}")
