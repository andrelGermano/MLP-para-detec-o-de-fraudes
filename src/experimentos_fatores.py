import os
import itertools

import pandas as pd

from preprocessamento import carregar_dados, separar_dados, gerar_folds
from treinamento import treinar_validacao_cruzada, resumir_resultados

NEURONIOS = [5, 10, 20]
RODADAS_TAXAS = [
    ('experimentos_fatores.csv', [0.001, 0.01, 0.1]),
    ('experimentos_fatores_novos.csv', [0.05, 0.2, 0.5]),
]
BATCH_SIZE = 32
# 10 execuções por configuração, com sementes distintas
SEMENTES = list(range(1, 11))

PASTA_RESULTADOS = 'resultados'


def executar_rodada(folds, taxas, arquivo_saida):
    linhas = []

    for neuronios, lr in itertools.product(NEURONIOS, taxas):
        for seed in SEMENTES:
            # validação cruzada completa (5 folds) com uma semente
            resultados, _ = treinar_validacao_cruzada(
                folds,
                camadas_escondidas=[neuronios],
                batch_size=BATCH_SIZE,
                lr=lr,
                seed=seed,
            )
            # média das métricas sobre os 5 folds desta execução.
            media = resumir_resultados(resultados)

            linhas.append({'neuronios': neuronios, 'lr': lr, 'seed': seed, **media})
            print(f"neuronios={neuronios:>2} lr={lr:<5} seed={seed:>2} "
                  f"| f1={media['f1']:.4f} acc={media['accuracy']:.4f}")

    df = pd.DataFrame(linhas)
    caminho_saida = os.path.join(PASTA_RESULTADOS, arquivo_saida)
    df.to_csv(caminho_saida, index=False)
    return df, caminho_saida


def executar_experimentos():
    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)
    folds = gerar_folds(X_treino, y_treino)

    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    dfs = []
    for arquivo_saida, taxas in RODADAS_TAXAS:
        print(f"\n=== Rodada: lr={taxas} -> {arquivo_saida} ===")
        df_rodada, caminho_saida = executar_rodada(folds, taxas, arquivo_saida)
        dfs.append(df_rodada)
        print(f"Resultados salvos em: {caminho_saida}")

    return pd.concat(dfs, ignore_index=True)


if __name__ == '__main__':
    df = executar_experimentos()

    print("\n=== Média por configuração (sobre as 10 execuções) ===")
    resumo = (
        df.drop(columns='seed')
        .groupby(['neuronios', 'lr'])
        .mean(numeric_only=True)
    )
    print(resumo.round(4).to_string())
    print(f"\nResultados salvos em: {PASTA_RESULTADOS}/experimentos_fatores*.csv")
