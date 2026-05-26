import os

import pandas as pd

PASTA_RESULTADOS = 'resultados'
ARQUIVOS_FATORES = [
    os.path.join(PASTA_RESULTADOS, 'experimentos_fatores.csv'),
    os.path.join(PASTA_RESULTADOS, 'experimentos_fatores_novos.csv'),
]
ARQUIVO_RANKING = os.path.join(PASTA_RESULTADOS, 'ranking_configuracoes.csv')

METRICAS = ['accuracy', 'precision', 'recall', 'f1', 'tempo']


def carregar_resultados():
    dfs = [pd.read_csv(a) for a in ARQUIVOS_FATORES if os.path.exists(a)]
    if not dfs:
        raise FileNotFoundError(
            "Nenhum CSV de fatores encontrado. Rode experimentos_fatores.py primeiro."
        )
    return pd.concat(dfs, ignore_index=True)


def gerar_ranking():
    df = carregar_resultados()
    # média das métricas sobre as 10 sementes de cada configuração, ordenada por f1 decrescente
    ranking = (
        df.groupby(['neuronios', 'lr'])[METRICAS]
        .mean()
        .reset_index()
        .sort_values('f1', ascending=False)
        .reset_index(drop=True)
    )
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    ranking.to_csv(ARQUIVO_RANKING, index=False)
    return ranking


if __name__ == '__main__':
    ranking = gerar_ranking()

    print("=== Ranking das configurações (média sobre as 10 execuções) ===")
    print(ranking.round(4).to_string(index=False))

    melhor = ranking.iloc[0]
    print(f"\nMelhor configuração: neuronios={int(melhor['neuronios'])} "
          f"lr={melhor['lr']} | f1={melhor['f1']:.4f}")
    print(f"Ranking salvo em: {ARQUIVO_RANKING}")
