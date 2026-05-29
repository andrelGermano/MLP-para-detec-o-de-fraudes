import pandas as pd

COLUNAS_DESCARTAR = ['nameOrig', 'nameDest', 'isFlaggedFraud']
COLUNA_ALVO = 'isFraud'


def carregar_dados(caminho='dados/paysim_sample.csv'):
    df = pd.read_csv(caminho)
    # descarta identificadores e coluna redundante de fraude
    df = df.drop(columns=COLUNAS_DESCARTAR)
    # features
    X = df.drop(columns=[COLUNA_ALVO])
    # alvo
    y = df[COLUNA_ALVO]
    return X, y


if __name__ == '__main__':
    X, y = carregar_dados()
    print("Features:", X.columns.tolist())
    print("Shape X:", X.shape, "| Shape y:", y.shape)
    print("Distribuição y:\n", y.value_counts())
