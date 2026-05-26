import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold

COLUNAS_DESCARTAR = ['nameOrig', 'nameDest', 'isFlaggedFraud']
COLUNA_ALVO = 'isFraud'

COLUNAS_NUMERICAS_CONTINUAS = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
]
COLUNAS_NUMERICAS_DISCRETAS = ['step']
COLUNAS_NUMERICAS = COLUNAS_NUMERICAS_CONTINUAS + COLUNAS_NUMERICAS_DISCRETAS
COLUNAS_CATEGORICAS = ['type']

N_FOLDS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def carregar_dados(caminho='dados/paysim_sample.csv'):
    df = pd.read_csv(caminho)
    # descarta identificadores e coluna redundante de fraude
    df = df.drop(columns=COLUNAS_DESCARTAR)
    # features
    X = df.drop(columns=[COLUNA_ALVO])
    # alvo
    y = df[COLUNA_ALVO]
    return X, y


def criar_pipeline():
    # contínuas: mediana
    transformador_continuo = Pipeline([
        ('imputacao', SimpleImputer(strategy='median')),
        ('normalizacao', StandardScaler()),
    ])

    # discretas: média
    transformador_discreto = Pipeline([
        ('imputacao', SimpleImputer(strategy='mean')),
        ('normalizacao', StandardScaler()),
    ])

    # categóricas: moda + one-hot encoding
    transformador_categorico = Pipeline([
        ('imputacao', SimpleImputer(strategy='most_frequent')),
        ('codificacao', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessador = ColumnTransformer([
        ('num_cont', transformador_continuo, COLUNAS_NUMERICAS_CONTINUAS),
        ('num_disc', transformador_discreto, COLUNAS_NUMERICAS_DISCRETAS),
        ('cat', transformador_categorico, COLUNAS_CATEGORICAS),
    ])

    return preprocessador


def separar_dados(X, y):
    # 20% reservado para teste final
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def gerar_folds(X_treino, y_treino):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []

    for idx_tr, idx_val in skf.split(X_treino, y_treino):
        # separa os índices do fold em subconjuntos de treino e validação
        X_train = X_treino.iloc[idx_tr]
        X_val = X_treino.iloc[idx_val]
        y_train = y_treino.iloc[idx_tr].values
        y_val = y_treino.iloc[idx_val].values

        # pipeline novo a cada fold
        # fit só no treino, transform no treino e validação
        pipeline = criar_pipeline()
        X_train_proc = pipeline.fit_transform(X_train)
        X_val_proc = pipeline.transform(X_val)

        # salvo para ser reutilizado no teste final pelo módulo de treinamento
        folds.append({
            'X_treino': X_train_proc,
            'y_treino': y_train,
            'X_val': X_val_proc,
            'y_val': y_val,
            'pipeline': pipeline,
        })

    return folds


def nomes_features(pipeline_ajustado):
    # recupera nomes após one-hot encoding para análise e relatório
    cat_names = list(
        pipeline_ajustado.named_transformers_['cat']
        .named_steps['codificacao']
        .get_feature_names_out(COLUNAS_CATEGORICAS)
    )
    return COLUNAS_NUMERICAS_CONTINUAS + COLUNAS_NUMERICAS_DISCRETAS + cat_names


if __name__ == '__main__':
    X, y = carregar_dados()
    print("Features:", X.columns.tolist())
    print("Shape X:", X.shape, "| Shape y:", y.shape)
    print("Distribuição y:\n", y.value_counts())

    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)
    print(f"\nTreino: {X_treino.shape} | Teste: {X_teste.shape}")

    folds = gerar_folds(X_treino, y_treino)
    print(f"\nFolds gerados: {len(folds)}")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: treino={fold['X_treino'].shape} | val={fold['X_val'].shape}")

    print("\nFeatures resultantes:", nomes_features(folds[0]['pipeline']))
