import pandas as pd

df = pd.read_csv('dados/paysim_sample.csv')

ALVO = 'isFraud'

print("SHAPE:", df.shape)
print("\nColunas e tipos:")
print(df.dtypes)

print("\nVALORES FALTANTES:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "  Nenhum.")

print("\nDISTRIBUIÇÃO DO ALVO (isFraud):")
print(df[ALVO].value_counts())
print(f"Proporção de fraudes: {df[ALVO].mean():.2%}")

print("\nCOLUNAS CATEGÓRICAS:")
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    n_unique = df[col].nunique()
    print(f"\n  coluna {col}: {n_unique} valores únicos de {len(df)} linhas")
    if n_unique <= 20:
        print(f"{df[col].value_counts().to_string()}")

print("\nANÁLISE DAS COLUNAS NUMÉRICAS (min → max):")
num_cols_all = df.select_dtypes(include='number').drop(columns=[ALVO]).columns
for col in num_cols_all:
    print(f"  {col}: {df[col].min():.2f} → {df[col].max():.2f}")

num_cols_all = df.select_dtypes(include='number').columns
constantes = [c for c in num_cols_all if df[c].nunique() <= 1]
print("\nCOLUNAS CONSTANTES (variância zero):")
print(constantes if constantes else "  Nenhuma.")


print("\nCORRELAÇÃO COM isFraud (numéricas):")
colunas_numericas = df.select_dtypes(include='number').columns.tolist()
colunas_numericas.remove(ALVO)
colunas_sem_constantes = [c for c in colunas_numericas if df[c].nunique() > 1]
correlacoes = df[colunas_sem_constantes].corrwith(df[ALVO])
print(correlacoes.sort_values(ascending=False).to_string())

