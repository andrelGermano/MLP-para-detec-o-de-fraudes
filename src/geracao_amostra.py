import pandas as pd

MATRICULA = 20230020736
SEED = MATRICULA % (2**32)

csv_path = "dados/PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(csv_path)
fraudes = df[df["isFraud"] == 1].sample(n=500, random_state=SEED)
normais = df[df["isFraud"] == 0].sample(n=2500, random_state=SEED)

amostra = pd.concat([fraudes, normais])
amostra = amostra.sample(frac=1, random_state=SEED).reset_index(drop=True)

amostra.to_csv("dados/paysim_sample.csv", index=False)
