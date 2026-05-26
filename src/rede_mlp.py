import torch
import torch.nn as nn

# entrada -> 10 -> saída
CAMADAS_ESCONDIDAS = [10]

ATIVACAO = nn.GELU
DROPOUT = 0.3


class MLP(nn.Module):
    """Rede Perceptron Multicamadas para classificação binária de fraudes.

    Estrutura: entrada -> [camadas escondidas com ativação + dropout] -> saída Sigmoid.
    A saída é um único número entre 0 e 1, interpretado como a probabilidade de fraude.
    """

    def __init__(self, dim_entrada, camadas_escondidas=CAMADAS_ESCONDIDAS,
                 ativacao=ATIVACAO, dropout=DROPOUT):
        super().__init__()

        funcao_ativacao = ativacao

        # montagem da rede acumulando as camadas na ordem em que aparecem
        camadas = []

        # guarda quantas features entram na próxima camada.
        dim_anterior = dim_entrada

        for n_neuronios in camadas_escondidas:
            # combina linearmente as entradas (pesos + viés)
            camadas.append(nn.Linear(dim_anterior, n_neuronios))
            # introduz não-linearidade (GELU)
            camadas.append(funcao_ativacao())
            # regularização
            camadas.append(nn.Dropout(dropout))
            # a saída dessa camada vira a entrada da próxima
            dim_anterior = n_neuronios

        # camada de saída. reduz para 1 neurônio e aplica sigmoid.
        camadas.append(nn.Linear(dim_anterior, 1))
        camadas.append(nn.Sigmoid())

        # nn.Sequential encadeia todas as camadas na ordem da lista,
        # passando a saída de uma como entrada da seguinte.
        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        """Retorna a probabilidade de fraude de cada amostra"""
        return self.rede(x)


def criar_modelo(dim_entrada, camadas_escondidas=CAMADAS_ESCONDIDAS,
                 ativacao=ATIVACAO, dropout=DROPOUT, seed=None):
    """Cria uma instância da MLP, opcionalmente fixando a semente dos pesos."""
    if seed is not None:
        torch.manual_seed(seed)
    return MLP(dim_entrada, camadas_escondidas, ativacao, dropout)


if __name__ == '__main__':
    from preprocessamento import carregar_dados, separar_dados, gerar_folds

    X, y = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = separar_dados(X, y)
    folds = gerar_folds(X_treino, y_treino)
    dim_entrada = folds[0]['X_treino'].shape[1]

    modelo = criar_modelo(dim_entrada, seed=42)
    n_params = sum(p.numel() for p in modelo.parameters())

    print(f"Dimensão de entrada: {dim_entrada}")
    print(f"Topologia: entrada({dim_entrada}) -> {CAMADAS_ESCONDIDAS} -> saída(1)")
    print(f"Parâmetros treináveis: {n_params}")
    print(modelo)
