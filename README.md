[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Xy566FGP)
# [TADS][IC] Projeto MLP + Detecção de Fraudes

Projeto da disciplina de Inteligência Computacional do curso de Análise e Desenvolvimento de Sistemas da UFRN aplicando **redes neurais Perceptron Multicamadas (MLP)** para **classificação de fraudes financeiras**.

## Descrição

Deve-se desenvolver e analisar modelos de redes neurais MLP para o problema de classificação de transações financeiras fraudulentas. Como objetivos específicos, deve-se:

* Carregar e preparar a base de dados indicada na seção dados;
* Implementar um pipeline de pré-processamento contendo tratamento de valores faltantes, transformação de atributos categóricos em numéricos e normalização dos dados;
* Implementar uma rede neural MLP para classificação binária;
* Determinar quais topologias da rede são mais adequadas para o problema;
* Determinar o comportamento médio da rede em termos de desempenho e variabilidade entre execuções;
* Comparar experimentalmente as estratégias de treinamento por gradiente descendente;
* Analisar a ocorrência de overfitting e o efeito das técnicas de regularização.

Espera-se que, ao final do trabalho, os alunos sejam capazes de modelar redes neurais multicamadas e analisar de modo empírico o comportamento da arquitetura, dos hiperparâmetros e das estratégias de treinamento.

## Dados

Deve-se utilizar a base abaixo:

Base de dados: [PaySim – Mobile Money Transactions Simulation](https://www.kaggle.com/datasets/ealaxi/paysim1)

A base original deverá ser modificada por uma amostragem. Cada aluno deverá gerar sua própria base utilizando sua **matrícula como semente aleatória**, de forma reprodutível.

A amostra deve conter:

* **500 instâncias fraudulentas**;
* **2500 instâncias normais**.

Total:

```text
3000 instâncias
```

Script base para geração da amostra:

```python
import pandas as pd

MATRICULA = 2023123456 # substitua pela sua matrícula

df = pd.read_csv("paysim.csv")
fraudes = df[df["isFraud"] == 1].sample(n=500,random_state=MATRICULA)
normais = df[df["isFraud"] == 0].sample(n=2500,random_state=MATRICULA)

amostra = pd.concat([fraudes, normais])
amostra = amostra.sample(frac=1,random_state=MATRICULA).reset_index(drop=True)

amostra.to_csv("paysim_sample.csv", index=False)
```

O algoritmo desenvolvido deve ser executado sobre essa amostra.

## Função de Custo

A função de custo da rede neural deve ser determinada por uma função de perda apropriada ao problema de classificação binária, como padrão, utilize a entropia cruzada binária:

* Binary Cross Entropy (BCE)

Além disso, a função de perda deve ser monitorada ao longo do treinamento para análise de convergência e overfitting.

## Metodologia

Sugere-se os seguintes passos:

1. Desenvolva um módulo para carregar a base de dados e gerar a amostra;
2. Desenvolva um módulo de pré-processamento utilizando pipeline;
3. Implemente:
   * imputação de valores faltantes por média/mediana/moda, dependendo do tipo do atributo;
   * transformação de atributos categóricos para numéricos;
   * normalização dos atributos;
4. Realize separação dos dados em treino, validação e teste;
   * Reserve 20% dos dados para teste final;
   * Nos 80% dos dados de treino, utilize validação cruzada estratificada com 5 folds;
   * Em cada fold, use early stopping.
5. Desenvolva uma rede MLP, escolhendo e justificando:
   * número de camadas escondidas;
   * número de neurônios por camada;
   * taxa de aprendizado;
   * número máximo de épocas;
   * função de ativação ReLU ou GELU, com Sigmoid para saída.
6. Utilize obrigatoriamente:
   * otimizador SGD;
   * dropout;
   * early stopping;
7. Compare as seguintes estratégias de gradiente descendente:
   * Batch Gradient Descent;
   * Mini-batch Gradient Descent (avalie ao menos 2 tamanhos de lote);
   * Stochastic Gradient Descent;
8. Aplique a melhor configuração encontrada para gerar o modelo final, avaliando-o sobre o conjunto de teste final.

Para o item 2, todo o pré-processamento deve ser implementado por pipeline, evitando vazamento de dados.

Para o item 4, o conjunto de teste não pode participar da escolha de parâmetros nem do treinamento.

Para o item 5, deve-se escolher apenas 2 fatores para análise experimental, mantendo os demais fixos. A comparação do item 7 deve ser realizada separadamente e não conta como um dos dois fatores experimentais do item 5.

Para o item 5, sugere-se uma topologia inicial: entrada -> 10 -> saída.

Cada configuração deve ser executada ao menos 10 vezes usando sementes distintas controladas para permitir a reprodutibilidade dos experimentos.

A solução deve ser desenvolvida na linguagem Python utilizando **PyTorch** para implementação da rede neural. Para geração de gráficos, é livre o uso de bibliotecas auxiliares.

## Entrega

A entrega será realizada via github classroom, assim, é importante que sejam feitos commits regulares no repositório, com comentários descrevendo o que foi feito. Devem ser entregues os seguintes itens:

1. Implementação de ao menos quatro módulos:
   * carregamento dos dados;
   * pré-processamento;
   * modelagem da rede;
   * treinamento e avaliação;
2. Relatório descrevendo os seguintes itens:
   * A arquitetura da rede, seus hiperparâmetros e justificativas;
   * Escolha dos parâmetros, explicando a metodologia e apresentando resultados por meio de gráficos;
   * Gráficos da evolução da accuracy e F1 em função das iterações para séries de treino e validação;
   * Gráficos de colunas comparativos entre diferentes configurações;
   * Gráficos de boxplot dos resultados por configuração;
   * Resultados da melhor configuração:
     * Matriz de confusão;
     * Tabela contendo o valor médio das métricas de Accuracy, Precision, Recall, F1-score, Tempo de Processamento das diferentes configurações;
   * Discussão dos resultados:
     * melhor configuração;
     * ocorrência de overfitting;
     * efeito do dropout;
     * efeito do early stopping;
     * comparação entre Batch, Mini-batch e Stochastic GD;
       * impacto do tamanho do mini-batch.

## Avaliação

A avaliação considerará os itens solicitados na seção de entrega, no entanto, a atribuição da nota é condicionada à comprovação da autoria por meio de perguntas durante a apresentação, em que poderão ser solicitadas explicações, justificativas e modificações no código apresentado.
