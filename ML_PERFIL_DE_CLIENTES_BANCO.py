# Databricks notebook source
# MAGIC %md
# MAGIC ## SEGMENTAÇÃO DE CLIENTES BASEADO NO USO DO CARTÃO DE CRÉDITO
# MAGIC
# MAGIC O conjunto de dados para este dataset consiste no comportamento de uso do cartão de crédito dos clientes, com cerca de 9000 titulares de cartão de crédito ativos durante os últimos 6 meses com 18 características comportamentais. A segmentação dos clientes pode ser utilizada para definir estratégias de marketing.
# MAGIC
# MAGIC **OBJETIVO**     
# MAGIC Identificar o perfil de clientes de cartão de crédito através de comportamentos relacionados as transações bancárias e extrair insights para área de marketing realizar estratégias e campanhas direcionadas para cada perfil de cliente.
# MAGIC
# MAGIC **INFO DATASET:**
# MAGIC - CUST_ID: Identificação do titular do cartão de crédito (Categórico)
# MAGIC - BALANCE: Valor do saldo restante em sua conta para fazer compras
# MAGIC - BALANCE_FREQUENCY: Com que frequência o saldo é atualizado, pontuação entre 0 e 1 (1 = frequentemente atualizado, 0 = não atualizado com frequência)
# MAGIC - PURCHASES: Valor das compras feitas a partir da conta
# MAGIC - ONEOFF_PURCHASES: Maior valor de compra feito de uma só vez
# MAGIC - INSTALLMENTS_PURCHASES: Valor das compras feitas em parcelas
# MAGIC - CASH_ADVANCE: Adiantamento em dinheiro fornecido pelo usuário
# MAGIC - PURCHASES_FREQUENCY: Com que frequência as compras são feitas, pontuação entre 0 e 1 (1 = frequentemente compradas, 0 = não compradas com frequência)
# MAGIC - ONEOFFPURCHASESFREQUENCY: Com que frequência as compras são feitas de uma só vez (1 = frequentemente compradas, 0 = não compradas com frequência)
# MAGIC - PURCHASESINSTALLMENTSFREQUENCY: Com que frequência as compras parceladas são feitas (1 = frequentemente feitas, 0 = não feitas com frequência)
# MAGIC - CASHADVANCEFREQUENCY: Com que frequência o adiantamento em dinheiro é pago
# MAGIC - CASHADVANCETRX: Número de transações feitas com "Adiantamento em Dinheiro"
# MAGIC - PURCHASES_TRX: Número de transações de compra realizadas
# MAGIC - CREDIT_LIMIT: Limite de crédito do cartão para o usuário
# MAGIC - PAYMENTS: Valor do pagamento feito pelo usuário
# MAGIC - MINIMUM_PAYMENTS: Valor mínimo dos pagamentos feitos pelo usuário
# MAGIC - PRCFULLPAYMENT: Percentual do pagamento total pago pelo usuário
# MAGIC - TENURE: Tempo de serviço do cartão de crédito para o usuário

# COMMAND ----------

# MAGIC %md # 1. IMPORTAÇÃO DE BIBLIOTECAS E VISUALIZAÇÃO DO DATASET
# MAGIC

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from pyspark.sql import SparkSession
warnings.filterwarnings("ignore")

# COMMAND ----------


# Iniciar uma sessão Spark
spark = SparkSession.builder.getOrCreate()

# Ler o arquivo CSV no Databricks usando spark.read.csv
df_spark = spark.read.csv("dbfs:/FileStore/shared_uploads/pauloalmeidalog@gmail.com/CC_GENERAL.csv", header=True, inferSchema=True)

# Converter o DataFrame do Spark para um DataFrame do Pandas
dataframe = df_spark.toPandas()

# Mostrar as primeiras linhas do DataFrame
dataframe.head()


# COMMAND ----------

dataframe.TENURE.describe()

# COMMAND ----------


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pandas_profiling
import seaborn as sns
import warnings
import os
#import yellowbrick
import scipy.cluster.hierarchy as shc
import matplotlib.patches as patches


# Importar o PySpark e o Pandas
from pyspark.sql import SparkSession
#import pandas as pd

# Iniciar uma sessão Spark
spark = SparkSession.builder.getOrCreate()

# Ler o arquivo CSV no Databricks usando spark.read.csv
df_spark = spark.read.csv("dbfs:/FileStore/shared_uploads/pauloalmeidalog@gmail.com/CC_GENERAL.csv", header=True, inferSchema=True)

# Converter o DataFrame do Spark para um DataFrame do Pandas
dataframe = df_spark.toPandas()

# Mostrar as primeiras linhas do DataFrame
dataframe.head()


# COMMAND ----------

dataframe.info()

# COMMAND ----------



dataframe.describe().T

# COMMAND ----------

dataframe.nunique()

# COMMAND ----------

# Verificando dados nulos

dataframe.isnull().sum().sort_values(ascending=False).head()


# COMMAND ----------

dataframe.fillna(dataframe.MINIMUM_PAYMENTS.median(), inplace=True)
dataframe.dropna(subset=['CREDIT_LIMIT'], inplace=True)

missing = dataframe.isna().sum()
dataframe.drop(['CUST_ID'], axis=1, inplace=True)

print(missing)

# COMMAND ----------

# MAGIC %md ## Tratamento de dados:
# MAGIC
# MAGIC - **MINIMUM_PAYMENTS:** Devido à grande discrepância observada entre os valores mínimo e máximo na coluna, optei por utilizar a mediana para imputar os valores ausentes (N.A) nesta coluna. 
# MAGIC
# MAGIC A escolha da mediana como método de imputação baseia-se na sua robustez em relação a valores extremos e sua capacidade de fornecer uma estimativa mais representativa da tendência central dos dados quando comparado à média.
# MAGIC
# MAGIC - **CUST_ID:** Dropei esta coluna, pois não agrega valor ao modelo.
# MAGIC
# MAGIC - **CREDIT_LIMIT:** Dropei a única linha (N.A)
# MAGIC

# COMMAND ----------

# MAGIC %md # 2. Análise exploratória de Dados - EDA
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md ## Boxplot
# MAGIC
# MAGIC 1. **Colunas com Caixas Estreitas e Sem Outliers:**
# MAGIC    - Os dados são menos dispersos e não contêm valores extremos.
# MAGIC
# MAGIC 2. **Colunas com Caixas Largas e Muitos Outliers:**
# MAGIC    - Os dados têm alta variabilidade e contêm muitos valores extremos, indicando possíveis anomalias ou grande variação nos dados.
# MAGIC
# MAGIC 3. **Colunas com Distribuição Assimétrica:**
# MAGIC    - Os dados podem ser inclinados para um lado, indicando uma distribuição não normal.
# MAGIC
# MAGIC 4. **Colunas com Medianas Diferentes:**
# MAGIC    - Pode indicar diferentes níveis centrais para as variáveis em comparação.
# MAGIC

# COMMAND ----------

# boxsplot
plt.figure(figsize=(20, 15))  

# Selecionar colunas numéricas
num_cols = dataframe.select_dtypes(include=['number']).columns

# Criar boxplots para cada coluna numérica
for i, column in enumerate(num_cols):
    plt.subplot(len(num_cols) // 3 + 1, 3, i + 1)  # Ajustar o layout com mais linhas e colunas
    sns.boxplot(x=dataframe[column])  # Dados no eixo X
    plt.title(f'Boxplot de {column}', fontsize=10)
    plt.xlabel(column)  



plt.tight_layout()
plt.show()



# COMMAND ----------

dataframe.MINIMUM_PAYMENTS.describe()

# COMMAND ----------


dataframe.CREDIT_LIMIT.describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Análise de Correlação - Heatmap
# MAGIC - Analisando a correlação foi possível identificaralgumas colunas com alto índice, criei um novo Dataframe para analisar-las separadamente com o grafico de dispersão entre as colunas com maior índice
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------


corr_matrix = dataframe.corr()

plt.figure(figsize=(12,12))
sns.heatmap(corr_matrix, annot=True)
plt.show()

# COMMAND ----------

important_columns = ['BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT'] 

# COMMAND ----------

# Encontrar pares de colunas com correlação maior que 0,60 ou menor que -0,6
high_corr_pairs = []  # Inicializa uma lista vazia para armazenar os pares de colunas altamente correlacionadas
checked = set()  # Inicializa um conjunto vazio para armazenar os pares de colunas já verificados

# Percorre cada coluna no DataFrame de correlações
for col1 in corr_matrix.columns:
    # Para cada coluna, percorre novamente todas as colunas para comparar cada par possível
    for col2 in corr_matrix.columns:
        # Verifica se as colunas são diferentes, se a correlação entre elas é maior que 0,6 ou menor que -0,6,
        # e se o par de colunas ainda não foi verificado (para evitar duplicatas)
        if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.6 and (col2, col1) not in checked:
            # Se todas as condições forem atendidas, adiciona o par de colunas à lista de pares altamente correlacionados
            high_corr_pairs.append((col1, col2))
            # Adiciona o par ao conjunto de pares verificados
            checked.add((col1, col2))

# Gera um gráfico de dispersão para cada par selecionado
for pair in high_corr_pairs:
    # Cria um par de gráficos de dispersão para as colunas do par selecionado
    sns.pairplot(dataframe, height=2, x_vars=pair[0], y_vars=pair[1], kind='scatter')
    # Adiciona um título ao gráfico com o nome das colunas comparadas
    plt.suptitle(f'Scatter Plot for {pair[0]} and {pair[1]}', y=1.35)  # Ajuste a posição do título
    # Ajusta o layout para adicionar mais espaço entre o título e o gráfico
    plt.subplots_adjust(top=0.85)  # Ajuste o valor conforme necessário
    # Exibe o gráfico
    plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlações Identificadas entre Principais Características

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC **PURCHASES e ONEOFF_PURCHASES:**
# MAGIC
# MAGIC **Insight:** Essas colunas exibem uma correlação significativa, sugerindo uma forte conexão entre compras gerais e compras avulsas. Clientes que se envolvem em compras gerais tendem a fazer transações únicas.
# MAGIC
# MAGIC **PURCHASES e INSTALLMENTS_PURCHASES:**
# MAGIC
# MAGIC **Insight:** Existe uma correlação notável entre PURCHASES e INSTALLMENTS_PURCHASES, indicando uma ligação coerente entre compras gerais e aquelas feitas em parcelas. Isso sugere que clientes que fazem compras frequentes também aproveitam a opção de dividir seus pagamentos.
# MAGIC
# MAGIC **PURCHASES e PURCHASES_TRX:**
# MAGIC
# MAGIC **Insight:** A correlação observada entre PURCHASES e PURCHASES_TRX implica uma relação entre o valor total das compras e o número de transações individuais de compra. Essa conexão pode refletir comportamentos de compra variados entre os clientes.
# MAGIC
# MAGIC **PURCHASES e PAYMENTS:**
# MAGIC
# MAGIC **Insight:** Uma correlação discernível é encontrada entre PURCHASES e PAYMENTS, destacando uma conexão entre a atividade de compra de um cliente e os valores de pagamento subsequentes. Essa correlação indica a dinâmica financeira dos clientes ao gerenciar seus saldos de cartão de crédito.
# MAGIC
# MAGIC **INSTALLMENTS_PURCHASES e PURCHASES_TRX:**
# MAGIC
# MAGIC **Insight:** Essas colunas exibem uma correlação significativa, sugerindo uma conexão entre compras feitas em parcelas e o número de transações individuais de compra. Clientes que optam por pagamentos parcelados podem se envolver em transações mais frequentes.
# MAGIC
# MAGIC **CASH_ADVANCE e CASH_ADVANCE_FREQUENCY:**
# MAGIC
# MAGIC **Insight:** Uma correlação notável entre CASH_ADVANCE e CASH_ADVANCE_FREQUENCY indica uma relação entre o montante de adiantamentos em dinheiro e a frequência com que os clientes utilizam essa opção. Essa correlação ilumina os hábitos financeiros dos clientes que buscam liquidez imediata.
# MAGIC
# MAGIC **CASH_ADVANCE e CASH_ADVANCE_TRX:**
# MAGIC
# MAGIC **Insight:** A correlação observada sugere uma conexão entre o valor total dos adiantamentos em dinheiro e o número de transações envolvendo adiantamentos. Isso pode indicar que os clientes que usam adiantamentos em dinheiro o fazem de forma consistente, com a frequência das transações desempenhando um papel.
# MAGIC
# MAGIC **PURCHASES_FREQUENCY e PURCHASES_INSTALLMENTS_FREQUENCY:**
# MAGIC
# MAGIC **Insight:** Existe uma correlação significativa entre PURCHASES_FREQUENCY e PURCHASES_INSTALLMENTS_FREQUENCY, indicando uma relação entre a frequência das compras gerais e a frequência das compras feitas em parcelas. Esse insight oferece uma visão sobre as preferências dos clientes em relação ao tempo e à natureza de suas compras.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC Essas correlações identificadas oferecem insights valiosos sobre a natureza inter-relacionada das principais características do conjunto de dados, fornecendo uma base para entender os comportamentos e padrões financeiros dos clientes.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md # 3. Machine learning

# COMMAND ----------

# MAGIC %md ## Método do Cotovelo (Elbow Method)
# MAGIC
# MAGIC **O Método do Cotovelo é uma técnica para determinar o número ideal de clusters em um conjunto de dados ao utilizar o algoritmo K-Means. Ele ajuda a identificar o ponto em que a adição de mais clusters não resulta em uma melhora significativa na qualidade do agrupamento.**
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC - **Determinação do Número de Clusters**: Encontrar o número ótimo de clusters que equilibra a complexidade do modelo e a variabilidade interna dos dados, minimizando a Soma dos Quadrados dos Erros (SSE).
# MAGIC
# MAGIC ### Como Funciona
# MAGIC
# MAGIC - **Execução do K-Means para Diversos Números de Clusters**: O algoritmo K-Means é executado para uma faixa de valores de \( k \), o número de clusters, e a SSE é calculada para cada valor.
# MAGIC - **Plotagem da SSE em Função do Número de Clusters**: Cria-se um gráfico com o número de clusters no eixo x e a SSE no eixo y. O gráfico geralmente mostra uma queda acentuada na SSE seguida de uma estabilização.
# MAGIC - **Identificação do "Cotovelo"**: O ponto no gráfico onde a taxa de redução da SSE começa a diminuir é conhecido como o "cotovelo" e indica o número ideal de clusters.
# MAGIC
# MAGIC ### Aplicações
# MAGIC
# MAGIC - **Seleção de Número de Clusters**: Auxiliar na escolha do número de clusters mais apropriado para análise e segmentação.
# MAGIC - **Qualidade do Agrupamento**: Avaliar a eficácia do clustering, garantindo que mais clusters não resultem em melhorias significativas.
# MAGIC
# MAGIC

# COMMAND ----------

from sklearn.cluster import KMeans

kmeans_models = [KMeans(n_clusters=k, random_state=23).fit(values) for k in range (1, 10)]
innertia = [model.inertia_ for model in kmeans_models]

plt.plot(range(1, 10), innertia)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# COMMAND ----------

# MAGIC %md ## Normalizer
# MAGIC
# MAGIC **O Normalizer é uma técnica de pré-processamento de dados utilizada para normalizar amostras de um conjunto de dados, ajustando a escala para que todas as amostras tenham uma norma (ou comprimento) unitária.**
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC - **Normalização de Dados**: Ajustar a magnitude dos dados para que cada amostra tenha uma norma igual a 1. Isso é útil para garantir que todos os dados sejam comparáveis em termos de escala.
# MAGIC
# MAGIC ### Como Funciona
# MAGIC
# MAGIC - **Transformação L2**: Normaliza cada amostra dividindo os valores pelos seus normais L2 (raiz quadrada da soma dos quadrados dos valores). O resultado é que cada amostra terá uma norma unitária.
# MAGIC
# MAGIC ### Aplicações
# MAGIC
# MAGIC - **Escalonamento para Algoritmos Sensíveis à Escala**: Preparar dados para algoritmos que são sensíveis à escala, como K-Means e SVM.
# MAGIC - **Comparabilidade de Amostras**: Garantir que todas as amostras tenham a mesma escala, o que pode ser importante para algoritmos que utilizam distâncias ou produtos escalares.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from sklearn.preprocessing import Normalizer

values = Normalizer().fit_transform(dataframe.values)


print(values)

# COMMAND ----------

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(values)

# COMMAND ----------

# MAGIC %md
# MAGIC # Validação do modelo

# COMMAND ----------

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_clustering_metrics(values, cluster_range):
    # Listas para armazenar os scores
    silhouette_scores = []
    calinski_scores = []
    davies_bouldin_scores = []

    # Looping pelos números de clusters
    for n_clusters in cluster_range:
        model = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300).fit(values)
        
        # Calcular e armazenar os scores
        silhouette_scores.append(silhouette_score(values, model.labels_))
        calinski_scores.append(calinski_harabasz_score(values, model.labels_))
        davies_bouldin_scores.append(davies_bouldin_score(values, model.labels_))

    # Criar o dashboard com 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Silhouette Score
    axs[0].plot(cluster_range, silhouette_scores, "bo-")
    axs[0].set_title('Silhouette Score vs Number of Clusters')
    axs[0].set_xlabel('Number of Clusters')
    axs[0].set_ylabel('Silhouette Score')
    axs[0].set_xticks(cluster_range)

    # Plot Calinski-Harabasz Score
    axs[1].plot(cluster_range, calinski_scores, "bo-")
    axs[1].set_title('Calinski-Harabasz Score vs Number of Clusters')
    axs[1].set_xlabel('Number of Clusters')
    axs[1].set_ylabel('Calinski-Harabasz Score')
    axs[1].set_xticks(cluster_range)

    # Plot Davies-Bouldin Score
    axs[2].plot(cluster_range, davies_bouldin_scores, "bo-")
    axs[2].set_title('Davies-Bouldin Score vs Number of Clusters')
    axs[2].set_xlabel('Number of Clusters')
    axs[2].set_ylabel('Davies-Bouldin Score')
    axs[2].set_xticks(cluster_range)

    # Ajustar layout para que os plots não se sobreponham
    plt.tight_layout()
    plt.show()

# Exemplo de uso da função
cluster_range = range(2, 6)
plot_clustering_metrics(values, cluster_range)


# COMMAND ----------

# MAGIC %md ## Silhouette Score
# MAGIC
# MAGIC **O Silhouette Score é uma métrica utilizada para avaliar a qualidade de uma clusterização, medindo a coesão e separação dos clusters. Ele fornece uma indicação de quão bem cada ponto se encaixa no seu próprio cluster em comparação com outros clusters.**
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC - **Avaliação da Qualidade da Clusterização**: Medir o quão bem definidos e separados estão os clusters em um conjunto de dados, ajudando a escolher o número ideal de clusters.
# MAGIC
# MAGIC ### Como Funciona
# MAGIC
# MAGIC - **Cálculo do Silhouette Score para Cada Ponto**: O Silhouette Score para cada ponto é calculado com base na distância média entre o ponto e outros pontos no mesmo cluster (cohesion) e a distância média entre o ponto e pontos no cluster mais próximo (separation). O score varia entre -1 e 1, onde valores próximos a 1 indicam que o ponto está bem posicionado no cluster, valores próximos a 0 indicam que o ponto está na fronteira entre clusters, e valores negativos indicam que o ponto pode ter sido alocado ao cluster errado.
# MAGIC
# MAGIC - **Média do Silhouette Score**: O Silhouette Score global para um conjunto de dados é a média dos scores individuais dos pontos. Um Silhouette Score mais alto indica uma melhor separação entre clusters e uma melhor definição dos clusters.
# MAGIC
# MAGIC ### Aplicações
# MAGIC
# MAGIC - **Escolha do Número de Clusters**: Auxilia na seleção do número ideal de clusters ao comparar o Silhouette Score para diferentes números de clusters.
# MAGIC - **Avaliação da Qualidade do Agrupamento**: Fornece uma medida quantitativa da qualidade do agrupamento, ajudando a identificar a configuração de clustering que resulta em clusters mais bem definidos e separados.
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md ## Índice de Calinski-Harabasz (Calinski-Harabasz Index)
# MAGIC
# MAGIC **O Índice de Calinski-Harabasz é uma métrica de avaliação de clustering que mede a qualidade dos clusters ao calcular a relação entre a variabilidade dentro dos clusters e a variabilidade entre os clusters.**
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC - **Avaliação da Qualidade dos Clusters**: Medir a separação entre os clusters e a coesão dentro dos clusters, ajudando a identificar o número ideal de clusters.
# MAGIC
# MAGIC ### Como Funciona
# MAGIC
# MAGIC - **Cálculo da Variabilidade**: O índice calcula a variabilidade entre os clusters (variância entre clusters) e a variabilidade dentro dos clusters (variância dentro dos clusters). O valor do índice é dado pela razão entre a variância entre clusters e a variância dentro dos clusters.
# MAGIC   
# MAGIC - **Interpretação do Índice**: Um valor mais alto do Índice de Calinski-Harabasz indica que os clusters são mais bem separados e mais coesos. Isso sugere uma melhor qualidade do clustering.
# MAGIC
# MAGIC ### Aplicações
# MAGIC
# MAGIC - **Escolha do Número de Clusters**: Ajuda na seleção do número ideal de clusters comparando o índice para diferentes números de clusters.
# MAGIC - **Avaliação da Qualidade do Agrupamento**: Fornece uma medida da separação e coesão dos clusters, ajudando a determinar a configuração de clustering que resulta em clusters bem definidos.
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %md ## Índice Davies-Bouldin (DBI)
# MAGIC
# MAGIC **O Índice Davies-Bouldin é uma métrica utilizada para avaliar a qualidade de uma clusterização ao medir a média da relação entre a dispersão dentro dos clusters e a separação entre clusters.**
# MAGIC
# MAGIC ### Objetivo
# MAGIC
# MAGIC - **Avaliação da Qualidade dos Clusters**: Medir a média da relação entre a dispersão dentro dos clusters e a separação entre clusters, ajudando a identificar a qualidade do clustering.
# MAGIC
# MAGIC ### Como Funciona
# MAGIC
# MAGIC - **Cálculo do Índice**: O DBI calcula a relação entre a distância média entre os centros dos clusters e a dispersão média dos pontos dentro dos clusters. Um valor baixo do DBI indica que os clusters são bem separados e coesos, enquanto um valor alto sugere que os clusters estão mal separados e/ou têm alta dispersão.
# MAGIC
# MAGIC - **Interpretação do Índice**: Um valor mais baixo do Índice Davies-Bouldin indica uma melhor qualidade do clustering, com clusters mais bem separados e mais coesos.
# MAGIC
# MAGIC ### Aplicações
# MAGIC
# MAGIC - **Escolha do Número de Clusters**: Auxilia na seleção do número ideal de clusters ao comparar o DBI para diferentes números de clusters.
# MAGIC - **Avaliação da Qualidade do Agrupamento**: Fornece uma medida da separação e coesão dos clusters, ajudando a avaliar a configuração de clustering que resulta em clusters mais distintos e bem definidos.

# COMMAND ----------

dataframe.count()

# COMMAND ----------

# MAGIC %md ## Validação da estabilidade do Cluster
# MAGIC
# MAGIC - **Validação 1:** Aplicação do cluster em um dataset randominco que retornou valores complemente diferentes dos resultados anteriores
# MAGIC - **Validação 2:** Divisão do dataset com numpy para check de estabilidade do cluster: os resultados seguem estáveis seguindo similaridade
# MAGIC

# COMMAND ----------

def clustering_algorithm(n_clusters, dataset):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(dataset)
    s = metrics.silhouette_score(dataset, labels, metric='euclidean')
    dbs = metrics.davies_bouldin_score(dataset, labels)
    calinski = metrics.calinski_harabasz_score(dataset, labels)
    return s, dbs, calinski

# COMMAND ----------


kmeans = KMeans(n_clusters=4, n_init=10, max_iter=300)
y_pred = kmeans.fit_predict(values)

# COMMAND ----------

s1, dbs1, calinski1 = clustering_algorithm(4, values)
print(s1, dbs1, calinski1)

# COMMAND ----------

s2, dbs2, calinski2 = clustering_algorithm(5, values)
print(s2, dbs2, calinski2)

# COMMAND ----------

s3, dbs3, calinski3 = clustering_algorithm(50, values)
print(s3, dbs3, calinski3)

# COMMAND ----------

# MAGIC %md
# MAGIC - **Validação 1:** Aplicação do cluster em um dataset randominco que retornou valores complemente diferentes dos resultados anteriores

# COMMAND ----------

random_data = np.random.rand(8950, 16)
s, dbs, calinski = clustering_algorithm(4, random_data)

print(s, dbs, calinski)
print(s2, dbs2, calinski2)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC - **Validação 2:** Divisão do dataset com numpy para check de estabilidade do cluster: os resultados seguem estáveis seguindo similaridade

# COMMAND ----------

set1, set2, set3 = np.array_split(values, 3)

s1, dbs1, calinski1 = clustering_algorithm(5, set1)
s2, dbs2, calinski2 = clustering_algorithm(5, set2)
s3, dbs3, calinski3 = clustering_algorithm(5, set3)

print(s1, dbs1, calinski1)
print(s2, dbs2, calinski2)
print(s3, dbs3, calinski3)

# COMMAND ----------

centroids

# COMMAND ----------

# MAGIC %md # 4.CONCLUSÃO
# MAGIC  **Foi decidido utilizar o modelo com 4 clusters, devido ao resultado das métricas de validação**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualização do cluster

# COMMAND ----------



# Número de clusters desejados
n_clusters = 4

# Criação do modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_principal)

# Previsão dos clusters
labels = kmeans.predict(X_principal)

# Criação de um mapa de cores para os clusters
colors = plt.cm.winter(np.linspace(0, 1, n_clusters))

# Visualizando a clusterização com um gráfico de dispersão
plt.figure(figsize=(10, 6))
for i in range(n_clusters):
    plt.scatter(X_principal.loc[labels == i, 'P1'], 
                X_principal.loc[labels == i, 'P2'], 
                color=colors[i], 
                label=f'Cluster {i}')

plt.title('KMeans Clustering')
plt.xlabel('P1')
plt.ylabel('P2')
plt.legend(title='Clusters')  # Adiciona uma legenda com o título 'Clusters'
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretando o perfil de clientes de acordo com Clusters
# MAGIC
# MAGIC
# MAGIC **Analisando as principais features agrupadas por Cluster, foi possível identificar as seguintes caracteristicas:**
# MAGIC - **CLUSTER 0:** Clientes que menos gastam. Clientes com o maior limite. Bons pagadores. Maior número de clientes.
# MAGIC - **CLUSTER 1:** Maior número de saques. Menor quantidade de compras. Limite de Crédito alto. Nem sempre pagam. 
# MAGIC - **CLUSTER 2:** Maior valor em compras. Menor quantidade de clientes. Melhores pagadores.
# MAGIC
# MAGIC - **CLUSTER 3:** Clientes que mais gastam. Piores pagadores. Boa quantidade de clientes.
# MAGIC

# COMMAND ----------


dataframe["CLUSTER"] = labels
description = dataframe.groupby("CLUSTER")["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]
n_clients = description.size()
description = description.mean()
description['n_clients'] = n_clients
description

# COMMAND ----------



dataframe.groupby("CLUSTER")["PRC_FULL_PAYMENT"].describe()

# COMMAND ----------

dataframe.groupby("CLUSTER")["CREDIT_LIMIT"].describe()

# COMMAND ----------

dataframe.groupby("CLUSTER")["BALANCE"].describe()

# COMMAND ----------

dataframe.groupby("CLUSTER")["PURCHASES"].describe()

df1 = dataframe.groupby("CLUSTER")["PURCHASES"].sum()
df1_percentage = df1 / df1.sum()

df1 = df1.to_frame()  # Converte a Série para um DataFrame
df1["%"] = df1_percentage.round(1)*100

df1["PURCHASES"] = df1["PURCHASES"].apply(lambda x: f'{x:,.2f}'.replace(',', '.'))


df1

# COMMAND ----------

# MAGIC %md ## Análise de correlação por Cluster

# COMMAND ----------




def plot_correlations_by_cluster(data):
    """
    Plota gráficos de dispersão para pares de colunas com correlação significativa, 
    usando a coluna 'CLUSTER' do DataFrame para classificação nos gráficos.

    :param data: DataFrame contendo os dados com a coluna 'CLUSTER'.
    """
    # Verifica se a coluna 'CLUSTER' está no DataFrame
    if 'CLUSTER' not in data.columns:
        raise ValueError("A coluna 'CLUSTER' não foi encontrada no DataFrame.")

    # Calcula a matriz de correlação
    corr_matrix = data.corr()

    # Encontrar pares de colunas com correlação maior que 0,60 ou menor que -0,6
    high_corr_pairs = []
    checked = set()

    for col1 in corr_matrix.columns:
        for col2 in corr_matrix.columns:
            if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.6 and (col2, col1) not in checked:
                high_corr_pairs.append((col1, col2))
                checked.add((col1, col2))
    
    # Gera um gráfico de dispersão para cada par selecionado
    for pair in high_corr_pairs:
        plt.figure(figsize=(10, 6))  # Ajusta o tamanho da figura
        sns.scatterplot(data=data, x=pair[0], y=pair[1], hue='CLUSTER', palette='viridis')
        #plt.title(f'Scatter Plot for {pair[0]} and {pair[1]}')
        plt.suptitle(f'Scatter Plot for {pair[0]} and {pair[1]}', y=1.05)  # Ajuste a posição do título
        plt.subplots_adjust(top=0.9)  # Ajuste o valor conforme necessário
        plt.xlabel(pair[0])
        plt.ylabel(pair[1])
        plt.legend(title='CLUSTER')  # Adiciona uma legenda com o título 'CLUSTER'
        plt.show()

# Exemplo de uso:
# plot_correlations_by_cluster(data=dataframe)


# COMMAND ----------

plot_correlations_by_cluster(dataframe)

# COMMAND ----------

# MAGIC %md ## Correlações Identificadas por perfil de cliente

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **PURCHASES e ONEOFF_PURCHASES:**
# MAGIC
# MAGIC **Insight:** Perfil de cliente do Cluster 0, bons pagadores e também maior quantidade de clientes, é o que mais faz compras avulsas e gerais, a maioria são compras até 10.000,00 $.
# MAGIC
# MAGIC **PURCHASES e INSTALLMENTS_PURCHASES:**
# MAGIC
# MAGIC **Insight:** Perfil de cliente do Cluster 0, representam a maioria das compras gerais e parceladas.
# MAGIC
# MAGIC **PURCHASES e PURCHASES_TRX:**
# MAGIC **Insight:**  Comportamento variado de compras e número de transações individuais, a maior parte são clientes Cluster 0, 2 e 3.
# MAGIC
# MAGIC **PURCHASES e PAYMENTS:**
# MAGIC
# MAGIC **Insight:** Clientes do cluster 2 costumam realizar compras em seu cadastro próprio, consequentemente é o grupo que realiza compras mais caras e são os melhores pagadores. Clientes do cluster 0, realizam mais quantidades de compras e pagamentos feitos por outros usuários ou sem cadastro.
# MAGIC **INSTALLMENTS_PURCHASES e PURCHASES_TRX:**
# MAGIC
# MAGIC **Insight:** Clientes do cluster 0, 1 e 3 são os que fazem mais compras parceladas e transações individuais de compra. 
# MAGIC
# MAGIC **CASH_ADVANCE e CASH_ADVANCE_FREQUENCY:**
# MAGIC
# MAGIC **Insight:** Clientes do cluster 1 e 2, são os que mais sacam dinheiro frequentemente, destes os do clusters 2 são os que fazem saques de maior valor
# MAGIC
# MAGIC **PURCHASES_FREQUENCY e PURCHASES_INSTALLMENTS_FREQUENCY:**
# MAGIC
# MAGIC **Insight:** Há uma grande variedade entre todos os tipos de clientes/clusters relacionados a correlação de PURCHASES_FREQUENCY e PURCHASES_INSTALLMENTS_FREQUENCY, indicando uma relação entre a frequência das compras gerais e a frequência das compras feitas em parcelas.
# MAGIC
# MAGIC **CASH_ADVANCE e CASH_ADVANCE_TRX:**
# MAGIC
# MAGIC **Insight:** há uma grande correlação entre clientes que fazem pagamentos e saque dinheiro frequentemente, a maior parte desses clientes são do cluster 2 e 1.
# MAGIC
