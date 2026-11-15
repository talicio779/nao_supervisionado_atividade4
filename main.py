import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


 
# 1. inicialização

ARQUIVO = "MusicasJogosDataset_20232_20262.csv"
PASTA_RESULTADOS =  "resultados"
os.makedirs(PASTA_RESULTADOS, exist_ok=True)
df = pd.read_csv(ARQUIVO)

print("\nInformações básicas do dataset:")
print(df.info())

# == FIM PARTE 1 (INCIALIZAÇÂO) == #
# 2. Pré-processamento

dados = df.select_dtypes(include=np.number).copy()

if dados.empty:
    raise ValueError("O dataset não possui colunas numéricas.")

dados = dados.dropna(axis=1, how="all")

imputer = SimpleImputer(strategy="median")
dados_imputados = imputer.fit_transform(dados)

scaler = StandardScaler()
dados_escalados = scaler.fit_transform(dados_imputados)

# ++ FIM DO PRÉ-PROCESSAMENTO ++ #

# Elbow Method para definição de clusters

inertias = []
valores_k = range(2, min(11, len(dados) + 1))

for k in valores_k:

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(dados_escalados)

    inertias.append(modelo.inertia_)

plt.figure(figsize=(8, 5))

plt.plot(
    list(valores_k),
    inertias,
    marker="o"
)

# plt.title("Método do Cotovelo")
# plt.xlabel("Quantidade de clusters")
# plt.ylabel("Inércia")
# plt.grid()

# plt.savefig("resultados/elbow.png", dpi=300, bbox_inches="tight")
# plt.close()


# Metodo Silhouette para definição de clusters

silhouette_scores = []

# Testar de 2 até 10 clusters
for k in valores_k:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = kmeans.fit_predict(dados_escalados)

    score = silhouette_score(
        dados_escalados,
        clusters
    )

    silhouette_scores.append(score)

    print(f"K={k} | Silhouette Score={score:.4f}")

# Encontrar o melhor K segundo a métrica
melhor_k = list(valores_k)[
    silhouette_scores.index(max(silhouette_scores))
]

print("\nMelhor quantidade de clusters:", melhor_k)

# Visualizar resultados
# plt.figure(figsize=(8, 5))

# plt.plot(
#     list(valores_k),
#     silhouette_scores,
#     marker="o"
# )

# plt.title("Silhouette Score por quantidade de clusters")
# plt.xlabel("Número de clusters (K)")
# plt.ylabel("Silhouette Score")
# plt.xticks(list(valores_k))
# plt.grid()

# plt.savefig(
#     "resultados/silhouette.png",
#     dpi=300,
#     bbox_inches="tight"
# )

plt.close()


# 5. K-MEANS 


# NUM_CLUSTERS = melhor_k
NUM_CLUSTERS = 2 # defini como 2 após resultado do segundo algoritmo, pois o elbow não deu muito bem

print("\nMelhor quantidade de clusters final para k-means = 2")


kmeans = KMeans(
    n_clusters=NUM_CLUSTERS,
    random_state=42,
    n_init=10
)

clusters_kmeans = kmeans.fit_predict(dados_escalados)

# Adiciona os grupos ao dataset original

df["cluster"] = clusters_kmeans

print("\n===== RESULTADO DOS CLUSTERS KMEANs =====")
print(df["cluster"].value_counts().sort_index())


# avaliação do modelo 

score_kmeans = silhouette_score(
    dados_escalados,
    clusters_kmeans
)

print("\nSilhouette Score K-Means:", round(score_kmeans, 4))
print("\n\n\n")

# 6. PCA do kmeans

# Reduz as dimensões para 2 componentes
pca = PCA(n_components=2)

dados_pca = pca.fit_transform(dados_escalados)

visualizacao = pd.DataFrame({
    "Componente 1": dados_pca[:, 0],
    "Componente 2": dados_pca[:, 1],
    "Cluster": clusters_kmeans
})

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=visualizacao,
    x="Componente 1",
    y="Componente 2",
    hue="Cluster",
    palette="viridis",
    s=70
)

plt.title("Visualização dos Clusters usando PCA")

plt.savefig(
    f"{PASTA_RESULTADOS}/clusters_kmeans.png"
)

plt.close()



# 7. DBSCAN - testes


valores_eps = [6.0, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.2, 7.5, 7.8, 8.0, 8.5, 20, 30, 45, 50]

valores_min_samples = [3, 4, 5, 6, 7, 8, 9, 10, 20, 50]


resultados_dbscan = []

print("RESULTADOS TESTES DBSCAN")

for eps in valores_eps:

    for min_samples in valores_min_samples:

        modelo = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        clusters_teste = modelo.fit_predict(dados_escalados)

        mascara_teste = clusters_teste != -1

        grupos = np.unique(clusters_teste[mascara_teste])

        quantidade_clusters = len(grupos)
        quantidade_ruidos = np.sum(~mascara_teste)

        percentual_ruidos = (
            quantidade_ruidos / len(clusters_teste)
        ) * 100

        score = None

        # Silhouette apenas com pelo menos 2 clusters
        # e menos clusters do que amostras avaliadas

        if quantidade_clusters >= 2:

            score = silhouette_score(
                dados_escalados[mascara_teste],
                clusters_teste[mascara_teste]
            )

            resultados_dbscan.append({
                "eps": eps,
                "min_samples": min_samples,
                "clusters": quantidade_clusters,
                "ruidos": quantidade_ruidos,
                "percentual_ruidos": percentual_ruidos,
                "silhouette": score
            })

        print(
            f"EPS={eps} | "
            f"Min Samples={min_samples} | "
            f"Clusters={quantidade_clusters} | "
            f"Ruídos={quantidade_ruidos} | "
            f"Silhouette={score}"
        )
# salvando os resultaddos

pd.DataFrame(resultados_dbscan).to_csv(
    f"{PASTA_RESULTADOS}/experimentos_dbscan.csv",
    index=False
)

# analise dos resultados foram péssimos, depois tentei o pca

# Testar diferentes quantidades de componentes
for componentes in [5, 8, 10, 15, 20]:

    pca_dbscan = PCA(
        n_components=componentes,
        random_state=42
    )

    dados_reduzidos = pca_dbscan.fit_transform(
        dados_escalados
    )

    print(
        f"\nComponentes: {componentes}"
    )

    print(
        "Variância explicada:",
        round(
            pca_dbscan.explained_variance_ratio_.sum(),
            4
        )
    )

    for eps in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        
        modelo = DBSCAN(
            eps=eps,
            min_samples=4
        )

        clusters_teste = modelo.fit_predict(
            dados_reduzidos
        )

        mascara = clusters_teste != -1

        quantidade_clusters = len(
            set(clusters_teste) - {-1}
        )

        if quantidade_clusters >= 2:

            score = silhouette_score(
                dados_reduzidos[mascara],
                clusters_teste[mascara]
            )

            print(
                f"EPS={eps} | "
                f"Clusters={quantidade_clusters} | "
                f"Ruídos={np.sum(~mascara)} | "
                f"Silhouette={score:.4f}"
            )


#testes finais 
print("\n\ntestes finais\n\n")
resultados = pd.DataFrame(resultados_dbscan)

resultados["cobertura"] = (
    1 - resultados["ruidos"] / len(dados)
)

# Filtrar resultados minimamente aceitáveis
validos = resultados[
    (resultados["clusters"] >= 2) &
    (resultados["cobertura"] >= 0.50) &
    (resultados["silhouette"] > 0)
]

if not validos.empty:

    melhores = validos.sort_values(
        by="silhouette",
        ascending=False
    )

    print("\nMELHORES CONFIGURAÇÕES DBSCAN:")
    print(melhores.head(10))

else:
    print("Nenhuma configuração atende aos critérios.")

# após a analise dos testes cheguei nesse resultado professor 
# passei um tempo e não consegui passar de 40% de cobertura
# essa é a melhor configuração


EPS_ESCOLHIDO = 6.9
MIN_SAMPLES_ESCOLHIDO = 9

# 8. utilização no dbscan

dbscan = DBSCAN(
    eps=EPS_ESCOLHIDO,
    min_samples=MIN_SAMPLES_ESCOLHIDO
)

clusters_dbscan = dbscan.fit_predict(dados_escalados)

# Adicionar classificação ao dataset

df["cluster_dbscan"] = clusters_dbscan

print("\n===== RESULTADO FINAL DBSCAN =====")

print(
    df["cluster_dbscan"]
    .value_counts()
    .sort_index()
)

# calculo pós processamento

quantidade_clusters = len(set(clusters_dbscan) - {-1})
quantidade_ruidos = np.sum(clusters_dbscan == -1)
percentual_ruidos = (quantidade_ruidos / len(df)) * 100

print("\nQuantidade de clusters:", quantidade_clusters)
print("Quantidade de ruídos:", quantidade_ruidos)
print("Percentual de ruídos:", round(percentual_ruidos, 2), "%")

# Avaliação do DBSCAN

mascara = clusters_dbscan != -1

if (
    quantidade_clusters >= 2
    and quantidade_clusters < np.sum(mascara)
):

    score_dbscan = silhouette_score(
        dados_escalados[mascara],
        clusters_dbscan[mascara]
    )

    print(
        "\nSilhouette Score DBSCAN:",
        round(score_dbscan, 4)
    )

else:

    score_dbscan = None

    print("\nNão foi possível calcular o Silhouette Score.")



# 9. PCA - VISUALIZAÇÃO DBSCAN


visualizacao_dbscan = pd.DataFrame({
    "Componente 1": dados_pca[:, 0],
    "Componente 2": dados_pca[:, 1],
    "Cluster": clusters_dbscan.astype(str)
})

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=visualizacao_dbscan,
    x="Componente 1",
    y="Componente 2",
    hue="Cluster",
    palette="viridis",
    s=70
)

plt.title("Visualização dos Clusters DBSCAN usando PCA")

plt.savefig(
    f"{PASTA_RESULTADOS}/clusters_dbscan.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# 10. PERFIL DOS GRUPOS K-MEANS

df_numerico = dados.copy()

df_numerico["cluster"] = clusters_kmeans

perfil_kmeans = df_numerico.groupby("cluster").mean()

print("\n===== PERFIL DOS CLUSTERS =====")
print(perfil_kmeans)

#  analise dos centroides

centroides_kmeans = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=dados.columns
)

centroides_kmeans.index.name = "cluster"

centroides_kmeans.to_csv(
    f"{PASTA_RESULTADOS}/centroides_kmeans.csv"
)


# 11. PERFIL DOS GRUPOS DBSCAN


df_numerico_dbscan = dados.copy()

df_numerico_dbscan["cluster_dbscan"] = clusters_dbscan

# Excluir ruídos da análise dos perfis

perfil_dbscan = (
    df_numerico_dbscan[
        df_numerico_dbscan["cluster_dbscan"] != -1
    ]
    .groupby("cluster_dbscan")
    .mean()
)

print("\n===== PERFIL DOS CLUSTERS DBSCAN =====")
print(perfil_dbscan)



# 12. CARACTERÍSTICAS PREDOMINANTES

for cluster_id, linha in centroides_kmeans.iterrows():

    print(f"\n===== K-MEANS - CLUSTER {cluster_id} =====")

    print("\nMaiores preferências:")
    print(linha.nlargest(5))

    print("\nMenores preferências:")
    print(linha.nsmallest(5))


for cluster_id, linha in perfil_dbscan.iterrows():

    print(f"\n===== DBSCAN - CLUSTER {cluster_id} =====")

    print("\nMaiores preferências:")
    print(linha.nlargest(5))

    print("\nMenores preferências:")
    print(linha.nsmallest(5))

# 13. COMPARAÇÃO DOS AGRUPAMENTOS

print("\n===== COMPARAÇÃO DOS AGRUPAMENTOS =====")

print("\nK-MEANS:")
print(df["cluster"].value_counts().sort_index())

print("\nDBSCAN:")
print(df["cluster_dbscan"].value_counts().sort_index())

print("\n===== COMPARAÇÃO DOS SCORES =====")

print("K-Means:", round(score_kmeans, 4))

if score_dbscan is not None:
    print("DBSCAN:", round(score_dbscan, 4))
else:
    print("DBSCAN: não disponível")


# 14. EXPORTAÇÃO DOS RESULTADOS


# Dataset com os dois algoritmos
df.to_csv(
    f"{PASTA_RESULTADOS}/dataset_classificado.csv",
    index=False
)

# Perfis dos grupos
perfil_kmeans.to_csv(
    f"{PASTA_RESULTADOS}/perfil_kmeans.csv"
)

perfil_dbscan.to_csv(
    f"{PASTA_RESULTADOS}/perfil_dbscan.csv"
)

# Centroides do K-Means
centroides_kmeans.to_csv(
    f"{PASTA_RESULTADOS}/centroides_kmeans.csv"
)

print("\n================================")
print("PROCESSAMENTO FINALIZADO!")
print("Resultados salvos na pasta resultados/")
print("================================")