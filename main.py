import pandas as pd
import numpy as np
import pickle
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_auc_score, top_k_accuracy_score
from utils.plotfuncs import *

# Primeiro, carrego os dados

with open("dataset/mini_gm_public_v0.1.p", "rb") as f:
    data = pickle.load(f)

# Dado a estrutura dos dados informada, acredito que através de um aninhamento de loops for seja a maneira ideal de criar o formato que precisamos

i = []

for syndrome_id, subjects in data.items():
    for subject_id, images in subjects.items():
        for image_id, embedding in images.items():
            i.append([syndrome_id, subject_id, image_id, np.array(embedding)])

df = pd.DataFrame(
    i, columns=["syndrome_id", "subject_id", "image_id", "embedding"])

# Verificando os dados
print(df.head())
print(df.info())
print(df.isnull().sum())

# --> É notável um desbalanceamento na coluna "syndrome_id", além da necessidade de converter os tipos de dados
print(df["syndrome_id"].value_counts())

# Convertendo
df["syndrome_id"] = df["syndrome_id"].astype("category")

# Transformando a coluna embedding em um array numpy
df["embedding"] = df["embedding"].to_numpy()
print(type(df["embedding"].iloc[0]))

# Rescaling
scaler = RobustScaler()
embeddings_rescaled = scaler.fit_transform(np.stack(df["embedding"]))

# Reduzindo as dimensionalidades dos dados para usar como entrada pro modelo
pca = PCA(n_components=20, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_rescaled)

print(df.dtypes)

# Agora, acredito que visualizar como a coluna syndrome_id está distribuída ajude a entender melhor os dados.

plot_bar(df["syndrome_id"], title="syndrome distribution",
         xlabel="syndrome_id", ylabel="quantity", color="orange")


# Reduzindo a dimensão com o t-SNE para visualização
tsne = TSNE(n_components=2, perplexity=30, random_state=42, learning_rate=900)
tsne_results = tsne.fit_transform(embeddings_pca)

df_tsne = pd.DataFrame(tsne_results, columns=["tsne_x", "tsne_y"])
df_tsne["syndrome_id"] = df["syndrome_id"].values

print(df_tsne.head())
print(df_tsne.shape)
print(df["syndrome_id"].shape)

# Visualizando os embeddings após a redução

plot_tsne(df_tsne, "syndrome_id")


# Finalmente, implementando o KNN
X = embeddings_pca
y = df["syndrome_id"].cat.codes  # Convertendo para valores numéricos

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Apliando balanceamento através do SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Vamos visualizar a distribuição após o balanceamento
plot_bar(pd.Series(y_train_resampled), title="syndrome distribution (POS SMOTE)",
         xlabel="syndrome_id", ylabel="quantity (POS SMOTE)", color="blue")

print(
    f"Distribuição das classes no conjunto de treino original: \n{y_train.value_counts()}\n")
print(
    f"Distribuição das classes no conjunto de treino após o SMOTE: \n{pd.Series(y_train_resampled).value_counts()}\n")

# Usando StratifiedKFold para "splittar" os dados
kf = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)

BEST_K = None
BEST_SCORE = 0

# Loop para percorrer os valores de cada K em cada métrica

metrics = ["euclidean", "cosine"]

BEST_SCORE = 0
BEST_K = None

for metric in metrics:
    for k in range(1, 16):
        print(f"Metric: {metric}")
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(
            knn, X_train_resampled, y_train_resampled, cv=kf, scoring="accuracy")
        mean_score = scores.mean()

        print(f"K --> {k}, accuracy: {mean_score:.4f}")

        if mean_score > BEST_SCORE:
            BEST_SCORE = mean_score
            BEST_K = (k, metric)

print(f"Best K: {BEST_K[0]}, metric: {BEST_K[1]}, accuracy: {BEST_SCORE:.4f}")


def evaluate_knn(best_k, metric, k=None):

    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)

    y_proba = cross_val_predict(
        knn, X_train_resampled, y_train_resampled, cv=kf, method='predict_proba')

    auc = roc_auc_score(y_train_resampled, y_proba, multi_class='ovr')
    f1 = f1_score(y_train_resampled, y_proba.argmax(
        axis=1), average='weighted')
    acc = accuracy_score(y_train_resampled, y_proba.argmax(axis=1))

    top_k_acc = top_k_accuracy_score(y_train_resampled, y_proba, k=k)

    return auc, f1, acc, top_k_acc


# Calculando com a função evaluate_knn para ambas as métricas
auc_euclidean, f1_euclidean, acc_euclidean, top_k_acc_euclidean = evaluate_knn(
    BEST_K[0], 'euclidean', k=3)
auc_cosine, f1_cosine, acc_cosine, top_k_acc_cosine = evaluate_knn(
    BEST_K[0], 'cosine', k=3)

# Visualizando os resultados
print("\nKNN results:\n")
print(
    f"Euclidean: [AUC] --> {auc_euclidean:.4f}, [F1-Score] --> {f1_euclidean:.4f}, [Accuracy] --> {acc_euclidean:.4f} [TOP-K Accuracy] --> {top_k_acc_euclidean:.4f}")
print(
    f"Cosine: [AUC] --> {auc_cosine:.4f}, [F1-Score] --> {f1_cosine:.4f}, [Accuracy] --> {acc_cosine:.4f} [TOP-K Accuracy] -->  {top_k_acc_cosine:.4f}")

# Comparando os resultados
plot_table_metrics(auc_euclidean, auc_cosine, f1_euclidean,
                   f1_cosine, acc_euclidean, acc_cosine)

# Visualizando a curva ROC
plot_roc_curve(y_train_resampled, cross_val_predict(KNeighborsClassifier(
    n_neighbors=BEST_K[0], metric='euclidean'), X_train_resampled, y_train_resampled, cv=kf, method='predict_proba'), 'Euclidean')
plot_roc_curve(y_train_resampled, cross_val_predict(KNeighborsClassifier(
    n_neighbors=BEST_K[0], metric='cosine'), X_train_resampled, y_train_resampled, cv=kf, method='predict_proba'), 'Cosine')
