import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# funções para plotagem dos gráficos


def plot_tsne(data, hue_column):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x="tsne_x", y="tsne_y",
                    hue=data[hue_column], palette="tab10", alpha=0.7)

    plt.title("embeddings com t-SNE")
    plt.legend(loc="best", bbox_to_anchor=(1, 1))
    plt.show()


def plot_bar(val, title, xlabel, ylabel, color, rotation=45):
    plt.figure(figsize=(12, 6))
    val.value_counts().plot(kind="bar", color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation)
    plt.show()


def plot_table_metrics(auc_euclidean, auc_cosine, f1_euclidean, f1_cosine, acc_euclidean, acc_cosine):

    data = pd.DataFrame({
        "Metric": ["Euclidean", "Cosine"],
        "AUC": [auc_euclidean, auc_cosine],
        "F1-SCORE": [f1_euclidean, f1_cosine],
        "Accuracy": [acc_euclidean, acc_cosine]
    })

    plt.figure(figsize=(12, 6))

    data.set_index("Metric").plot(kind="bar", color=[
        "orange", "blue", "green"], width=0.8, ax=plt.gca())

    plt.title("Comparison between euclidean and cosine metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="Metrics", loc="upper left", bbox_to_anchor=(1, 1))

    plt.show()


def plot_roc_curve(y_true, y_proba, metric):

    n_classes = y_true.nunique()

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_proba[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=1, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=0.8)

    plt.title(f'ROC Curve for {metric} Metric')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")

    plt.show()
