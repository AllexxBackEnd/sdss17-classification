"""Этап 5. Построение всех иллюстраций (PNG, 300 dpi) из сохранённых таблиц results/.

Скрипт не обращается к моделям — все графики строятся только по CSV,
поэтому их можно перерисовывать без повторного обучения.
"""

import sys

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from config import (  # noqa: E402
    CLASS_ORDER,
    FIGURE_DPI,
    FIGURES_DIR,
    N_CLUSTERS,
    RESULTS_DIR,
    ensure_dirs,
)

sns.set_theme(style="whitegrid")


def plot_confusion_matrix() -> None:
    """Матрица ошибок Подхода A: абсолютные числа и доля от истинного класса."""
    matrix = pd.read_csv(RESULTS_DIR / "confusion_matrix_a.csv", index_col="true_class")
    shares = matrix.div(matrix.sum(axis=1), axis=0)
    annotations = matrix.astype(str) + "\n" + (shares * 100).round(1).astype(str) + "%"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(shares, annot=annotations, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Подход A (Random Forest): матрица ошибок на test-выборке")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_a.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_pca() -> None:
    """Одна и та же PCA-проекция, раскрашенная по истинному классу и по кластеру K-Means."""
    projection = pd.read_csv(RESULTS_DIR / "pca_projection.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)

    for cls in CLASS_ORDER:
        subset = projection[projection["true_class"] == cls]
        axes[0].scatter(subset["pc1"], subset["pc2"], s=2, alpha=0.3, label=cls, rasterized=True)
    axes[0].set_title("Истинные классы (SDSS17)")

    for cluster in range(N_CLUSTERS):
        subset = projection[projection["cluster_id"] == cluster]
        axes[1].scatter(
            subset["pc1"],
            subset["pc2"],
            s=2,
            alpha=0.3,
            label=f"кластер {cluster}",
            rasterized=True,
        )
    axes[1].set_title(f"Кластеры K-Means (k={N_CLUSTERS})")

    for ax in axes:
        ax.set_xlabel("PC1")
        ax.legend(markerscale=6, framealpha=0.9)
    axes[0].set_ylabel("PC2")

    fig.suptitle("PCA-проекция признаков u, g, r, i, z, redshift")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_true_class_vs_cluster.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_elbow() -> None:
    """Elbow-график инерции и silhouette score для обоснования выбора k."""
    scan = pd.read_csv(RESULTS_DIR / "elbow_silhouette.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(scan["k"], scan["inertia"], marker="o")
    axes[0].set_ylabel("Инерция (within-cluster SSE)")
    axes[0].set_title("Метод локтя")

    axes[1].plot(scan["k"], scan["silhouette_score"], marker="o", color="tab:orange")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette score")

    for ax in axes:
        ax.set_xlabel("Число кластеров k")
        ax.axvline(N_CLUSTERS, color="grey", linestyle="--", linewidth=1)

    fig.suptitle("Подход B: выбор числа кластеров (пунктир — k = 3)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "elbow_plot.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_feature_importance() -> None:
    """Важность признаков Random Forest."""
    importance = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=importance, x="importance", y="feature", hue="feature", legend=False, ax=ax)
    for index, value in enumerate(importance["importance"]):
        ax.text(value + 0.008, index, f"{value:.3f}", va="center")
    ax.set_xlim(0, importance["importance"].max() * 1.18)
    ax.set_xlabel("Важность признака (Gini importance)")
    ax.set_ylabel("Признак")
    ax.set_title("Подход A: важность признаков")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=FIGURE_DPI)
    plt.close(fig)


def main() -> int:
    ensure_dirs()
    if not (RESULTS_DIR / "metrics_summary.csv").exists():
        print("Нет результатов. Запустите скрипты 01-04.", file=sys.stderr)
        return 1

    plot_confusion_matrix()
    plot_pca()
    plot_elbow()
    plot_feature_importance()
    print(f"Графики сохранены в {FIGURES_DIR} ({FIGURE_DPI} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
