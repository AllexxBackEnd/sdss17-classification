"""Этап 8. Дополнительные иллюстрации к работе.

Как и этап 5, строится только по сохранённым CSV — модели не загружаются.
"""

import sys

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from config import (  # noqa: E402
    CLASS_ORDER,
    FEATURES,
    FIGURE_DPI,
    FIGURES_DIR,
    N_CLUSTERS,
    PROCESSED_CSV,
    PROCESSED_ID_DTYPE,
    RESULTS_DIR,
    TARGET,
    ensure_dirs,
)

sns.set_theme(style="whitegrid")
CLASS_COLORS = {"GALAXY": "tab:blue", "QSO": "tab:orange", "STAR": "tab:green"}


def save(fig: plt.Figure, name: str) -> None:
    """Сохраняет фигуру в figures/ и закрывает её."""
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=FIGURE_DPI)
    plt.close(fig)
    print(f"  {name}")


def plot_class_distribution(data: pd.DataFrame) -> None:
    """Баланс классов — обязательная иллюстрация к разделу о данных."""
    counts = data[TARGET].value_counts().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(counts.index, counts.to_numpy(), color=[CLASS_COLORS[c] for c in counts.index])
    for bar, value in zip(bars, counts.to_numpy(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + len(data) * 0.01,
            f"{value}\n{value / len(data) * 100:.1f}%",
            ha="center",
        )
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_ylabel("Число объектов")
    ax.set_title(f"Распределение классов (всего {len(data)} объектов)")
    save(fig, "class_distribution.png")


def plot_feature_distributions(data: pd.DataFrame) -> None:
    """Гистограммы всех шести признаков в разрезе классов."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.flat, FEATURES, strict=True):
        # Хвосты обрезаются по перцентилям, иначе редкие выбросы сплющивают всю гистограмму.
        low, high = data[feature].quantile([0.001, 0.999])
        for cls in CLASS_ORDER:
            values = data.loc[data[TARGET] == cls, feature]
            ax.hist(
                values, bins=80, range=(low, high), density=True,
                alpha=0.55, label=cls, color=CLASS_COLORS[cls],
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("Плотность")
    axes.flat[0].legend()
    fig.suptitle("Распределения признаков по классам")
    save(fig, "feature_distributions.png")


def plot_color_diagram(colors: pd.DataFrame) -> None:
    """Классическая цвет-цветовая диаграмма: по ней видно физику разделения типов."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cls in CLASS_ORDER:
        subset = colors[colors[TARGET] == cls]
        ax.scatter(
            subset["g_r"], subset["u_g"], s=2, alpha=0.25,
            label=cls, color=CLASS_COLORS[cls], rasterized=True,
        )
    ax.set_xlim(colors["g_r"].quantile(0.005), colors["g_r"].quantile(0.995))
    ax.set_ylim(colors["u_g"].quantile(0.005), colors["u_g"].quantile(0.995))
    ax.set_xlabel("g − r")
    ax.set_ylabel("u − g")
    ax.set_title("Цвет-цветовая диаграмма")
    ax.legend(markerscale=8)
    save(fig, "color_color_diagram.png")


def plot_redshift(data: pd.DataFrame) -> None:
    """Распределение красного смещения — ключевого признака — по классам."""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for cls in CLASS_ORDER:
        values = data.loc[data[TARGET] == cls, "redshift"]
        ax.hist(values, bins=120, range=(-0.1, 3.5), alpha=0.55,
                label=cls, color=CLASS_COLORS[cls])
    ax.set_yscale("log")
    ax.set_xlabel("redshift")
    ax.set_ylabel("Число объектов (лог. шкала)")
    ax.set_title("Красное смещение по классам")
    ax.legend()
    save(fig, "redshift_by_class.png")


def plot_correlation() -> None:
    """Матрица корреляций признаков."""
    corr = pd.read_csv(RESULTS_DIR / "feature_correlation.csv", index_col="feature")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Корреляции признаков (Пирсон)")
    save(fig, "correlation_heatmap.png")


def plot_roc() -> None:
    """ROC-кривые Подхода A в схеме one-vs-rest."""
    curves = pd.read_csv(RESULTS_DIR / "roc_curves_a.csv")
    auc = pd.read_csv(RESULTS_DIR / "roc_auc_a.csv").set_index("class")["roc_auc_ovr"]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for cls in CLASS_ORDER:
        subset = curves[curves["class"] == cls]
        ax.plot(subset["fpr"], subset["tpr"], label=f"{cls} (AUC = {auc[cls]:.4f})",
                color=CLASS_COLORS[cls])
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="случайное угадывание")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Подход A: ROC-кривые (one-vs-rest)")
    ax.legend(loc="lower right")
    save(fig, "roc_curves_a.png")


def plot_cluster_composition() -> None:
    """Состав каждого кластера в долях истинных классов."""
    table = pd.read_csv(RESULTS_DIR / "cluster_vs_class_normalized.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bottom = np.zeros(len(table))
    for cls in CLASS_ORDER:
        values = table[f"{cls}_share"].to_numpy()
        ax.bar(table["cluster_id"].astype(str), values, bottom=bottom,
               label=cls, color=CLASS_COLORS[cls])
        bottom += values
    for idx, row in table.iterrows():
        ax.text(idx, 1.02, f"n = {row['total']}\n→ {row['mapped_class']}", ha="center")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Кластер K-Means")
    ax.set_ylabel("Доля класса внутри кластера")
    ax.set_title("Подход B: состав кластеров")
    ax.legend(loc="lower right")
    save(fig, "cluster_composition.png")


def plot_pca_centroids() -> None:
    """PCA-проекция с центроидами кластеров.

    Центроид считается как среднее проекций точек кластера: PCA — линейное
    преобразование, поэтому это в точности проекция центроида из пространства признаков.
    """
    projection = pd.read_csv(RESULTS_DIR / "pca_projection.csv", dtype=PROCESSED_ID_DTYPE)
    centroids = projection.groupby("cluster_id")[["pc1", "pc2"]].mean()

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cluster in range(N_CLUSTERS):
        subset = projection[projection["cluster_id"] == cluster]
        ax.scatter(subset["pc1"], subset["pc2"], s=2, alpha=0.25,
                   label=f"кластер {cluster}", rasterized=True)
    ax.scatter(centroids["pc1"], centroids["pc2"], s=320, marker="X",
               c="black", edgecolors="white", linewidths=2, zorder=5, label="центроиды")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Подход B: кластеры и их центроиды в пространстве PCA")
    ax.legend(markerscale=4)
    save(fig, "pca_with_centroids.png")


def plot_approach_comparison() -> None:
    """Столбиковое сравнение ключевых метрик обоих подходов."""
    metrics = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    lookup = {(row.approach, row.metric): row.value for row in metrics.itertuples()}
    labels = ["accuracy /\npurity", "f1_macro /\nARI", "NMI", "silhouette"]
    values_a = [lookup[("A", "accuracy")], lookup[("A", "f1_macro")], np.nan, np.nan]
    values_b = [
        lookup[("B", "purity")], lookup[("B", "ari")],
        lookup[("B", "nmi")], lookup[("B", "silhouette")],
    ]

    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(positions - 0.2, values_a, 0.4, label="Подход A (Random Forest)", color="tab:blue")
    ax.bar(positions + 0.2, values_b, 0.4, label="Подход B (K-Means)", color="tab:red")
    for pos, value in zip(positions - 0.2, values_a, strict=True):
        if not np.isnan(value):
            ax.text(pos, value + 0.015, f"{value:.3f}", ha="center")
    for pos, value in zip(positions + 0.2, values_b, strict=True):
        ax.text(pos, value + 0.015, f"{value:.3f}", ha="center")
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Сравнение подходов")
    ax.legend()
    save(fig, "approach_comparison.png")


def plot_control_experiments() -> None:
    """Результаты контрольных экспериментов с разными наборами признаков."""
    table = pd.read_csv(RESULTS_DIR / "control_experiments.csv")
    positions = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(positions - 0.2, table["rf_f1_macro"], 0.4, label="Random Forest, f1_macro",
           color="tab:blue")
    ax.bar(positions + 0.2, table["kmeans_ari"], 0.4, label="K-Means, ARI", color="tab:red")
    for pos, value in zip(positions - 0.2, table["rf_f1_macro"], strict=True):
        ax.text(pos, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    for pos, value in zip(positions + 0.2, table["kmeans_ari"], strict=True):
        ax.text(pos, value + 0.015, f"{value:.3f}", ha="center", fontsize=8)
    ax.set_xticks(positions, table["experiment"], rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Значение метрики")
    ax.set_title("Контрольные эксперименты: влияние состава признаков")
    ax.legend()
    save(fig, "control_experiments.png")


def plot_learning_curve() -> None:
    """Кривая обучения: хватило ли объёма выборки."""
    curve = pd.read_csv(RESULTS_DIR / "learning_curve.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(curve["n_train"], curve["cv_f1_macro_mean"], marker="o", label="кросс-валидация")
    ax.fill_between(
        curve["n_train"],
        curve["cv_f1_macro_mean"] - curve["cv_f1_macro_std"],
        curve["cv_f1_macro_mean"] + curve["cv_f1_macro_std"],
        alpha=0.2,
    )
    ax.plot(curve["n_train"], curve["test_f1_macro"], marker="s", label="отложенная test-выборка")
    ax.set_xlabel("Размер обучающей выборки")
    ax.set_ylabel("f1_macro")
    ax.set_title("Подход A: кривая обучения")
    ax.legend()
    save(fig, "learning_curve.png")


def main() -> int:
    ensure_dirs()
    if not (RESULTS_DIR / "control_experiments.csv").exists():
        print("Нет расширенных таблиц. Запустите скрипты 06 и 07.", file=sys.stderr)
        return 1

    data = pd.read_csv(PROCESSED_CSV, dtype=PROCESSED_ID_DTYPE)
    colors = pd.read_csv(RESULTS_DIR / "color_indices.csv")

    print("Дополнительные графики:")
    plot_class_distribution(data)
    plot_feature_distributions(data)
    plot_color_diagram(colors)
    plot_redshift(data)
    plot_correlation()
    plot_roc()
    plot_cluster_composition()
    plot_pca_centroids()
    plot_approach_comparison()
    plot_control_experiments()
    plot_learning_curve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
