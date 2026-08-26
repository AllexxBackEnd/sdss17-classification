"""Этап 4. Оценка обоих подходов и сведение всех метрик в единые таблицы.

Результат: results/metrics_summary.csv, results/confusion_matrix_a.csv,
results/cluster_vs_class_table.csv, results/run_config.json.
"""

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
import sklearn
from config import (
    CLASS_ORDER,
    FEATURES,
    KMEANS_N_INIT,
    N_CLUSTERS,
    RANDOM_STATE,
    RESULTS_DIR,
    SILHOUETTE_SAMPLE_SIZE,
    TEST_SIZE,
    ensure_dirs,
)
from meta import load_meta
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)


def evaluate_a(predictions: pd.DataFrame, train_time: float) -> tuple[list[dict], pd.DataFrame]:
    """Считает метрики Подхода A и матрицу ошибок 3x3 (строки — истина, столбцы — предсказание)."""
    y_true = predictions["true_class"]
    y_pred = predictions["predicted_class"]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    rows = [
        {"approach": "A", "metric": "accuracy", "value": accuracy_score(y_true, y_pred)},
        {"approach": "A", "metric": "precision_macro", "value": precision},
        {"approach": "A", "metric": "recall_macro", "value": recall},
        {"approach": "A", "metric": "f1_macro", "value": f1},
        {"approach": "A", "metric": "train_time_sec", "value": train_time},
    ]

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    frame = pd.DataFrame(matrix, index=CLASS_ORDER, columns=CLASS_ORDER)
    frame.index.name = "true_class"
    return rows, frame


def cluster_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Строит перекрёстную таблицу кластер x класс с majority-меткой и чистотой кластера."""
    contingency = pd.crosstab(predictions["cluster_id"], predictions["true_class"])
    for cls in CLASS_ORDER:
        if cls not in contingency.columns:
            contingency[cls] = 0

    table = pd.DataFrame(
        {
            "cluster_id": contingency.index,
            "STAR_count": contingency["STAR"].to_numpy(),
            "GALAXY_count": contingency["GALAXY"].to_numpy(),
            "QSO_count": contingency["QSO"].to_numpy(),
        }
    )
    table["total"] = table[["STAR_count", "GALAXY_count", "QSO_count"]].sum(axis=1)
    table["mapped_class"] = contingency[CLASS_ORDER].idxmax(axis=1).to_numpy()
    table["purity_of_cluster"] = (
        contingency[CLASS_ORDER].max(axis=1).to_numpy() / table["total"].to_numpy()
    )
    return table


def evaluate_b(
    predictions: pd.DataFrame, table: pd.DataFrame, silhouette: float, train_time: float
) -> list[dict]:
    """Считает ARI, NMI, purity и собирает метрики Подхода B."""
    y_true = predictions["true_class"]
    clusters = predictions["cluster_id"]
    purity = float(
        (table["purity_of_cluster"] * table["total"]).sum() / table["total"].sum()
    )
    return [
        {"approach": "B", "metric": "ari", "value": adjusted_rand_score(y_true, clusters)},
        {"approach": "B", "metric": "nmi", "value": normalized_mutual_info_score(y_true, clusters)},
        {"approach": "B", "metric": "purity", "value": purity},
        {"approach": "B", "metric": "silhouette", "value": silhouette},
        {"approach": "B", "metric": "train_time_sec", "value": train_time},
    ]


def main() -> int:
    ensure_dirs()
    meta = load_meta()
    if "approach_a" not in meta or "approach_b" not in meta:
        print("Нет данных обучения. Запустите скрипты 02 и 03.", file=sys.stderr)
        return 1

    pred_a = pd.read_csv(RESULTS_DIR / "predictions_approach_a.csv")
    pred_b = pd.read_csv(RESULTS_DIR / "predictions_approach_b.csv")
    scan = pd.read_csv(RESULTS_DIR / "elbow_silhouette.csv")

    rows_a, matrix_a = evaluate_a(pred_a, meta["approach_a"]["train_time_sec"])
    matrix_a.to_csv(RESULTS_DIR / "confusion_matrix_a.csv", encoding="utf-8")

    table = cluster_table(pred_b)
    table.to_csv(RESULTS_DIR / "cluster_vs_class_table.csv", index=False, encoding="utf-8")

    silhouette = float(scan.loc[scan["k"] == N_CLUSTERS, "silhouette_score"].iloc[0])
    rows_b = evaluate_b(pred_b, table, silhouette, meta["approach_b"]["train_time_sec"])

    rows_common = [
        {"approach": "common", "metric": "train_size", "value": meta["approach_a"]["n_train"]},
        {"approach": "common", "metric": "test_size", "value": meta["approach_a"]["n_test"]},
        {
            "approach": "common",
            "metric": "n_samples_total",
            "value": meta["approach_b"]["n_samples"],
        },
    ]

    summary = pd.DataFrame(rows_a + rows_b + rows_common)
    summary["value"] = summary["value"].astype(float).round(6)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False, encoding="utf-8")

    run_config = {
        "run_date": date.today().isoformat(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "features": FEATURES,
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "random_forest": {
            **meta["approach_a"]["best_params"],
            "cv_folds": meta["approach_a"]["cv_folds"],
            "cv_scoring": meta["approach_a"]["cv_scoring"],
            "cv_best_score": meta["approach_a"]["cv_best_score"],
        },
        "kmeans": {
            "n_clusters": N_CLUSTERS,
            "n_init": KMEANS_N_INIT,
            "random_state": RANDOM_STATE,
            "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
            "cluster_to_class": meta["approach_b"]["cluster_to_class"],
        },
        "pca_explained_variance_ratio": meta["approach_b"]["pca_explained_variance_ratio"],
    }
    (RESULTS_DIR / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print()
    print(matrix_a.to_string())
    print()
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
