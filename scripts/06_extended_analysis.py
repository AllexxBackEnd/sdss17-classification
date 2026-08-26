"""Этап 6. Расширенные таблицы для текста работы.

Считает всё, что не входит в обязательный минимум раздела 8 ТЗ, но нужно для
содержательного описания данных и результатов: метрики по каждому классу,
описательные статистики, корреляции, цветовые индексы, ROC/PR-кривые
и нормированные версии перекрёстных таблиц.
"""

import sys

import numpy as np
import pandas as pd
from config import (
    CLASS_ORDER,
    FEATURES,
    PROCESSED_CSV,
    PROCESSED_ID_DTYPE,
    RESULTS_DIR,
    TARGET,
    ensure_dirs,
)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# Цветовые индексы — стандартный астрономический признак: разность звёздных величин
# в соседних фильтрах. Именно по ним объекты разных типов расходятся на диаграммах.
COLOR_INDICES = {
    "u_g": ("u", "g"),
    "g_r": ("g", "r"),
    "r_i": ("r", "i"),
    "i_z": ("i", "z"),
}
ROC_GRID_POINTS = 200


def per_class_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall/f1/support по каждому классу отдельно плюс macro- и weighted-строки."""
    y_true = predictions["true_class"]
    y_pred = predictions["predicted_class"]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    rows = [
        {
            "class": cls,
            "precision": precision[idx],
            "recall": recall[idx],
            "f1": f1[idx],
            "support": int(support[idx]),
        }
        for idx, cls in enumerate(CLASS_ORDER)
    ]

    for average in ("macro", "weighted"):
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        rows.append(
            {
                "class": f"{average} avg",
                "precision": avg_p,
                "recall": avg_r,
                "f1": avg_f1,
                "support": int(len(y_true)),
            }
        )
    return pd.DataFrame(rows).round(6)


def feature_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика каждого признака в разрезе классов и по выборке целиком."""
    rows = []
    groups = [(cls, data[data[TARGET] == cls]) for cls in CLASS_ORDER]
    groups.append(("ALL", data))

    for label, subset in groups:
        for feature in FEATURES:
            values = subset[feature]
            rows.append(
                {
                    "class": label,
                    "feature": feature,
                    "count": int(values.count()),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "q25": values.quantile(0.25),
                    "median": values.median(),
                    "q75": values.quantile(0.75),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows).round(6)


def color_stats(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Средние и разброс цветовых индексов по классам."""
    colors = pd.DataFrame({TARGET: data[TARGET]})
    for name, (left, right) in COLOR_INDICES.items():
        colors[name] = data[left] - data[right]

    rows = []
    for cls in [*CLASS_ORDER, "ALL"]:
        subset = colors if cls == "ALL" else colors[colors[TARGET] == cls]
        for name in COLOR_INDICES:
            rows.append(
                {
                    "class": cls,
                    "color_index": name,
                    "mean": subset[name].mean(),
                    "std": subset[name].std(),
                    "median": subset[name].median(),
                }
            )
    return pd.DataFrame(rows).round(6), colors


def roc_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ROC-AUC и average precision в схеме one-vs-rest плюс точки самих ROC-кривых."""
    y_true = predictions["true_class"]
    summary_rows = []
    curve_rows = []
    grid = np.linspace(0.0, 1.0, ROC_GRID_POINTS)

    for cls in CLASS_ORDER:
        binary = (y_true == cls).astype(int)
        scores = predictions[f"prob_{cls}"]
        summary_rows.append(
            {
                "class": cls,
                "roc_auc_ovr": roc_auc_score(binary, scores),
                "average_precision": average_precision_score(binary, scores),
                "positives": int(binary.sum()),
            }
        )
        fpr, tpr, _ = roc_curve(binary, scores)
        # Кривая прореживается на равномерную сетку: без этого файл разрастается
        # до десятков тысяч строк на класс и становится неудобным для перерисовки.
        curve_rows.append(
            pd.DataFrame({"class": cls, "fpr": grid, "tpr": np.interp(grid, fpr, tpr)})
        )

    return pd.DataFrame(summary_rows).round(6), pd.concat(curve_rows, ignore_index=True).round(6)


def normalized_cluster_table() -> pd.DataFrame:
    """Состав кластеров в долях — удобнее для текста, чем абсолютные числа."""
    table = pd.read_csv(RESULTS_DIR / "cluster_vs_class_table.csv")
    counts = table[["STAR_count", "GALAXY_count", "QSO_count"]]
    shares = counts.div(table["total"], axis=0)
    shares.columns = ["STAR_share", "GALAXY_share", "QSO_share"]
    return pd.concat(
        [table[["cluster_id", "total", "mapped_class"]], shares.round(6)], axis=1
    )


def approach_comparison() -> pd.DataFrame:
    """Сводит метрики обоих подходов в один вид «показатель — A — B» для итоговой таблицы."""
    metrics = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    lookup = {(row.approach, row.metric): row.value for row in metrics.itertuples()}

    rows = [
        ("Доля верных отнесений", lookup[("A", "accuracy")], lookup[("B", "purity")],
         "accuracy vs purity после majority vote"),
        ("Согласие с истинной разметкой", lookup[("A", "f1_macro")], lookup[("B", "ari")],
         "f1_macro vs ARI"),
        ("Время обучения, с", lookup[("A", "train_time_sec")], lookup[("B", "train_time_sec")], ""),
        ("Использует метки при обучении", 1.0, 0.0, "1 — да, 0 — нет"),
    ]
    return pd.DataFrame(rows, columns=["indicator", "approach_a", "approach_b", "note"]).round(6)


def main() -> int:
    ensure_dirs()
    predictions_path = RESULTS_DIR / "predictions_approach_a.csv"
    if not predictions_path.exists():
        print("Нет результатов Подхода A. Запустите скрипты 01-04.", file=sys.stderr)
        return 1

    data = pd.read_csv(PROCESSED_CSV, dtype=PROCESSED_ID_DTYPE)
    predictions = pd.read_csv(predictions_path, dtype=PROCESSED_ID_DTYPE)

    per_class_report(predictions).to_csv(
        RESULTS_DIR / "classification_report_a.csv", index=False, encoding="utf-8"
    )
    feature_stats(data).to_csv(
        RESULTS_DIR / "feature_stats_by_class.csv", index=False, encoding="utf-8"
    )

    correlation = data[FEATURES].corr().round(6)
    correlation.index.name = "feature"
    correlation.to_csv(RESULTS_DIR / "feature_correlation.csv", encoding="utf-8")

    colors_table, colors = color_stats(data)
    colors_table.to_csv(RESULTS_DIR / "color_indices_by_class.csv", index=False, encoding="utf-8")
    colors.to_csv(RESULTS_DIR / "color_indices.csv", index=False, encoding="utf-8")

    roc_summary, roc_curves = roc_tables(predictions)
    roc_summary.to_csv(RESULTS_DIR / "roc_auc_a.csv", index=False, encoding="utf-8")
    roc_curves.to_csv(RESULTS_DIR / "roc_curves_a.csv", index=False, encoding="utf-8")

    normalized_cluster_table().to_csv(
        RESULTS_DIR / "cluster_vs_class_normalized.csv", index=False, encoding="utf-8"
    )
    approach_comparison().to_csv(
        RESULTS_DIR / "approach_comparison.csv", index=False, encoding="utf-8"
    )

    print("Расширенные таблицы:")
    print(per_class_report(predictions).to_string(index=False))
    print()
    print(roc_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
