"""Этап 7. Контрольные эксперименты с разными наборами признаков.

Проверяет два тезиса, заявленных в ТЗ:
  * координаты (alpha, delta) дают завышенную и физически бессмысленную точность
    и потому исключены из рабочего набора (раздел 3 и риск из раздела 12);
  * насколько задача решается без redshift — признака, на который у Random Forest
    приходится две трети важности.

Оба подхода (RF и K-Means) прогоняются на каждом наборе признаков в одинаковых условиях.
Результат: results/control_experiments.csv.
"""

import sys
import time

import pandas as pd
from config import (
    KMEANS_N_INIT,
    N_CLUSTERS,
    PROCESSED_CSV,
    PROCESSED_ID_DTYPE,
    RANDOM_STATE,
    RAW_CSV,
    RAW_ID_DTYPE,
    RESULTS_DIR,
    SILHOUETTE_SAMPLE_SIZE,
    TARGET,
    ensure_dirs,
)
from meta import load_meta
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

MAGNITUDES = ["u", "g", "r", "i", "z"]
COLORS = ["u_g", "g_r", "r_i", "i_z"]

EXPERIMENTS: dict[str, list[str]] = {
    "baseline": [*MAGNITUDES, "redshift"],
    "with_coords": [*MAGNITUDES, "redshift", "alpha", "delta"],
    "coords_only": ["alpha", "delta"],
    "no_redshift": MAGNITUDES,
    "redshift_only": ["redshift"],
    "colors_and_r": [*COLORS, "r"],
    "colors_and_redshift": [*COLORS, "redshift"],
}

DESCRIPTIONS = {
    "baseline": "рабочий набор признаков (Подход A и B)",
    "with_coords": "рабочий набор + координаты на небесной сфере",
    "coords_only": "только координаты — контроль на утечку через геометрию обзора",
    "no_redshift": "только фотометрия, без красного смещения",
    "redshift_only": "только красное смещение",
    "colors_and_r": "цветовые индексы + звёздная величина r, без redshift",
    "colors_and_redshift": "цветовые индексы + красное смещение",
}


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Дополняет рабочую таблицу координатами из исходника и цветовыми индексами."""
    raw = pd.read_csv(RAW_CSV, dtype=RAW_ID_DTYPE)
    # spec_obj_id — единственный уникальный ключ строки: obj_ID в датасете повторяется.
    coords = raw[["spec_obj_ID", "alpha", "delta"]].rename(columns={"spec_obj_ID": "spec_obj_id"})
    frame = data.merge(coords, on="spec_obj_id", how="left", validate="one_to_one")

    frame["u_g"] = frame["u"] - frame["g"]
    frame["g_r"] = frame["g"] - frame["r"]
    frame["r_i"] = frame["r"] - frame["i"]
    frame["i_z"] = frame["i"] - frame["z"]
    return frame


def run_experiment(name: str, features: list[str], frame: pd.DataFrame, rf_params: dict) -> dict:
    """Обучает RF на train и K-Means на всём наборе для одного набора признаков."""
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    scaler = StandardScaler().fit(train[features])
    x_train = scaler.transform(train[features])
    x_test = scaler.transform(test[features])
    x_all = scaler.transform(frame[features])

    forest = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
    rf_start = time.perf_counter()
    forest.fit(x_train, train[TARGET])
    rf_time = time.perf_counter() - rf_start
    predicted = forest.predict(x_test)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    km_start = time.perf_counter()
    clusters = kmeans.fit_predict(x_all)
    km_time = time.perf_counter() - km_start

    contingency = pd.crosstab(clusters, frame[TARGET])
    purity = float(contingency.max(axis=1).sum() / len(frame))

    return {
        "experiment": name,
        "description": DESCRIPTIONS[name],
        "features": " ".join(features),
        "n_features": len(features),
        "rf_accuracy": round(accuracy_score(test[TARGET], predicted), 6),
        "rf_f1_macro": round(f1_score(test[TARGET], predicted, average="macro"), 6),
        "rf_train_time_sec": round(rf_time, 3),
        "kmeans_ari": round(adjusted_rand_score(frame[TARGET], clusters), 6),
        "kmeans_nmi": round(normalized_mutual_info_score(frame[TARGET], clusters), 6),
        "kmeans_purity": round(purity, 6),
        "kmeans_silhouette": round(
            float(
                silhouette_score(
                    x_all,
                    clusters,
                    sample_size=SILHOUETTE_SAMPLE_SIZE,
                    random_state=RANDOM_STATE,
                )
            ),
            6,
        ),
        "kmeans_train_time_sec": round(km_time, 3),
    }


LEARNING_CURVE_FRACTIONS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
LEARNING_CURVE_FOLDS = 3


def learning_curve(frame: pd.DataFrame, features: list[str], rf_params: dict) -> pd.DataFrame:
    """Оценивает f1_macro при разных объёмах обучающей выборки — хватило ли данных."""
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    scaler = StandardScaler().fit(train[features])
    x_test = scaler.transform(test[features])

    rows = []
    for fraction in LEARNING_CURVE_FRACTIONS:
        # Подвыборка стратифицирована по классу, иначе на малых долях редкие
        # классы вырождаются и f1_macro падает по совершенно другой причине.
        subset = train.groupby(TARGET, group_keys=False).sample(
            frac=fraction, random_state=RANDOM_STATE
        )
        x_subset = scaler.transform(subset[features])
        y_subset = subset[TARGET]

        forest = RandomForestClassifier(**rf_params, random_state=RANDOM_STATE, n_jobs=-1)
        scores = cross_val_score(
            forest, x_subset, y_subset, cv=LEARNING_CURVE_FOLDS, scoring="f1_macro", n_jobs=-1
        )
        forest.fit(x_subset, y_subset)

        rows.append(
            {
                "n_train": int(len(subset)),
                "fraction": fraction,
                "cv_f1_macro_mean": round(float(scores.mean()), 6),
                "cv_f1_macro_std": round(float(scores.std()), 6),
                "test_f1_macro": round(
                    f1_score(test[TARGET], forest.predict(x_test), average="macro"), 6
                ),
            }
        )
        print(f"  n_train={len(subset)}: cv f1_macro={scores.mean():.4f}")
    return pd.DataFrame(rows)


def main() -> int:
    ensure_dirs()
    meta = load_meta()
    if "approach_a" not in meta:
        print("Нет данных обучения. Запустите скрипт 02.", file=sys.stderr)
        return 1

    # Гиперпараметры берутся из основного GridSearchCV: эксперименты должны отличаться
    # только составом признаков, иначе сравнение перестаёт быть контрольным.
    rf_params = dict(meta["approach_a"]["best_params"])

    data = pd.read_csv(PROCESSED_CSV, dtype=PROCESSED_ID_DTYPE)
    frame = build_feature_frame(data)

    rows = []
    for name, features in EXPERIMENTS.items():
        print(f"  {name}: {len(features)} признаков...")
        rows.append(run_experiment(name, features, frame, rf_params))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "control_experiments.csv", index=False, encoding="utf-8")

    print("Кривая обучения:")
    curve = learning_curve(frame, EXPERIMENTS["baseline"], rf_params)
    curve.to_csv(RESULTS_DIR / "learning_curve.csv", index=False, encoding="utf-8")

    columns = ["experiment", "n_features", "rf_accuracy", "rf_f1_macro", "kmeans_ari",
               "kmeans_purity"]
    print()
    print(table[columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
