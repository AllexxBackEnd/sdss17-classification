"""Этап 3. Подход B — кластеризация без учителя (K-Means, k=3).

Результат: models/kmeans_model.pkl, results/predictions_approach_b.csv,
results/elbow_silhouette.csv, results/pca_projection.csv.
"""

import sys
import time

import joblib
import numpy as np
import pandas as pd
from config import (
    FEATURES,
    K_RANGE,
    KMEANS_N_INIT,
    KMEANS_PKL,
    N_CLUSTERS,
    PROCESSED_CSV,
    PROCESSED_ID_DTYPE,
    RANDOM_STATE,
    RESULTS_DIR,
    SCALER_PKL,
    SILHOUETTE_SAMPLE_SIZE,
    TARGET,
    ensure_dirs,
)
from meta import update_meta
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def scan_k(features: np.ndarray) -> pd.DataFrame:
    """Считает инерцию и silhouette score для каждого k из K_RANGE (обоснование выбора k)."""
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        labels = model.fit_predict(features)
        score = silhouette_score(
            features, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=RANDOM_STATE
        )
        # Инерция округляется: порядок редукции в BLAS плавает между запусками
        # и меняет последние знаки, из-за чего файл переставал быть побайтово стабильным.
        rows.append(
            {"k": k, "inertia": round(float(model.inertia_), 6), "silhouette_score": float(score)}
        )
        print(f"  k={k}: inertia={model.inertia_:.1f}, silhouette={score:.4f}")
    return pd.DataFrame(rows)


def map_clusters(frame: pd.DataFrame) -> dict[int, str]:
    """Сопоставляет каждому кластеру класс по большинству истинных меток (majority vote)."""
    majority = frame.groupby("cluster_id")[TARGET].agg(lambda s: s.value_counts().idxmax())
    return {int(cluster): str(cls) for cluster, cls in majority.items()}


def main() -> int:
    ensure_dirs()

    if not PROCESSED_CSV.exists():
        print(f"Нет {PROCESSED_CSV}. Сначала запустите scripts/01_preprocess.py", file=sys.stderr)
        return 1

    data = pd.read_csv(PROCESSED_CSV, dtype=PROCESSED_ID_DTYPE)
    scaler = joblib.load(SCALER_PKL)
    features = scaler.transform(data[FEATURES])

    print(f"Подбор k на {len(data)} объектах (метки не используются):")
    scan = scan_k(features)
    scan.to_csv(RESULTS_DIR / "elbow_silhouette.csv", index=False, encoding="utf-8")

    model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    fit_start = time.perf_counter()
    cluster_ids = model.fit_predict(features)
    train_time = time.perf_counter() - fit_start
    joblib.dump(model, KMEANS_PKL)

    predictions = pd.DataFrame(
        {
            "obj_id": data["obj_id"].to_numpy(),
            "true_class": data[TARGET].to_numpy(),
            "cluster_id": cluster_ids,
        }
    )
    mapping = map_clusters(predictions.rename(columns={"true_class": TARGET}))
    predictions["mapped_class"] = predictions["cluster_id"].map(mapping)
    predictions.to_csv(RESULTS_DIR / "predictions_approach_b.csv", index=False, encoding="utf-8")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(features)
    pd.DataFrame(
        {
            "obj_id": data["obj_id"].to_numpy(),
            "pc1": components[:, 0],
            "pc2": components[:, 1],
            "true_class": data[TARGET].to_numpy(),
            "cluster_id": cluster_ids,
        }
    ).to_csv(RESULTS_DIR / "pca_projection.csv", index=False, encoding="utf-8")

    update_meta(
        "approach_b",
        {
            "n_clusters": N_CLUSTERS,
            "n_init": KMEANS_N_INIT,
            "train_time_sec": round(train_time, 3),
            "n_samples": int(len(data)),
            "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
            "cluster_to_class": {str(k): v for k, v in mapping.items()},
            "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
            "pca_explained_variance_total": float(pca.explained_variance_ratio_.sum()),
        },
    )

    print(f"\nk=3 обучен за {train_time:.2f} с, majority-vote: {mapping}")
    print(f"PCA объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
