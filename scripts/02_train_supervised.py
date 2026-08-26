"""Этап 2. Подход A — классификация с учителем (Random Forest).

Результат: models/random_forest_model.pkl, results/predictions_approach_a.csv,
results/feature_importance.csv.
"""

import sys
import time

import joblib
import pandas as pd
from config import (
    FEATURES,
    PROCESSED_CSV,
    PROCESSED_ID_DTYPE,
    RANDOM_STATE,
    RESULTS_DIR,
    RF_PKL,
    SCALER_PKL,
    TARGET,
    ensure_dirs,
)
from meta import update_meta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
}
CV_FOLDS = 5


def main() -> int:
    ensure_dirs()

    if not PROCESSED_CSV.exists():
        print(f"Нет {PROCESSED_CSV}. Сначала запустите scripts/01_preprocess.py", file=sys.stderr)
        return 1

    data = pd.read_csv(PROCESSED_CSV, dtype=PROCESSED_ID_DTYPE)
    scaler = joblib.load(SCALER_PKL)

    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]

    x_train = scaler.transform(train[FEATURES])
    x_test = scaler.transform(test[FEATURES])
    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        PARAM_GRID,
        cv=CV_FOLDS,
        scoring="f1_macro",
        n_jobs=-1,
    )

    print(f"GridSearchCV: {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth'])} "
          f"комбинаций x {CV_FOLDS} фолдов...")
    search_start = time.perf_counter()
    search.fit(x_train, y_train)
    search_time = time.perf_counter() - search_start

    # Финальная модель переобучается отдельно, чтобы измерить чистое время обучения
    # лучшей конфигурации (метрика train_time_sec из раздела 7 ТЗ).
    model = RandomForestClassifier(**search.best_params_, random_state=RANDOM_STATE, n_jobs=-1)
    fit_start = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - fit_start

    joblib.dump(model, RF_PKL)

    probabilities = model.predict_proba(x_test)
    prob_by_class = {cls: probabilities[:, idx] for idx, cls in enumerate(model.classes_)}

    predictions = pd.DataFrame(
        {
            "obj_id": test["obj_id"].to_numpy(),
            "true_class": y_test,
            "predicted_class": model.predict(x_test),
            "prob_STAR": prob_by_class["STAR"],
            "prob_GALAXY": prob_by_class["GALAXY"],
            "prob_QSO": prob_by_class["QSO"],
        }
    )
    # Вероятности округляются: усреднение по деревьям идёт в несколько потоков, порядок
    # суммирования плавает между запусками и меняет последние биты (расхождение ~5e-16).
    # На метки и метрики это не влияет, но без округления файл не побайтово стабилен.
    predictions[["prob_STAR", "prob_GALAXY", "prob_QSO"]] = predictions[
        ["prob_STAR", "prob_GALAXY", "prob_QSO"]
    ].round(6)
    predictions.to_csv(RESULTS_DIR / "predictions_approach_a.csv", index=False, encoding="utf-8")

    # Полная таблица перебора: в работе она обосновывает выбор гиперпараметров,
    # а не только называет победившую конфигурацию.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_columns = [
        "param_n_estimators", "param_max_depth", "mean_test_score",
        "std_test_score", "rank_test_score", "mean_fit_time",
    ]
    cv_results[cv_columns].sort_values("rank_test_score").to_csv(
        RESULTS_DIR / "gridsearch_cv_results.csv", index=False, encoding="utf-8"
    )

    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False, encoding="utf-8")

    update_meta(
        "approach_a",
        {
            "best_params": search.best_params_,
            "cv_folds": CV_FOLDS,
            "cv_scoring": "f1_macro",
            "cv_best_score": float(search.best_score_),
            "cv_search_time_sec": round(search_time, 3),
            "train_time_sec": round(train_time, 3),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
        },
    )

    print(f"best_params: {search.best_params_}")
    print(f"cv f1_macro: {search.best_score_:.4f}")
    print(f"поиск: {search_time:.1f} с, финальное обучение: {train_time:.2f} с")
    print(importance.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
