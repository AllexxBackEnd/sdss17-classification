"""Этап 1. EDA, очистка, отбор признаков, масштабирование и разбиение train/test.

Результат: data/processed.csv, models/scaler.pkl, results/dataset_report.json.
"""

import json
import sys

import joblib
import pandas as pd
from config import (
    CLASS_ORDER,
    FEATURES,
    MAG_MAX,
    MAG_MIN,
    PROCESSED_CSV,
    RANDOM_STATE,
    RAW_CSV,
    RAW_ID_DTYPE,
    REDSHIFT_MAX,
    REDSHIFT_MIN,
    RESULTS_DIR,
    SCALER_PKL,
    TARGET,
    TEST_SIZE,
    ensure_dirs,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MAG_COLUMNS = ["u", "g", "r", "i", "z"]


def clean(frame: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Удаляет дубликаты и физически невозможные значения, записывая счётчики в report."""
    report["n_raw"] = int(len(frame))
    report["n_missing_cells"] = int(frame[FEATURES + [TARGET]].isna().sum().sum())

    frame = frame.dropna(subset=FEATURES + [TARGET])

    # Единица наблюдения — спектр: именно по нему определён класс, и spec_obj_ID уникален.
    # obj_ID в датасете ключом не является: у строк с одинаковым obj_ID различаются
    # фотометрия, класс и красное смещение, поэтому дедупликация по нему удалила бы
    # ~19 тыс. полноценных наблюдений.
    report["n_nonunique_obj_id"] = int(frame["obj_ID"].duplicated().sum())

    n_before = len(frame)
    frame = frame.drop_duplicates(subset="spec_obj_ID", keep="first")
    report["n_duplicate_spec_obj_id"] = int(n_before - len(frame))

    # В SDSS отсутствующая фотометрия кодируется значением -9999; кроме того
    # отбрасываем звёздные величины и красные смещения вне физически осмысленных границ.
    mag_ok = frame[MAG_COLUMNS].ge(MAG_MIN).all(axis=1) & frame[MAG_COLUMNS].le(MAG_MAX).all(axis=1)
    redshift_ok = frame["redshift"].between(REDSHIFT_MIN, REDSHIFT_MAX)

    report["n_bad_photometry"] = int((~mag_ok).sum())
    report["n_bad_redshift"] = int((~redshift_ok).sum())

    frame = frame[mag_ok & redshift_ok]
    report["n_clean"] = int(len(frame))
    return frame


def main() -> int:
    ensure_dirs()

    if not RAW_CSV.exists():
        print(f"Нет {RAW_CSV}. Сначала запустите scripts/00_fetch_data.py", file=sys.stderr)
        return 1

    raw = pd.read_csv(RAW_CSV, dtype=RAW_ID_DTYPE)
    report: dict = {}

    clean_frame = clean(raw, report)

    # spec_obj_id сохраняется как единственный уникальный ключ строки: obj_id в датасете
    # повторяется, и без него результаты нельзя надёжно связать с исходным файлом.
    processed = clean_frame[["obj_ID", "spec_obj_ID", *FEATURES, TARGET]].copy()
    processed = processed.rename(columns={"obj_ID": "obj_id", "spec_obj_ID": "spec_obj_id"})

    counts = processed[TARGET].value_counts()
    report["class_counts"] = {cls: int(counts.get(cls, 0)) for cls in CLASS_ORDER}
    report["class_shares"] = {
        cls: round(counts.get(cls, 0) / len(processed), 6) for cls in CLASS_ORDER
    }

    train_idx, test_idx = train_test_split(
        processed.index,
        test_size=TEST_SIZE,
        stratify=processed[TARGET],
        random_state=RANDOM_STATE,
    )
    processed["split"] = "train"
    processed.loc[test_idx, "split"] = "test"

    report["n_train"] = int(len(train_idx))
    report["n_test"] = int(len(test_idx))

    # Scaler обучается только на train, чтобы исключить утечку в оценку Подхода A;
    # Подход B затем использует тот же scaler для всего набора.
    scaler = StandardScaler().fit(processed.loc[train_idx, FEATURES])
    joblib.dump(scaler, SCALER_PKL)

    processed.to_csv(PROCESSED_CSV, index=False, encoding="utf-8")
    (RESULTS_DIR / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {PROCESSED_CSV}\n-> {SCALER_PKL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
