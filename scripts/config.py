"""Общие пути, константы и параметры воспроизводимости для всего пайплайна."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"

RAW_CSV: Path = DATA_DIR / "raw_sdss17.csv"
PROCESSED_CSV: Path = DATA_DIR / "processed.csv"

SCALER_PKL: Path = MODELS_DIR / "scaler.pkl"
RF_PKL: Path = MODELS_DIR / "random_forest_model.pkl"
KMEANS_PKL: Path = MODELS_DIR / "kmeans_model.pkl"

# --- параметры эксперимента (раздел 4-6 ТЗ) ---

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Идентификаторы SDSS (~1.2e18) превышают 2**53 и не представимы в float64 без потерь,
# а в CSV часть значений записана в научной нотации — поэтому читаем их только как строки.
RAW_ID_DTYPE: dict[str, type] = {"obj_ID": str, "spec_obj_ID": str}
PROCESSED_ID_DTYPE: dict[str, type] = {"obj_id": str, "spec_obj_id": str}

FEATURES: list[str] = ["u", "g", "r", "i", "z", "redshift"]
TARGET: str = "class"
CLASS_ORDER: list[str] = ["GALAXY", "QSO", "STAR"]

N_CLUSTERS: int = 3
K_RANGE: range = range(2, 9)
KMEANS_N_INIT: int = 10

# silhouette_score на 100k объектах требует матрицу расстояний 100k x 100k,
# поэтому считаем на детерминированной подвыборке.
SILHOUETTE_SAMPLE_SIZE: int = 10_000

# Значения-заглушки SDSS для отсутствующей фотометрии и физически допустимые границы.
SDSS_SENTINEL: float = -9000.0
MAG_MIN: float = 0.0
MAG_MAX: float = 40.0
REDSHIFT_MIN: float = -0.1
REDSHIFT_MAX: float = 10.0

FIGURE_DPI: int = 300


def ensure_dirs() -> None:
    """Создаёт все выходные каталоги проекта, если их ещё нет."""
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
