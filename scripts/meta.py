"""Накопление метаданных обучения (тайминги, гиперпараметры) между этапами пайплайна."""

import json

from config import RESULTS_DIR

META_PATH = RESULTS_DIR / "training_meta.json"


def load_meta() -> dict:
    """Читает накопленные метаданные; возвращает пустой словарь, если файла ещё нет."""
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def update_meta(section: str, payload: dict) -> None:
    """Записывает payload в указанную секцию, не затирая остальные секции."""
    meta = load_meta()
    meta[section] = payload
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
