"""Этап 0. Получение исходного датасета в data/raw_sdss17.csv.

Если файл уже лежит на месте (например, скачан с Kaggle вручную) — скрипт ничего
не делает. Иначе эквивалентная выборка запрашивается напрямую у первоисточника —
публичного SQL-сервиса SDSS SkyServer DR17, из которого собран и Kaggle-датасет.
"""

import sys
import urllib.parse
import urllib.request

import pandas as pd
from config import DATA_DIR, RANDOM_STATE, RAW_CSV, ensure_dirs

SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

# Схема столбцов и соединение таблиц повторяют запрос, которым собран Kaggle-датасет
# "Stellar Classification Dataset - SDSS17". Фильтр по fiberID даёт равномерную
# псевдослучайную выборку ~1/47 спектров, распределённую по всем пластинам.
QUERY = """
SELECT
    p.objid AS obj_ID, p.ra AS alpha, p.dec AS delta,
    p.u AS u, p.g AS g, p.r AS r, p.i AS i, p.z AS z,
    p.run AS run_ID, p.rerun AS rerun_ID, p.camcol AS cam_col, p.field AS field_ID,
    s.specobjid AS spec_obj_ID, s.class AS class, s.z AS redshift,
    s.plate AS plate, s.mjd AS MJD, s.fiberid AS fiber_ID
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE s.fiberID % 47 = 11
"""

TARGET_ROWS = 100_000
DOWNLOAD_TIMEOUT_SEC = 900


def download_from_skyserver(destination_csv: str) -> None:
    """Скачивает выборку SkyServer во временный файл и сохраняет ровно TARGET_ROWS строк."""
    params = urllib.parse.urlencode({"cmd": QUERY, "format": "csv"})
    print(f"Запрос к SkyServer DR17 (это занимает 1-3 минуты)...\n  {SKYSERVER_URL}")

    with urllib.request.urlopen(f"{SKYSERVER_URL}?{params}", timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
        raw_bytes = resp.read()

    tmp_path = DATA_DIR / "_skyserver_raw.csv"
    tmp_path.write_bytes(raw_bytes)

    # Первая строка ответа SkyServer — служебный маркер "#Table1".
    frame = pd.read_csv(tmp_path, skiprows=1)
    tmp_path.unlink()

    if len(frame) < TARGET_ROWS:
        raise RuntimeError(
            f"SkyServer вернул только {len(frame)} строк, ожидалось >= {TARGET_ROWS}"
        )

    frame = frame.sample(n=TARGET_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)
    frame.to_csv(destination_csv, index=False, encoding="utf-8")
    print(f"Сохранено {len(frame)} строк -> {destination_csv}")


def main() -> int:
    ensure_dirs()

    if RAW_CSV.exists():
        print(f"{RAW_CSV} уже существует — скачивание пропущено.")
        return 0

    download_from_skyserver(str(RAW_CSV))
    return 0


if __name__ == "__main__":
    sys.exit(main())
