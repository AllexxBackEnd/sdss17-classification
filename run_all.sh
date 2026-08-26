#!/usr/bin/env bash
# Полный прогон: от загрузки данных до REPORT.md со всеми таблицами и графиками.
set -euo pipefail
cd "$(dirname "$0")"

# Обязательный минимум по ТЗ
uv run python scripts/00_fetch_data.py
uv run python scripts/01_preprocess.py
uv run python scripts/02_train_supervised.py
uv run python scripts/03_cluster_unsupervised.py
uv run python scripts/04_evaluate.py
uv run python scripts/05_figures.py

# Расширенный слой для текста работы
uv run python scripts/06_extended_analysis.py
uv run python scripts/07_control_experiments.py
uv run python scripts/08_extended_figures.py
uv run python scripts/09_build_report.py

echo "Готово. Сводный отчёт: REPORT.md"
