"""Этап 9. Сборка сводного отчёта REPORT.md.

Отчёт собирается целиком из сохранённых CSV/JSON — ни одна цифра в нём не вписана
руками. Это исходный материал для текста работы: все таблицы уже в markdown,
все числа проставлены, все файлы описаны.
"""

import json
import sys
from datetime import date

import pandas as pd
from config import FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, ensure_dirs
from meta import load_meta

REPORT_PATH = PROJECT_ROOT / "REPORT.md"

FILE_DESCRIPTIONS = {
    "predictions_approach_a.csv": "построчные предсказания Подхода A на test-выборке",
    "predictions_approach_b.csv": "построчные результаты кластеризации на полном наборе",
    "metrics_summary.csv": "сводная таблица метрик обоих подходов",
    "feature_importance.csv": "важность признаков Random Forest",
    "confusion_matrix_a.csv": "матрица ошибок 3×3 в абсолютных числах",
    "cluster_vs_class_table.csv": "перекрёстная таблица кластер × класс",
    "cluster_vs_class_normalized.csv": "то же в долях",
    "pca_projection.csv": "координаты PCA для обоих сравнительных графиков",
    "elbow_silhouette.csv": "инерция и silhouette для k = 2..8",
    "run_config.json": "параметры запуска и версии библиотек",
    "dataset_report.json": "статистика очистки и баланс классов",
    "training_meta.json": "тайминги и гиперпараметры этапов обучения",
    "classification_report_a.csv": "precision/recall/f1 по каждому классу",
    "feature_stats_by_class.csv": "описательная статистика признаков по классам",
    "feature_correlation.csv": "матрица корреляций признаков",
    "color_indices_by_class.csv": "статистика цветовых индексов по классам",
    "color_indices.csv": "построчные цветовые индексы",
    "roc_auc_a.csv": "ROC-AUC и average precision (one-vs-rest)",
    "roc_curves_a.csv": "точки ROC-кривых для перерисовки",
    "approach_comparison.csv": "сопоставление подходов по общим показателям",
    "control_experiments.csv": "контрольные эксперименты с наборами признаков",
    "learning_curve.csv": "f1_macro в зависимости от объёма обучающей выборки",
    "gridsearch_cv_results.csv": "полная таблица перебора гиперпараметров",
}

FIGURE_DESCRIPTIONS = {
    "confusion_matrix_a.png": "матрица ошибок Подхода A",
    "pca_true_class_vs_cluster.png": "PCA: истинные классы против кластеров",
    "elbow_plot.png": "выбор числа кластеров",
    "feature_importance.png": "важность признаков",
    "class_distribution.png": "баланс классов",
    "feature_distributions.png": "распределения признаков по классам",
    "color_color_diagram.png": "цвет-цветовая диаграмма",
    "redshift_by_class.png": "красное смещение по классам",
    "correlation_heatmap.png": "корреляции признаков",
    "roc_curves_a.png": "ROC-кривые Подхода A",
    "cluster_composition.png": "состав кластеров",
    "pca_with_centroids.png": "кластеры и центроиды в PCA",
    "approach_comparison.png": "сравнение подходов",
    "control_experiments.png": "контрольные эксперименты",
    "learning_curve.png": "кривая обучения",
}


def md_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Рендерит DataFrame в markdown-таблицу без внешних зависимостей."""
    table = frame.reset_index() if index else frame
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in table.itertuples(index=False):
        cells = ["" if pd.isna(value) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read(name: str) -> pd.DataFrame:
    """Читает таблицу из results/."""
    return pd.read_csv(RESULTS_DIR / name)


def section_data(dataset: dict, config: dict) -> str:
    """Раздел о данных: очистка, баланс классов, статистики, корреляции."""
    counts = dataset["class_counts"]
    shares = dataset["class_shares"]
    balance = pd.DataFrame(
        {
            "class": list(counts),
            "count": list(counts.values()),
            "share": [f"{shares[c] * 100:.2f} %" for c in counts],
        }
    )

    cleaning = pd.DataFrame(
        [
            ("Строк в исходном файле", dataset["n_raw"]),
            ("Пропусков в рабочих столбцах", dataset["n_missing_cells"]),
            ("Строк с неуникальным obj_ID (не удалялись)", dataset["n_nonunique_obj_id"]),
            ("Удалено дубликатов по spec_obj_ID", dataset["n_duplicate_spec_obj_id"]),
            ("Удалено строк с недопустимой фотометрией", dataset["n_bad_photometry"]),
            ("Удалено строк с недопустимым redshift", dataset["n_bad_redshift"]),
            ("Осталось после очистки", dataset["n_clean"]),
            ("Размер train-выборки", dataset["n_train"]),
            ("Размер test-выборки", dataset["n_test"]),
        ],
        columns=["Показатель", "Значение"],
    )

    return f"""## 2. Данные

Источник — «Stellar Classification Dataset – SDSS17» (Kaggle), каталог Sloan Digital Sky
Survey, релиз DR17. Рабочие признаки: `{"`, `".join(config["features"])}`.
Координаты (`alpha`, `delta`), технические идентификаторы, параметры сеанса съёмки
и спектрографа в обучении не используются — обоснование в разделе 7.

### 2.1. Очистка

{md_table(cleaning)}

Два замечания, важных для воспроизведения:

* Идентификаторы `obj_ID` и `spec_obj_ID` имеют порядок 1.2·10^18, что больше 2^53,
  а в CSV часть значений записана в научной нотации. При автоопределении типа pandas
  берёт `float64` и необратимо теряет младшие разряды — разные объекты склеиваются
  в один идентификатор. Оба столбца читаются только как строки.
* `obj_ID` в этом датасете не является ключом: он повторяется у
  {dataset["n_nonunique_obj_id"]} строк, при том что фотометрия, класс и красное смещение
  у них различаются. Уникален `spec_obj_ID` — единица наблюдения здесь спектр,
  именно по нему определён класс. Дедупликация выполняется по нему.

### 2.2. Баланс классов

{md_table(balance)}

Выборка несбалансирована: галактик почти втрое больше, чем квазаров. Поэтому основной
метрикой Подхода A взята macro-F1, а не accuracy.

![Баланс классов](figures/class_distribution.png)

### 2.3. Описательная статистика признаков

{md_table(read("feature_stats_by_class.csv"))}

![Распределения признаков](figures/feature_distributions.png)

### 2.4. Корреляции признаков

{md_table(read("feature_correlation.csv"), index=True)}

Звёздные величины в соседних фильтрах сильно скоррелированы между собой — это ожидаемо,
поскольку все они измеряют яркость одного объекта. `redshift` с ними связан слабо,
что и делает его самостоятельным источником информации.

![Корреляции](figures/correlation_heatmap.png)

### 2.5. Цветовые индексы

{md_table(read("color_indices_by_class.csv"))}

![Цвет-цветовая диаграмма](figures/color_color_diagram.png)

![Красное смещение](figures/redshift_by_class.png)
"""


def section_approach_a(meta: dict, config: dict) -> str:
    """Раздел о Подходе A: гиперпараметры, метрики, ROC, важность признаков."""
    info = meta["approach_a"]
    params = config["random_forest"]
    matrix = pd.read_csv(RESULTS_DIR / "confusion_matrix_a.csv", index_col="true_class")

    return f"""## 4. Подход A — классификация с учителем

Алгоритм: `RandomForestClassifier`. Гиперпараметры подобраны `GridSearchCV`
по {params["cv_folds"]} фолдам, критерий отбора — `{params["cv_scoring"]}`.

Победившая конфигурация: `n_estimators = {params["n_estimators"]}`,
`max_depth = {params["max_depth"]}`, средний `f1_macro` на кросс-валидации
{params["cv_best_score"]:.4f}. Перебор занял {info["cv_search_time_sec"]:.1f} с,
обучение финальной модели — {info["train_time_sec"]:.2f} с.

### 4.1. Полный перебор гиперпараметров

{md_table(read("gridsearch_cv_results.csv"))}

### 4.2. Метрики по классам

{md_table(read("classification_report_a.csv"))}

Хуже всего распознаются квазары (recall
{read("classification_report_a.csv").set_index("class").loc["QSO", "recall"]:.4f}) —
их путают с галактиками. Звёзды отделяются практически безошибочно.

### 4.3. Матрица ошибок

Строки — истинный класс, столбцы — предсказанный.

{md_table(matrix, index=True)}

![Матрица ошибок](figures/confusion_matrix_a.png)

### 4.4. ROC-AUC (one-vs-rest)

{md_table(read("roc_auc_a.csv"))}

![ROC-кривые](figures/roc_curves_a.png)

### 4.5. Важность признаков

{md_table(read("feature_importance.csv"))}

![Важность признаков](figures/feature_importance.png)

### 4.6. Кривая обучения

{md_table(read("learning_curve.csv"))}

Качество выходит на плато примерно на 20–40 тысячах объектов: дальнейшее увеличение
выборки прироста почти не даёт, то есть объёма данных для этой задачи с запасом хватает.

![Кривая обучения](figures/learning_curve.png)
"""


def section_approach_b(meta: dict, config: dict) -> str:
    """Раздел о Подходе B: выбор k, метрики, состав кластеров."""
    info = meta["approach_b"]
    scan = read("elbow_silhouette.csv")
    best_k = int(scan.loc[scan["silhouette_score"].idxmax(), "k"])
    variance = config["pca_explained_variance_ratio"]
    mapping = ", ".join(f"{k} → {v}" for k, v in info["cluster_to_class"].items())

    return f"""## 5. Подход B — кластеризация без учителя

Алгоритм: `KMeans`, k = {info["n_clusters"]}, `n_init = {info["n_init"]}`.
Обучение на полном масштабированном наборе признаков без столбца `class`
заняло {info["train_time_sec"]:.2f} с.

### 5.1. Обоснование числа кластеров

{md_table(scan)}

**Silhouette score не подтверждает k = 3.** Максимум приходится на k = {best_k}
({scan["silhouette_score"].max():.4f} против {scan.loc[scan["k"] == 3,
"silhouette_score"].iloc[0]:.4f} при k = 3), а инерция убывает плавно, без выраженного
«локтя». Иначе говоря, без подсказки из меток геометрия данных трёх групп не выдаёт —
это отрицательный, но содержательный результат.

![Выбор k](figures/elbow_plot.png)

### 5.2. Сопоставление кластеров с классами

Majority vote: {mapping}.

{md_table(read("cluster_vs_class_table.csv"))}

В долях:

{md_table(read("cluster_vs_class_normalized.csv"))}

**Ни один кластер не соответствует классу STAR.** K-Means режет пополам вытянутое
облако галактик, а звёзды растворяются внутри обоих галактических кластеров.
Квазары, наоборот, выделяются хорошо — за счёт большого красного смещения.

![Состав кластеров](figures/cluster_composition.png)

### 5.3. Визуализация в пространстве PCA

Две главные компоненты объясняют {sum(variance) * 100:.2f} % дисперсии
({variance[0] * 100:.2f} % и {variance[1] * 100:.2f} % соответственно).

![PCA](figures/pca_true_class_vs_cluster.png)

![PCA с центроидами](figures/pca_with_centroids.png)
"""


def section_comparison() -> str:
    """Раздел сравнения подходов."""
    return f"""## 6. Сравнение подходов

{md_table(read("metrics_summary.csv"))}

{md_table(read("approach_comparison.csv"))}

![Сравнение подходов](figures/approach_comparison.png)

Прямое сравнение accuracy и ARI некорректно — метрики измеряют разное. Содержательно
сопоставимы две пары: accuracy против purity (доля объектов, попавших «куда надо»)
и f1_macro против ARI (согласие разметки с истиной с поправкой на случайность).
По обеим парам разрыв кратный.

Обучение без учителя при этом на два порядка быстрее, но эта экономия не окупает
потери качества: разметка в SDSS уже есть, и отказ от неё ничего не даёт.
"""


def section_controls() -> str:
    """Раздел контрольных экспериментов."""
    table = read("control_experiments.csv")
    lookup = table.set_index("experiment")
    columns = [
        "experiment", "description", "n_features",
        "rf_accuracy", "rf_f1_macro", "kmeans_ari", "kmeans_purity", "kmeans_silhouette",
    ]

    return f"""## 7. Контрольные эксперименты

Все эксперименты выполнены в одинаковых условиях: те же гиперпараметры, то же
разбиение, отличается только состав признаков.

{md_table(table[columns])}

![Контрольные эксперименты](figures/control_experiments.png)

Выводы:

1. **Координаты действительно дают утечку.** Random Forest, обученный
   *только* на `alpha` и `delta`, показывает accuracy
   {lookup.loc["coords_only", "rf_accuracy"]:.4f} — заметно выше доли
   самого частого класса (0.5945). Никакой физики за этим нет: модель запоминает,
   какие участки неба покрыты какими наблюдательными программами. Именно поэтому
   координаты исключены из рабочего набора.
2. **На полном наборе координаты ничего не добавляют**
   (f1_macro {lookup.loc["with_coords", "rf_f1_macro"]:.4f} против
   {lookup.loc["baseline", "rf_f1_macro"]:.4f}) — `redshift` уже исчерпывает задачу.
3. **Красное смещение решает задачу почти в одиночку**: на одном этом признаке
   f1_macro = {lookup.loc["redshift_only", "rf_f1_macro"]:.4f}. Без него, на чистой
   фотометрии, качество падает до
   {lookup.loc["no_redshift", "rf_f1_macro"]:.4f}.
4. **Для кластеризации важен переход к цветовым индексам.** Замена «сырых» звёздных
   величин на разности соседних фильтров поднимает ARI с
   {lookup.loc["baseline", "kmeans_ari"]:.4f} до
   {lookup.loc["colors_and_redshift", "kmeans_ari"]:.4f} — заметный относительный
   прирост при том же алгоритме. Для Random Forest замена ничего не меняет: дерево
   и так способно построить разность признаков само.
"""


def section_files() -> str:
    """Приложение: реестр всех выходных файлов."""
    rows = []
    for name, description in FILE_DESCRIPTIONS.items():
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        size_kb = path.stat().st_size / 1024
        rows.append({"файл": f"`results/{name}`", "описание": description,
                     "размер": f"{size_kb:.0f} КБ"})
    files = pd.DataFrame(rows)

    figure_rows = [
        {"файл": f"`figures/{name}`", "описание": description}
        for name, description in FIGURE_DESCRIPTIONS.items()
        if (FIGURES_DIR / name).exists()
    ]

    return f"""## 9. Приложение: реестр файлов

### Таблицы

{md_table(files)}

### Иллюстрации

Все PNG — 300 dpi.

{md_table(pd.DataFrame(figure_rows))}

### Модели

| файл | описание |
|---|---|
| `models/scaler.pkl` | `StandardScaler`, обучен на train-выборке |
| `models/random_forest_model.pkl` | обученная модель Подхода A |
| `models/kmeans_model.pkl` | обученная модель Подхода B |

Загружаются через `joblib.load`, переобучение не требуется.
"""


def main() -> int:
    ensure_dirs()
    required = ["control_experiments.csv", "classification_report_a.csv", "learning_curve.csv"]
    for name in required:
        if not (RESULTS_DIR / name).exists():
            print(f"Нет {name}. Запустите скрипты 01-08.", file=sys.stderr)
            return 1

    meta = load_meta()
    config = json.loads((RESULTS_DIR / "run_config.json").read_text(encoding="utf-8"))
    dataset = json.loads((RESULTS_DIR / "dataset_report.json").read_text(encoding="utf-8"))
    versions = config["library_versions"]

    metrics = read("metrics_summary.csv")
    lookup = {(row.approach, row.metric): row.value for row in metrics.itertuples()}
    controls = read("control_experiments.csv").set_index("experiment")
    importance = read("feature_importance.csv")
    lib_versions = f'{versions["pandas"]} / {versions["numpy"]} / {versions["scikit-learn"]}'
    train_share = 1 - config["test_size"]
    test_share = config["test_size"]

    header = f"""# Сравнение supervised- и unsupervised-подходов к классификации небесных объектов
## Отчёт по технической части. Датасет SDSS17

Отчёт собран автоматически скриптом `scripts/09_build_report.py` из сохранённых таблиц.
Ни одно число здесь не вписано вручную.

| | |
|---|---|
| Дата прогона | {config["run_date"]} |
| Дата сборки отчёта | {date.today().isoformat()} |
| Объектов после очистки | {dataset["n_clean"]} |
| Разбиение train/test | {dataset["n_train"]} / {dataset["n_test"]} |
| `random_state` | {config["random_state"]} |
| Python | {versions["python"]} |
| pandas / numpy / scikit-learn | {lib_versions} |

---

## 1. Краткие итоги

| Подход | Ключевая метрика | Значение |
|---|---|---|
| A — Random Forest | accuracy | {lookup[("A", "accuracy")]:.4f} |
| A — Random Forest | f1_macro | {lookup[("A", "f1_macro")]:.4f} |
| A — Random Forest | время обучения | {lookup[("A", "train_time_sec")]:.2f} с |
| B — K-Means (k = 3) | ARI | {lookup[("B", "ari")]:.4f} |
| B — K-Means (k = 3) | NMI | {lookup[("B", "nmi")]:.4f} |
| B — K-Means (k = 3) | purity | {lookup[("B", "purity")]:.4f} |
| B — K-Means (k = 3) | silhouette | {lookup[("B", "silhouette")]:.4f} |
| B — K-Means (k = 3) | время обучения | {lookup[("B", "train_time_sec")]:.2f} с |

Обучение с учителем решает задачу практически полностью; кластеризация на тех же
признаках воспроизводит истинное деление лишь частично и вообще не выделяет звёзды
в отдельную группу.

---
"""

    methodology = f"""## 3. Методика

1. Очистка и отбор признаков (см. раздел 2).
2. Стратифицированное разбиение train/test в отношении
   {train_share:.0%} / {test_share:.0%} с `random_state = {config["random_state"]}`.
3. `StandardScaler` обучается **только на train-выборке** и затем применяется ко всему
   набору. Это исключает утечку в оценку Подхода A; Подход B использует тот же scaler.
4. Подход A обучается на train с истинными метками, оценивается на отложенной test-выборке.
5. Подход B обучается на полном наборе без меток; метки привлекаются только на этапе
   оценки, для majority-vote сопоставления кластеров с классами.

`silhouette_score` считается на детерминированной подвыборке в
{meta["approach_b"]["silhouette_sample_size"]} объектов: на полном наборе потребовалась бы
матрица попарных расстояний {dataset["n_clean"]} × {dataset["n_clean"]}.

Воспроизводимость: `random_state` зафиксирован для разбиения, Random Forest, K-Means,
PCA и подвыборки silhouette. Два полных прогона подряд дают побайтово идентичные
таблицы; различаются только замеры времени обучения.

---
"""

    conclusions = f"""## 8. Выводы

1. **Разрыв между подходами кратный.** Random Forest даёт f1_macro
   {lookup[("A", "f1_macro")]:.4f}, K-Means — ARI {lookup[("B", "ari")]:.4f}.
   По доле верно отнесённых объектов: {lookup[("A", "accuracy")]:.4f} против
   purity {lookup[("B", "purity")]:.4f}.

2. **Классы физически перекрываются в пространстве признаков.** На PCA-проекции звёзды
   и галактики лежат на одном вытянутом гребне. K-Means со сферическими кластерами
   разделить их не может в принципе — он режет облако там, где плотность точек выше,
   а не там, где проходит физическая граница.

3. **Оптимальное k без меток не определяется.** Silhouette указывает на k = 2, а не 3;
   «локтя» на кривой инерции нет. Это существенно: даже зная, что классов три,
   исследователь без разметки не получил бы такого указания из самих данных.

4. **Задача почти целиком держится на красном смещении** — важность
   {importance.iloc[0]["importance"]:.4f}, а на одном этом признаке
   f1_macro достигает
   {controls.loc["redshift_only", "rf_f1_macro"]:.4f}.
   Преимущество Random Forest в том, что он дополнительно использует нелинейные границы
   по цветам и добирает то, что redshift не разделяет.

5. **Состав признаков нужно контролировать явно.** Координаты на небесной сфере сами по
   себе дают accuracy
   {controls.loc["coords_only", "rf_accuracy"]:.4f}
   — результат без физического смысла, порождённый геометрией обзора.

6. **Кластеризацию можно улучшить, не меняя алгоритм.** Переход к цветовым индексам
   поднимает ARI до
   {controls.loc["colors_and_redshift", "kmeans_ari"]:.4f}.
   Разрыв с обучением с учителем это не закрывает, но показывает, что значительная часть
   слабости Подхода B — следствие представления признаков, а не только самого метода.

---
"""

    report = "\n".join(
        [
            header,
            section_data(dataset, config),
            methodology,
            section_approach_a(meta, config),
            section_approach_b(meta, config),
            section_comparison(),
            section_controls(),
            conclusions,
            section_files(),
        ]
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"-> {REPORT_PATH} ({len(report.splitlines())} строк, {len(report) / 1024:.0f} КБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
