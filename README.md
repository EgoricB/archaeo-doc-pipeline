# Воспроизводимый отчёт OCR

Дополнение к Archaeo Doc Pipeline. Python 3.10+, только стандартная библиотека.
Не меняет run_pipeline.py, исходные PDF, OCR-тексты и исторические результаты.

## Установка

Распакуйте ZIP так, чтобы research_reporting оказался непосредственно в корне
репозитория, рядом с run_pipeline.py. Не вкладывайте каталог в ещё один
research_reporting. Если такая папка уже есть, сначала сравните версии и сохраните
свои изменения, не заменяйте её молча.

## Проверка и запуск из корня репозитория

```sh
python -m unittest research_reporting.test_report -v
python -m research_reporting.report --out research_report_check
```

Ожидается 18 пройденных тестов. Папка вывода должна быть новой или пустой.
При повторе укажите другую папку. Результаты не перезаписываются автоматически.

Готовый пример находится в research_reporting/derived/:
- effects.svg — вклад последовательных изменений конфигурации с сохранёнными 95% ДИ;
- tradeoff.svg — словарная доля и наблюдаемое время обработки для 9 конфигураций;
- changes.csv — разницы метрик, вычисленные через VariantMetrics.__sub__;
- manifest.json — SHA-256 входов и параметры отчёта.

SVG открываются в браузере. Новый запуск не требует PDF, Tesseract или сети.
Папка research_report_check — локальный результат проверки, её не требуется коммитить.

## Данные

По умолчанию читаются data/summary.csv и data/significance.csv — точные копии
таблиц experiments/ocr_tuning из commit b3768a712208a038db639ee93de0aae14d24e73f.
Копии намеренно включены для воспроизводимости без зависимости от будущих изменений.
Git blob SHA-1 и SHA-256 приведены в data/provenance.json.
Интервалы импортируются, а не пересчитываются. Новых OCR-измерений нет.

Для чтения таблиц из существующего каталога проекта:

```sh
python -m research_reporting.report --data experiments/ocr_tuning --cohort ocr_tuning_50 --out research_report_local
```

Cohort — явное обозначение общей выборки. Равенство числа страниц не доказывает
совпадение страниц. Использовать режим только для одной парной выборки.

## Требования преподавателя

| Требование | Реализация |
|---|---|
| 3+ класса | VariantMetrics, MetricDelta, ExperimentRepository, SvgFigures, EvidenceReport |
| Композиция | EvidenceReport использует ExperimentRepository и SvgFigures |
| Данные из файлов | Локальные summary.csv и significance.csv |
| 2+ визуализации | effects.svg и tradeoff.svg |
| Специальный метод | VariantMetrics.__sub__, вызовы в load и build |

## Что коммитить

Только каталог research_reporting целиком. Внутри есть локальный .gitignore,
разрешающий отслеживание небольших CSV/SVG и исключающий __pycache__.
Презентация, исходные PDF и весь корпус в этот коммит не входят.
Перед коммитом проверьте `git diff --cached --name-only`: все пути должны
начинаться с research_reporting/. Ранее staged-изменения не должны попасть случайно.

См. AUDIT.md, TEST_RESULTS.txt и RUN_RESULTS.txt.
