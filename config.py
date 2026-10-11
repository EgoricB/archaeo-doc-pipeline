import os
from pathlib import Path

# Корень проекта — папка, в которой лежит этот файл
PROJECT_ROOT = Path(__file__).resolve().parent

# Внешние программы. Можно переопределить через переменные окружения
# POPPLER_PATH и TESSERACT_EXE.
POPPLER_PATH = os.environ.get(
    'POPPLER_PATH',
    str(PROJECT_ROOT / 'tools' / 'poppler-26.02.0' / 'Library' / 'bin'),
)
TESSERACT_EXE = Path(os.environ.get(
    'TESSERACT_EXE',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
))

# Настройки распознавания. Выбраны по эксперименту experiments/ocr_tuning:
# без бинаризации и с моделью tessdata_best правильных слов на 44 % больше,
# чем в исходном варианте (200 dpi, бинаризация Оцу, установленная модель).
RENDER_DPI = 300
BINARIZE = False
# Что сохранять кроме обработанных страниц (processed_pages) и текстов OCR.
# pages_raw — исходный рендер (около 5 МБ на страницу при 300 dpi, всегда можно
# получить заново из PDF); layout_debug — отладочная картинка с типом страницы (около 4 МБ).
SAVE_RAW_PAGES = False
SAVE_LAYOUT_DEBUG = False

# Сколько страниц PDF рендерить за один вызов Poppler. Ограничивает память:
# весь отчёт в память не загружается.
RENDER_CHUNK = 8

# Сколько отчётов обрабатывать параллельно (можно переопределить: --workers N).
# Tesseract в каждом процессе работает в один поток, чтобы процессы не мешали друг другу.
WORKERS = 4

OCR_LANG = 'rus'
OCR_PSM = '3'

# Папка с моделью rus.traineddata из tessdata_best:
# https://github.com/tesseract-ocr/tessdata_best/raw/main/rus.traineddata
# Если папки нет, используется модель, установленная вместе с Tesseract.
TESSDATA_DIR = Path(os.environ.get(
    'TESSDATA_DIR',
    str(PROJECT_ROOT / 'tools' / 'tessdata_best'),
))

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

ARCHAEO_TERMS = [
    'раскоп', 'курган', 'поселение', 'могильник', 'керамика', 'погребение',
    'культурный слой', 'находка', 'шурф', 'стратиграфия', 'городище',
    'экспедиция', 'инвентарь', 'слой', 'яма', 'сооружение', 'артефакт'
]

PERIOD_TERMS = [
    'бронзового века', 'железного века', 'средневеков', 'неолит', 'энеолит',
    'античн', 'раннего железного века', 'палеолит'
]

DEFAULT_OUTPUT = Path('results')