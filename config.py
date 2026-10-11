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