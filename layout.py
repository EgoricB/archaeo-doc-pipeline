import numpy as np
from PIL import Image, ImageDraw


def analyze_layout(image_path, debug_path=None):
    img = Image.open(image_path).convert('L')
    arr = np.array(img)
    ink = arr < 200
    h, w = arr.shape

    row_density = ink.mean(axis=1)
    col_density = ink.mean(axis=0)

    dense_rows = int((row_density > 0.15).sum())
    dense_cols = int((col_density > 0.15).sum())
    text_density = float(ink.mean())

    if text_density < 0.03:
        page_type = 'image_or_blank'
    elif dense_rows < h * 0.2:
        page_type = 'figure_or_table'
    elif text_density > 0.12:
        page_type = 'dense_text'
    else:
        page_type = 'mixed'

    if debug_path:
        rgb = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(rgb)
        draw.rectangle([10, 10, w - 10, h - 10], outline='red', width=3)
        draw.text((20, 20), f'{page_type} | density={text_density:.3f}', fill='blue')
        rgb.save(debug_path)

    return {
        'page_type': page_type,
        'text_density': text_density,
        'dense_rows': dense_rows,
        'dense_cols': dense_cols,
    }