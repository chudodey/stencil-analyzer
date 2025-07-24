#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_and_crop_content.py

Скрипт для предварительного анализа изображений трафаретов.
Находит область с полезным содержимым (матрицу апертур),
отсекая пустые поля и шум, и сохраняет обрезанные изображения.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def configure_logging(debug: bool = False) -> None:
    """Настройка логирования: INFO по умолчанию, DEBUG при отладке."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    p = argparse.ArgumentParser(
        description="Поиск и обрезка полезной области на изображениях трафаретов."
    )
    p.add_argument("--ref",      required=True, type=Path,
                   help="Эталонное изображение.")
    p.add_argument("--scan",     required=True, type=Path,
                   help="Сканированное изображение.")
    p.add_argument("--out-ref",  required=True, type=Path, dest="out_ref",
                   help="Путь для сохранения обрезанного эталона.")
    p.add_argument("--out-scan", required=True, type=Path, dest="out_scan",
                   help="Путь для сохранения обрезанного скана.")
    p.add_argument("--min-area",       type=int,   default=10,
                   help="Минимальная площадь контура (пиксели).")
    p.add_argument("--max-area-ratio", type=float, default=0.25,
                   help="Макс. площадь контура как доля от всей площади (для рамок).")
    p.add_argument("--padding",        type=int,   default=20,
                   help="Отступ (пиксели) вокруг найденной рамки.")
    p.add_argument("--debug", action="store_true",
                   help="Сохранять дополнительные отладочные изображения и логи.")
    return p.parse_args()


def load_and_preprocess(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает файл в градациях серого, улучшает контраст CLAHE
    и возвращает (gray, binary_inv) с адаптивной бинаризацией.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path!s} не найден")

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Не удалось загрузить {path!s} как изображение")

    # CLAHE — улучшение локального контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Инвертированная адаптивная бинаризация
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=5
    )
    return gray, binary


def find_content_bbox(
    binary: np.ndarray,
    min_area: int,
    max_area_ratio: float
) -> Optional[Tuple[int, int, int, int]]:
    """
    Ищет все внешние контуры, фильтрует по площади и
    возвращает общую bounding box (x, y, w, h) или None.
    """
    # Морфологическая очистка шума
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logging.warning("Контуры не найдены.")
        return None

    h, w = binary.shape
    max_area = h * w * max_area_ratio

    valid = [cnt for cnt in contours
             if min_area < cv2.contourArea(cnt) < max_area]

    if not valid:
        logging.warning(
            "Нет контуров в диапазоне площадей: "
            f"{min_area}–{int(max_area)} px."
        )
        return None

    # объединяем все точки для единой рамки
    all_pts = np.vstack(valid)
    return cv2.boundingRect(all_pts)  # x, y, w, h


def crop_and_save(
    gray: np.ndarray,
    bbox: Tuple[int, int, int, int],
    out_path: Path,
    padding: int,
    debug: bool = False
) -> None:
    """
    Обрезает `gray` по `bbox` с `padding`, сохраняет в out_path.
    При debug рисует рамку и сохраняет отдельный файл.
    """
    x, y, w, h = bbox
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(gray.shape[1], x + w + padding)
    y2 = min(gray.shape[0], y + h + padding)

    cropped = gray[y1:y2, x1:x2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cropped)
    logging.info(f"Сохранено: {out_path!s}")

    if debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dbg_path = out_path.with_name(out_path.stem + "_debug.png")
        cv2.imwrite(str(dbg_path), vis)
        logging.debug(f"Отладочное изображение: {dbg_path!s}")


def process_one(
    img_path: Path,
    out_path: Path,
    args: argparse.Namespace
) -> bool:
    """
    Загрузка, поиск bbox и сохранение.
    Возвращает True, если всё прошло без ошибок.
    """
    logging.info(f"--- Обработка {img_path.name} ---")
    try:
        gray, binary = load_and_preprocess(img_path)
        bbox = find_content_bbox(binary, args.min_area, args.max_area_ratio)
        if bbox is None:
            logging.error(f"Полезная область не найдена: {img_path.name}")
            return False

        logging.debug(f"Найден bbox: {bbox}")
        crop_and_save(gray, bbox, out_path, args.padding, args.debug)
        return True

    except Exception:
        logging.exception(f"Ошибка при обработке {img_path!s}")
        return False


def main():
    start = time.time()
    args = parse_args()
    configure_logging(args.debug)

    ok_ref = process_one(args.ref,  args.out_ref,  args)
    ok_scan = process_one(args.scan, args.out_scan, args)

    elapsed = time.time() - start
    logging.info(f"Готово. Время: {elapsed:.2f} сек.")

    # Завершаем с кодом != 0, если были ошибки
    if not (ok_ref and ok_scan):
        sys.exit(1)


if __name__ == "__main__":
    main()
