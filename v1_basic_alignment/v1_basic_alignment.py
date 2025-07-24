#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v1_basic_alignment.py

Алгоритм базового выравнивания сканированных изображений трафаретов паяльной пасты 
с эталонными изображениями через анализ угловых зон.

КЛЮЧЕВЫЕ ОСОБЕННОСТИ:
    - Двухэтапная предобработка (нормализация + бинаризация)
    - ORB-детектирование в угловых зонах
    - Similarity transform (масштаб + поворот + перенос)
    - Генерация маски различий с морфологической очисткой

Использование:
    python v1_basic_alignment.py --ref ref.png --scan scan.jpg --out output_imgs/diff_mask.png
    python v1_basic_alignment.py --ref ref.png --scan scan.jpg --two-stage --debug

Зависимости:
    - Python 3.6+
    - OpenCV (cv2)
    - NumPy
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple, List, Any
import time

import cv2
import numpy as np


def configure_logging() -> None:
    """Настраивает базовые параметры логирования."""
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")


def validate_file_extension(filename: str, valid_extensions: tuple = ('.png', '.jpg', '.jpeg')) -> None:
    """Проверяет расширение файла."""
    if not filename.lower().endswith(valid_extensions):
        raise ValueError(
            f"Неподдерживаемый формат файла {filename}. Допустимы: {valid_extensions}")


def parse_args() -> argparse.Namespace:
    """Парсит и валидирует аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Алгоритм базового выравнивания трафаретов паяльной пасты"
    )
    parser.add_argument("--ref", required=True,
                        help="Эталонное изображение (из CAM350)")
    parser.add_argument("--scan", required=True,
                        help="Сканированное изображение трафарета")
    parser.add_argument("--out", default=os.path.join("output_imgs", "diff_mask.png"),
                        help="Файл карты различий (по умолчанию: output_imgs/diff_mask.png)")
    parser.add_argument("--corner-size", type=float, default=0.15,
                        help="Размер угловых зон для анализа (доля от размера изображения)")
    parser.add_argument("--min-matches", type=int, default=8,
                        help="Минимальное число соответствий в каждом углу")
    parser.add_argument("--debug", action="store_true",
                        help="Режим отладки с сохранением промежуточных результатов")
    parser.add_argument("--two-stage", action="store_true",
                        help="Двухэтапное выравнивание (грубое + точное)")

    args = parser.parse_args()

    # Валидация расширений файлов
    validate_file_extension(args.ref)
    validate_file_extension(args.scan)
    validate_file_extension(args.out, ('.png',))

    return args


class CornerBasedMatcher:
    """Класс для сопоставления изображений на основе угловых зон."""

    def __init__(self, corner_ratio: float = 0.15, min_matches_per_corner: int = 8):
        self.corner_ratio = corner_ratio
        self.min_matches_per_corner = min_matches_per_corner
        self.orb = cv2.ORB_create(nfeatures=1000)

    def get_corner_regions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Возвращает координаты угловых зон (x, y, w, h)."""
        h, w = image_shape[:2]
        corner_w = int(w * self.corner_ratio)
        corner_h = int(h * self.corner_ratio)

        return [
            (0, 0, corner_w, corner_h),                    # Top-Left
            (w - corner_w, 0, corner_w, corner_h),         # Top-Right
            (0, h - corner_h, corner_w, corner_h),         # Bottom-Left
            (w - corner_w, h - corner_h, corner_w, corner_h)  # Bottom-Right
        ]

    def extract_corner_features(self, image: np.ndarray, debug_mode: bool = False,
                                debug_folder: str = None) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Извлекает ORB-признаки из угловых зон."""
        corners = self.get_corner_regions(image.shape)
        all_keypoints = []
        all_descriptors = []

        for i, (x, y, w, h) in enumerate(corners):
            corner_roi = image[y:y+h, x:x+w]
            kp_corner, desc_corner = self.orb.detectAndCompute(
                corner_roi, None)

            if kp_corner is None or desc_corner is None:
                continue

            for kp in kp_corner:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

            all_keypoints.extend(kp_corner)
            all_descriptors = desc_corner if len(all_descriptors) == 0 else np.vstack([
                all_descriptors, desc_corner])

            logging.info(f"Угол {i+1}: найдено {len(kp_corner)} признаков")

        if debug_mode and debug_folder:
            self._save_corner_debug(
                image, corners, all_keypoints, debug_folder)

        return all_keypoints, all_descriptors

    def _save_corner_debug(self, image: np.ndarray, corners: List[Tuple[int, int, int, int]],
                           keypoints: List[cv2.KeyPoint], debug_folder: str) -> None:
        """Сохраняет отладочную визуализацию угловых зон."""
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(
            image.shape) == 2 else image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for i, (x, y, w, h) in enumerate(corners):
            cv2.rectangle(vis_image, (x, y), (x+w, y+h),
                          colors[i % len(colors)], 2)

        if keypoints:
            vis_image = cv2.drawKeypoints(vis_image, keypoints, None, color=(0, 255, 255),
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        debug_path = os.path.join(debug_folder, "corner_analysis.png")
        cv2.imwrite(debug_path, vis_image)
        logging.info(f"Сохранен отладочный файл: {debug_path}")


class SimilarityTransformer:
    """Класс для вычисления similarity transformation (масштаб + поворот + перенос)."""

    @staticmethod
    def estimate_similarity_transform(pts_src: np.ndarray, pts_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Вычисляет similarity transformation (4 DOF: tx, ty, scale, rotation).
        Возвращает матрицу 2x3 и маску инлайеров.
        """
        if len(pts_src) < 2 or len(pts_dst) < 2:
            raise ValueError("Недостаточно точек для вычисления трансформации")

        M, mask = cv2.estimateAffinePartial2D(pts_src, pts_dst,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0,
                                              maxIters=2000,
                                              confidence=0.99)
        if M is None:
            raise ValueError("Не удалось вычислить трансформацию")

        return M, mask


def load_and_preprocess(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Загружает изображение и выполняет предобработку (нормализация + бинаризация)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение {image_path}")

    # Нормализация контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Бинаризация
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return enhanced, binary


def create_difference_mask(ref_binary: np.ndarray, scan_aligned: np.ndarray,
                           erosion_kernel_size: int = 3) -> np.ndarray:
    """Создает маску различий с морфологической очисткой."""
    diff_mask = cv2.bitwise_xor(ref_binary, scan_aligned)

    if erosion_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_kernel_size, erosion_kernel_size))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    return diff_mask


def save_debug_output(debug_folder: str, filename: str, *images: np.ndarray) -> None:
    """Сохраняет отладочные изображения в output_imgs."""
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder, exist_ok=True)

    path = os.path.join(debug_folder, filename)
    try:
        if len(images) == 1:
            cv2.imwrite(path, images[0])
        else:
            # Объединение нескольких изображений по горизонтали
            combined = np.hstack(images)
            cv2.imwrite(path, combined)
        logging.info(f"Сохранен отладочный файл: {path}")
    except Exception as e:
        logging.warning(f"Ошибка сохранения {path}: {str(e)}")


def main():
    """Основная функция выполнения скрипта."""
    start_time = time.time()
    configure_logging()
    args = parse_args()

    try:
        logging.info("=== АЛГОРИТМ БАЗОВОГО ВЫРАВНИВАНИЯ ТРАФАРЕТОВ ===")

        # Создание папки для выходных данных
        output_dir = "output_imgs"
        os.makedirs(output_dir, exist_ok=True)

        # 1. Загрузка и предобработка
        logging.info("Загрузка и предобработка изображений...")
        ref_gray, ref_binary = load_and_preprocess(args.ref)
        scan_gray, scan_binary = load_and_preprocess(args.scan)

        if args.debug:
            save_debug_output(output_dir, "01_original.png",
                              ref_gray, scan_gray)
            save_debug_output(output_dir, "02_binary.png",
                              ref_binary, scan_binary)

        # 2. Анализ угловых зон
        logging.info("Анализ угловых зон...")
        matcher = CornerBasedMatcher(args.corner_size, args.min_matches)
        ref_kp, ref_desc = matcher.extract_corner_features(
            ref_binary, args.debug, output_dir)
        scan_kp, scan_desc = matcher.extract_corner_features(
            scan_binary, args.debug, output_dir)

        if ref_desc is None or scan_desc is None:
            raise ValueError("Не удалось извлечь признаки из угловых зон")

        # 3. Сопоставление признаков
        logging.info("Сопоставление признаков...")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(ref_desc, scan_desc, k=2)
        good_matches = [m for m, n in matches if len(
            matches[0]) == 2 and m.distance < 0.7 * n.distance]

        if len(good_matches) < args.min_matches:
            raise ValueError(
                f"Недостаточно соответствий: {len(good_matches)} < {args.min_matches}")

        if args.debug:
            matches_img = cv2.drawMatches(ref_gray, ref_kp, scan_gray, scan_kp,
                                          good_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            save_debug_output(output_dir, "03_matches.png", matches_img)

        # 4. Вычисление трансформации
        logging.info("Вычисление трансформации...")
        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches])
        scan_pts = np.float32([scan_kp[m.trainIdx].pt for m in good_matches])

        M, mask = SimilarityTransformer.estimate_similarity_transform(
            scan_pts, ref_pts)

        # Логирование параметров трансформации
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        logging.info(
            f"Параметры трансформации: масштаб={scale:.3f}, угол={angle:.2f}°, сдвиг=({tx:.1f}, {ty:.1f})")

        # 5. Применение трансформации
        logging.info("Выравнивание изображения...")
        h, w = ref_gray.shape
        scan_aligned = cv2.warpAffine(scan_binary, M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

        if args.debug:
            save_debug_output(output_dir, "04_aligned.png",
                              ref_binary, scan_aligned)

        # 6. Создание маски различий
        logging.info("Генерация маски различий...")
        diff_mask = create_difference_mask(ref_binary, scan_aligned)
        diff_pixels = np.count_nonzero(diff_mask)
        diff_percent = (diff_pixels / (h * w)) * 100
        logging.info(
            f"Обнаружено различий: {diff_pixels} пикселей ({diff_percent:.2f}%)")

        # 7. Сохранение результатов
        logging.info(f"Сохранение результата в {args.out}...")
        cv2.imwrite(args.out, diff_mask)

        if args.debug:
            overlay = cv2.cvtColor(scan_aligned, cv2.COLOR_GRAY2BGR)
            overlay[diff_mask > 0] = [0, 0, 255]
            save_debug_output(output_dir, "05_differences.png", overlay)

        logging.info(
            f"✅ Успешно завершено за {time.time()-start_time:.2f} сек")

    except Exception as e:
        logging.error(f"❌ Ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
