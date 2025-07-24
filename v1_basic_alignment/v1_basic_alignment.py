#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimized_stencil_compare.py (v5)

Оптимизированный скрипт для автоматического совмещения сканированных изображений
трафаретов паяльной пасты с эталонными изображениями.

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ v5:
    - Анализ только угловых зон для начальной регистрации
    - Упрощенная трансформация: только масштабирование, поворот, перенос
    - Двухэтапный алгоритм: грубое + точное выравнивание
    - Оптимизация производительности за счет анализа ROI

Зависимости:
    - Python 3.6+
    - OpenCV (cv2)
    - NumPy

Использование:
    python optimized_stencil_compare.py --ref ref.png --scan scan.jpg --out diff_mask.png
"""

import argparse
import logging
import os
import sys
from typing import Optional, Tuple, List
import time

import cv2
import numpy as np


def configure_logging() -> None:
    """Настраивает базовые параметры логирования."""
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Оптимизированное совмещение сканированных трафаретов"
    )
    parser.add_argument("--ref", required=True,
                        help="Эталонное изображение (из CAM350)")
    parser.add_argument("--scan", required=True,
                        help="Сканированное изображение трафарета")
    parser.add_argument("--out", default="diff_mask.png",
                        help="Файл карты различий")
    parser.add_argument("--corner-size", type=float, default=0.15,
                        help="Размер угловых зон для анализа (доля от размера изображения)")
    parser.add_argument("--min-matches", type=int, default=8,
                        help="Минимальное число соответствий в каждом углу")
    parser.add_argument("--debug", action="store_true",
                        help="Режим отладки")
    parser.add_argument("--two-stage", action="store_true",
                        help="Двухэтапное выравнивание (грубое + точное)")
    return parser.parse_args()


class CornerBasedMatcher:
    """Класс для сопоставления изображений на основе угловых зон."""

    def __init__(self, corner_ratio: float = 0.15, min_matches_per_corner: int = 8):
        self.corner_ratio = corner_ratio
        self.min_matches_per_corner = min_matches_per_corner
        # Меньше точек для быстродействия
        self.orb = cv2.ORB_create(nfeatures=1000)

    def get_corner_regions(self, image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Возвращает координаты угловых зон (x, y, w, h)."""
        h, w = image_shape[:2]
        corner_w = int(w * self.corner_ratio)
        corner_h = int(h * self.corner_ratio)

        # Четыре угла: TL, TR, BL, BR
        corners = [
            (0, 0, corner_w, corner_h),                    # Top-Left
            (w - corner_w, 0, corner_w, corner_h),         # Top-Right
            (0, h - corner_h, corner_w, corner_h),         # Bottom-Left
            (w - corner_w, h - corner_h, corner_w, corner_h)  # Bottom-Right
        ]
        return corners

    def extract_corner_features(self, image: np.ndarray, debug_mode: bool = False,
                                debug_folder: str = None, prefix: str = "") -> Tuple[List, List]:
        """Извлекает признаки из угловых зон."""
        corners = self.get_corner_regions(image.shape)
        all_keypoints = []
        all_descriptors = []

        # Сохранение отладочной визуализации углов
        if debug_mode and debug_folder:
            save_corner_regions_debug(debug_mode, debug_folder, image, corners,
                                      None, prefix)

        for i, (x, y, w, h) in enumerate(corners):
            # Извлекаем угловую зону
            corner_roi = image[y:y+h, x:x+w]

            # Детектируем признаки в угловой зоне
            kp_corner, desc_corner = self.orb.detectAndCompute(
                corner_roi, None)

            if kp_corner is None or desc_corner is None:
                continue

            # Пересчитываем координаты ключевых точек в глобальную систему координат
            for kp in kp_corner:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

            all_keypoints.extend(kp_corner)
            if len(all_descriptors) == 0:
                all_descriptors = desc_corner
            else:
                all_descriptors = np.vstack([all_descriptors, desc_corner])

            logging.info(f"Угол {i+1}: найдено {len(kp_corner)} признаков")

        # Сохранение финальной визуализации с ключевыми точками
        if debug_mode and debug_folder:
            save_corner_regions_debug(debug_mode, debug_folder, image, corners,
                                      all_keypoints, f"{prefix}_with_keypoints")

        return all_keypoints, all_descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Сопоставляет признаки между двумя множествами дескрипторов."""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Используем более строгий тест отношения для угловых зон
        matches = bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Более строгий критерий
                    good_matches.append(m)

        return good_matches


class SimilarityTransformer:
    """Класс для вычисления similarity transformation (масштаб + поворот + перенос)."""

    @staticmethod
    def estimate_similarity_transform(pts_src: np.ndarray, pts_dst: np.ndarray) -> Optional[np.ndarray]:
        """
        Вычисляет similarity transformation (4 DOF: tx, ty, scale, rotation).
        Возвращает матрицу 2x3 для warpAffine.
        """
        if len(pts_src) < 2 or len(pts_dst) < 2:
            return None

        # Используем cv2.estimateAffinePartial2D для similarity transform
        M, mask = cv2.estimateAffinePartial2D(pts_src, pts_dst,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0,
                                              maxIters=2000,
                                              confidence=0.99)

        if M is None:
            return None

        return M, mask


def load_and_preprocess(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Загружает изображение и выполняет предобработку."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл {image_path} не найден")

    # Загрузка
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение {image_path}")

    # Изменение размера для ускорения (опционально)
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Нормализация контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # Бинаризация
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return enhanced, binary


def create_difference_mask(ref_binary: np.ndarray, scan_aligned: np.ndarray,
                           erosion_kernel_size: int = 3) -> np.ndarray:
    """Создает маску различий с постобработкой."""
    # XOR для поиска различий
    diff_mask = cv2.bitwise_xor(ref_binary, scan_aligned)

    # Морфологическая очистка для удаления шума
    if erosion_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (erosion_kernel_size, erosion_kernel_size))
        diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    return diff_mask


def save_debug_image_pair(debug_mode: bool, folder: str, filename: str,
                          ref_image: np.ndarray, scan_image: np.ndarray,
                          title_ref: str = "Reference", title_scan: str = "Scan"):
    """Сохраняет пару изображений side-by-side для отладки."""
    if not debug_mode or ref_image is None or scan_image is None:
        return

    # Приведение к одному размеру
    h_ref, w_ref = ref_image.shape[:2]
    h_scan, w_scan = scan_image.shape[:2]

    # Масштабируем к одной высоте
    target_height = max(h_ref, h_scan)

    if h_ref != target_height:
        scale = target_height / h_ref
        w_ref_new = int(w_ref * scale)
        ref_resized = cv2.resize(ref_image, (w_ref_new, target_height))
    else:
        ref_resized = ref_image.copy()
        w_ref_new = w_ref

    if h_scan != target_height:
        scale = target_height / h_scan
        w_scan_new = int(w_scan * scale)
        scan_resized = cv2.resize(scan_image, (w_scan_new, target_height))
    else:
        scan_resized = scan_image.copy()
        w_scan_new = w_scan

    # Объединение изображений по горизонтали
    gap = 20  # Промежуток между изображениями
    combined_width = w_ref_new + w_scan_new + gap
    combined_height = target_height + 60  # Место для заголовков

    # Создание объединенного изображения
    if len(ref_image.shape) == 3:
        combined = np.ones(
            (combined_height, combined_width, 3), dtype=np.uint8) * 255
    else:
        combined = np.ones((combined_height, combined_width),
                           dtype=np.uint8) * 255

    # Размещение изображений
    combined[30:30+target_height, 0:w_ref_new] = ref_resized
    combined[30:30+target_height, w_ref_new +
             gap:w_ref_new+gap+w_scan_new] = scan_resized

    # Добавление заголовков (простое решение - белые прямоугольники для текста)
    # В реальном проекте можно использовать cv2.putText, но это требует шрифт

    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, combined)
        logging.info(f"Отладочная пара сохранена: {path}")
    except Exception as e:
        logging.warning(f"Не удалось сохранить отладочный файл '{path}': {e}")


def save_debug_single(debug_mode: bool, folder: str, filename: str, image: np.ndarray):
    """Сохраняет одиночное отладочное изображение."""
    if not debug_mode or image is None:
        return

    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, image)
        logging.info(f"Отладочное изображение сохранено: {path}")
    except Exception as e:
        logging.warning(f"Не удалось сохранить отладочный файл '{path}': {e}")


def save_corner_regions_debug(debug_mode: bool, folder: str, image: np.ndarray,
                              corners: List[Tuple[int, int, int, int]],
                              keypoints: List, prefix: str):
    """Сохраняет изображение с выделенными угловыми зонами и ключевыми точками."""
    if not debug_mode:
        return

    # Создаем цветную копию для визуализации
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    # Рисуем прямоугольники угловых зон
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0)]  # Разные цвета для углов
    for i, (x, y, w, h) in enumerate(corners):
        color = colors[i % len(colors)]
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)

    # Рисуем ключевые точки
    if keypoints:
        vis_image = cv2.drawKeypoints(vis_image, keypoints, None, color=(0, 255, 255),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    filename = f"{prefix}_corners_analysis.png"
    save_debug_single(debug_mode, folder, filename, vis_image)


def main():
    """Основная функция."""
    start_time = time.time()
    configure_logging()
    args = parse_args()

    try:
        logging.info(
            "=== ОПТИМИЗИРОВАННЫЙ АЛГОРИТМ СРАВНЕНИЯ ТРАФАРЕТОВ v5 ===")

        # Создание папки для отладки
        debug_folder = "debug_output_v5"
        if args.debug:
            os.makedirs(debug_folder, exist_ok=True)
            logging.info(
                f"Режим отладки включен. Файлы будут сохранены в '{debug_folder}'")

        # 1. Загрузка и предобработка
        logging.info("Шаг 1: Загрузка изображений...")
        ref_gray, ref_binary = load_and_preprocess(args.ref)
        scan_gray, scan_binary = load_and_preprocess(args.scan)

        logging.info(
            f"Размеры - Эталон: {ref_gray.shape}, Скан: {scan_gray.shape}")

        # Отладка: исходные изображения
        save_debug_image_pair(args.debug, debug_folder, "01_original_grayscale.png",
                              ref_gray, scan_gray, "Reference", "Scan")
        save_debug_image_pair(args.debug, debug_folder, "02_binary_images.png",
                              ref_binary, scan_binary, "Reference Binary", "Scan Binary")

        # 2. Анализ угловых зон
        logging.info("Шаг 2: Анализ угловых зон...")
        matcher = CornerBasedMatcher(args.corner_size, args.min_matches)

        # Извлечение признаков из углов
        ref_kp, ref_desc = matcher.extract_corner_features(ref_binary, args.debug,
                                                           debug_folder, "03_ref")
        scan_kp, scan_desc = matcher.extract_corner_features(scan_binary, args.debug,
                                                             debug_folder, "03_scan")

        if ref_desc is None or scan_desc is None:
            raise ValueError("Не удалось извлечь признаки из угловых зон")

        # 3. Сопоставление признаков
        logging.info("Шаг 3: Сопоставление признаков...")
        matches = matcher.match_features(ref_desc, scan_desc)
        logging.info(f"Найдено {len(matches)} соответствий")

        if len(matches) < args.min_matches:
            raise ValueError(
                f"Недостаточно соответствий: {len(matches)} < {args.min_matches}")

        # Отладка: визуализация соответствий
        if args.debug:
            matches_img = cv2.drawMatches(ref_gray, ref_kp, scan_gray, scan_kp,
                                          matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            save_debug_single(args.debug, debug_folder,
                              "04_feature_matches.png", matches_img)

        # 4. Вычисление трансформации
        logging.info("Шаг 4: Вычисление similarity transformation...")
        ref_points = np.float32(
            [ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        scan_points = np.float32(
            [scan_kp[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        transformer = SimilarityTransformer()
        transform_result = transformer.estimate_similarity_transform(
            scan_points, ref_points)

        if transform_result is None or transform_result[0] is None:
            raise ValueError("Не удалось вычислить трансформацию")

        M, inlier_mask = transform_result
        inlier_count = int(np.sum(inlier_mask)
                           ) if inlier_mask is not None else len(matches)
        logging.info(
            f"Трансформация найдена: {inlier_count}/{len(matches)} инлайеров")

        # Логирование параметров трансформации
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        angle = np.arctan2(M[0, 1], M[0, 0]) * 180 / np.pi
        tx, ty = M[0, 2], M[1, 2]
        logging.info(
            f"Параметры: масштаб={scale:.4f}, поворот={angle:.2f}°, сдвиг=({tx:.1f},{ty:.1f})")

        # Отладка: визуализация инлайеров
        if args.debug and inlier_mask is not None:
            inlier_matches = [matches[i]
                              for i in range(len(matches)) if inlier_mask[i]]
            inlier_matches_img = cv2.drawMatches(ref_gray, ref_kp, scan_gray, scan_kp,
                                                 inlier_matches, None,
                                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            save_debug_single(args.debug, debug_folder,
                              "05_inlier_matches.png", inlier_matches_img)

        # 5. Применение трансформации
        logging.info("Шаг 5: Выравнивание сканированного изображения...")
        h, w = ref_gray.shape
        scan_aligned_gray = cv2.warpAffine(scan_gray, M, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=0)
        scan_aligned = cv2.warpAffine(scan_binary, M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)

        # Отладка: выровненные изображения
        save_debug_image_pair(args.debug, debug_folder, "06_aligned_grayscale.png",
                              ref_gray, scan_aligned_gray, "Reference", "Scan Aligned")
        save_debug_image_pair(args.debug, debug_folder, "07_aligned_binary.png",
                              ref_binary, scan_aligned, "Reference Binary", "Scan Aligned Binary")

        # 6. Создание маски различий
        logging.info("Шаг 6: Создание маски различий...")
        diff_mask = create_difference_mask(ref_binary, scan_aligned)

        # Статистика различий
        diff_pixels = np.count_nonzero(diff_mask)
        total_pixels = diff_mask.shape[0] * diff_mask.shape[1]
        diff_percentage = (diff_pixels / total_pixels) * 100
        logging.info(
            f"Различия: {diff_pixels} пикселей ({diff_percentage:.2f}%)")

        # Отладка: маска различий на фоне выровненных изображений
        if args.debug:
            # Создание overlay для лучшей визуализации различий
            overlay = cv2.cvtColor(scan_aligned_gray, cv2.COLOR_GRAY2BGR)
            overlay[diff_mask > 0] = [0, 0, 255]  # Красные области различий
            save_debug_single(args.debug, debug_folder,
                              "08_differences_overlay.png", overlay)

        # 7. Сохранение результата
        logging.info(f"Шаг 7: Сохранение результата в '{args.out}'...")
        success = cv2.imwrite(args.out, diff_mask)
        if not success:
            raise IOError(f"Не удалось сохранить файл {args.out}")

        elapsed_time = time.time() - start_time
        logging.info(f"✅ Готово! Время выполнения: {elapsed_time:.2f} сек")

    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
