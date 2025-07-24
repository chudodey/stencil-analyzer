#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Усовершенствованный конвейер для анализа трафаретов:
1. Раздельная обработка эталона и скана
2. Улучшенное выравнивание
3. Детекция дефектов с разделением на типы
4. Профессиональная визуализация результатов
"""

import argparse
import logging
import sys
import time
import os
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

# Конфигурация логирования


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)

# Утилиты для сохранения отладочных изображений


def save_debug_image(image: np.ndarray, filename: str, debug_dir: str, description: str = "") -> None:
    path = os.path.join(debug_dir, filename)
    try:
        cv2.imwrite(path, image)
        if description:
            logging.debug(f"Saved debug image {filename}: {description}")
        else:
            logging.debug(f"Saved debug image: {path}")
    except Exception as e:
        logging.warning(f"Failed to save '{path}': {e}")


def save_debug_pair(img1: np.ndarray, img2: np.ndarray, filename: str,
                    debug_dir: str, title1: str = "Ref", title2: str = "Scan",
                    description: str = "") -> None:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = max(h1, h2)

    def scale(img, target_h):
        h, w = img.shape[:2]
        if h != target_h:
            scale = target_h / h
            return cv2.resize(img, (int(w*scale), target_h))
        return img

    r1 = scale(img1, target_h)
    r2 = scale(img2, target_h)
    gap = 20

    if len(r1.shape) == 2:
        r1 = cv2.cvtColor(r1, cv2.COLOR_GRAY2BGR)
    if len(r2.shape) == 2:
        r2 = cv2.cvtColor(r2, cv2.COLOR_GRAY2BGR)

    combined = np.ones(
        (target_h+50, r1.shape[1]+r2.shape[1]+gap, 3), dtype=np.uint8)*255

    # Размещаем изображения
    combined[50:50+target_h, :r1.shape[1]] = r1
    combined[50:50+target_h, r1.shape[1]+gap:] = r2

    # Добавляем подписи
    cv2.putText(combined, title1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(combined, title2,
                (r1.shape[1]+gap+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Добавляем описание
    if description:
        cv2.putText(combined, description, (10, target_h+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    save_debug_image(combined, filename, debug_dir, description)

# Пользовательское исключение для ошибок выравнивания


class AlignmentError(Exception):
    pass

# Класс для интеллектуальной обрезки


class ContentCropper:
    def __init__(self, padding: int = 20, aggressive_clean: bool = False, clahe_clip: float = 3.0):
        self.padding = padding
        self.aggressive_clean = aggressive_clean
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def process_ref(self, image: np.ndarray) -> np.ndarray:
        """Минимальная обработка эталона для сохранения качества"""
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return self._crop(image, binary)

    def process_scan(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Агрессивная обработка скана для компенсации артефактов"""
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        enhanced = self.clahe.apply(gray)

        # Динамический размер блока для адаптивной бинаризации
        h, w = gray.shape
        block_size = max(21, int(min(h, w) * 0.02)) | 1
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 7
        )

        # Морфологическая очистка при необходимости
        if self.aggressive_clean:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return self._crop(image, binary), binary

    def _crop(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        """Общая логика обрезки по контуру"""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.warning("No contours found for cropping")
            return image

        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        p = self.padding
        return image[max(0, y-p):y+h+p, max(0, x-p):x+w+p]

# Класс для точного выравнивания


class StencilAligner:
    def __init__(self, corner_size: float = 0.2, min_matches: int = 15):
        self.corner_size = corner_size
        self.min_matches = min_matches
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(self, ref_gray: np.ndarray, scan_gray: np.ndarray,
              debug: bool = False, debug_dir: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """Выравнивание скана относительно эталона с визуализацией"""
        # Визуализация зон поиска
        if debug:
            self._visualize_search_zones(
                ref_gray, "02_ref_search_zones.png", debug_dir,
                "Feature search zones (20% of corners)")
            self._visualize_search_zones(
                scan_gray, "03_scan_search_zones.png", debug_dir,
                "Feature search zones (20% of corners)")

        # Извлечение особенностей
        ref_kp, ref_desc = self._extract_features(ref_gray)
        scan_kp, scan_desc = self._extract_features(scan_gray)

        # Визуализация найденных фич
        if debug:
            self._visualize_features(
                ref_gray, ref_kp, "04_ref_features.png", debug_dir,
                "ORB features detected in reference")
            self._visualize_features(
                scan_gray, scan_kp, "05_scan_features.png", debug_dir,
                "ORB features detected in scan")

        if ref_desc is None or scan_desc is None:
            raise AlignmentError("Not enough features for matching")

        # Фильтрация совпадений
        matches = self.bf.knnMatch(ref_desc, scan_desc, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < self.min_matches:
            raise AlignmentError(
                f"Not enough good matches: {len(good)}/{self.min_matches}")

        # Визуализация совпадений
        if debug:
            self._visualize_matches(ref_gray, ref_kp, scan_gray, scan_kp, good,
                                    "06_feature_matches.png", debug_dir,
                                    f"Feature matches (good: {len(good)})")

        # Преобразование точек
        pts_ref = np.float32([ref_kp[m.queryIdx].pt for m in good])
        pts_scan = np.float32([scan_kp[m.trainIdx].pt for m in good])

        # Расчет преобразования
        M, inliers = cv2.estimateAffinePartial2D(
            pts_scan, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=2.5
        )

        if M is None:
            raise AlignmentError("Affine transformation estimation failed")

        # Применение преобразования
        aligned = cv2.warpAffine(
            scan_gray, M, (ref_gray.shape[1], ref_gray.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        return M, aligned

    def _extract_features(self, gray: np.ndarray) -> Tuple[List, Optional[np.ndarray]]:
        """Извлечение особенностей из угловых областей"""
        corners = self._get_corner_regions(gray.shape)
        kps = []
        descs = None

        for x, y, w, h in corners:
            roi = gray[y:y+h, x:x+w]
            kp, desc = self.orb.detectAndCompute(roi, None)

            if kp and desc is not None:
                # Корректировка координат особенностей
                for p in kp:
                    p.pt = (p.pt[0] + x, p.pt[1] + y)
                kps.extend(kp)

                if descs is None:
                    descs = desc
                else:
                    descs = np.vstack((descs, desc))

        return kps, descs

    def _get_corner_regions(self, shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Определение угловых областей"""
        h, w = shape[:2]
        cw, ch = int(w * self.corner_size), int(h * self.corner_size)
        return [
            (0, 0, cw, ch),
            (w - cw, 0, cw, ch),
            (0, h - ch, cw, ch),
            (w - cw, h - ch, cw, ch)
        ]

    def _visualize_search_zones(self, gray: np.ndarray, filename: str,
                                debug_dir: str, description: str) -> None:
        """Визуализация зон поиска"""
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        corners = self._get_corner_regions(gray.shape)

        for i, (x, y, w, h) in enumerate(corners):
            # Рисуем прямоугольник зоны поиска
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Подписываем зону
            cv2.putText(img_color, f"Zone {i+1}", (x+5, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Добавляем описание
        cv2.putText(img_color, description, (10, img_color.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_debug_image(img_color, filename, debug_dir, description)

    def _visualize_features(self, gray: np.ndarray, keypoints: List,
                            filename: str, debug_dir: str, description: str) -> None:
        """Визуализация найденных особенностей"""
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Рисуем все ключевые точки
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size
            cv2.circle(img_color, (int(x), int(y)),
                       int(size/2), (0, 0, 255), 1)
            # Ориентация фичи
            angle = kp.angle * np.pi / 180.0
            end_x = int(x + size/2 * np.cos(angle))
            end_y = int(y + size/2 * np.sin(angle))
            cv2.line(img_color, (int(x), int(y)),
                     (end_x, end_y), (255, 0, 0), 1)

        # Добавляем статистику и описание
        cv2.putText(img_color, f"Features: {len(keypoints)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_color, description, (10, img_color.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_debug_image(img_color, filename, debug_dir, description)

    def _visualize_matches(self, ref_gray: np.ndarray, ref_kp: List,
                           scan_gray: np.ndarray, scan_kp: List,
                           good_matches: List, filename: str,
                           debug_dir: str, description: str) -> None:
        """Визуализация совпадений особенностей"""
        # Создаем цветное изображение для визуализации
        ref_color = cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2BGR)
        scan_color = cv2.cvtColor(scan_gray, cv2.COLOR_GRAY2BGR)

        # Рисуем только хорошие совпадения
        match_img = cv2.drawMatches(
            ref_color, ref_kp,
            scan_color, scan_kp,
            good_matches, None,
            matchColor=(0, 255, 0),    # Зеленый цвет для совпадений
            singlePointColor=(255, 0, 0),  # Синий цвет для одиночных точек
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )

        # Добавляем статистику и описание
        cv2.putText(match_img, f"Good matches: {len(good_matches)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(match_img, description, (10, match_img.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        save_debug_image(match_img, filename, debug_dir, description)

# Класс для детекции дефектов с разделением типов


class DefectDetector:
    def __init__(self, min_defect_area: int = 10, morph_size: int = 3):
        self.min_defect_area = min_defect_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    def detect(self, ref_binary: np.ndarray, scan_binary: np.ndarray,
               debug: bool = False, debug_dir: str = "") -> Dict[str, np.ndarray]:
        """Детекция дефектов с разделением на излишки и недостачи"""
        # Разделение типов дефектов
        excess_mask = cv2.bitwise_and(scan_binary, cv2.bitwise_not(ref_binary))
        deficit_mask = cv2.bitwise_and(
            ref_binary, cv2.bitwise_not(scan_binary))

        # Отладочная визуализация
        if debug:
            save_debug_image(excess_mask, "07_excess_raw.png", debug_dir,
                             "Raw excess material mask (before processing)")
            save_debug_image(deficit_mask, "08_deficit_raw.png", debug_dir,
                             "Raw deficit material mask (before processing)")

        # Постобработка масок
        return {
            "excess": self._process_mask(excess_mask, "excess", debug, debug_dir),
            "deficit": self._process_mask(deficit_mask, "deficit", debug, debug_dir)
        }

    def _process_mask(self, mask: np.ndarray, defect_type: str,
                      debug: bool, debug_dir: str) -> np.ndarray:
        """Очистка и фильтрация маски дефектов"""
        # Морфологическая обработка
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)

        # Фильтрация по размеру
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(cleaned)
        defect_count = 0
        total_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area:
                cv2.drawContours(result, [cnt], -1, 255, -1)
                defect_count += 1
                total_area += area

        if debug:
            desc = (f"Processed {defect_type} mask: {defect_count} defects, "
                    f"total area: {total_area} px")
            filename = f"09_{defect_type}_processed.png"
            save_debug_image(result, filename, debug_dir, desc)

        return result

# Визуализация результатов


def visualize_results(ref: np.ndarray, scan: np.ndarray, M: np.ndarray,
                      defects: Dict[str, np.ndarray]) -> np.ndarray:
    """Цветная визуализация результатов с легендой"""
    # Выравнивание цветного скана
    aligned_color = cv2.warpAffine(
        scan, M, (ref.shape[1], ref.shape[0])
    )

    # Создание результата
    result = aligned_color.copy()

    # Наложение дефектов с прозрачностью
    overlay = result.copy()
    overlay[defects['excess'] > 0] = [0, 0, 255]  # Красный - излишки
    overlay[defects['deficit'] > 0] = [255, 0, 0]  # Синий - недостачи
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

    # Добавление информативной легенды
    legend_height = 100
    expanded_result = np.zeros(
        (result.shape[0] + legend_height, result.shape[1], 3), dtype=np.uint8)
    expanded_result[:result.shape[0], :] = result

    # Рисуем легенду
    cv2.rectangle(expanded_result, (0, result.shape[0]),
                  (expanded_result.shape[1], expanded_result.shape[1]),
                  (255, 255, 255), -1)

    # Статистика дефектов
    excess_area = np.sum(defects['excess'] > 0)
    deficit_area = np.sum(defects['deficit'] > 0)

    # Текст легенды
    cv2.putText(expanded_result, "Defect Analysis Results", (10, result.shape[0]+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(expanded_result, f"Excess Material (Red): {excess_area} px",
                (10, result.shape[0]+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(expanded_result, f"Missing Material (Blue): {deficit_area} px",
                (10, result.shape[0]+90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return expanded_result

# Главный конвейер обработки


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Усовершенствованный анализ трафаретов")
    # Основные параметры
    parser.add_argument("--ref", required=True, help="Эталонное изображение")
    parser.add_argument("--scan", required=True,
                        help="Сканируемое изображение")
    parser.add_argument("--out", default="result.png",
                        help="Выходное изображение")

    # Параметры обрезки
    parser.add_argument("--padding", type=int, default=20,
                        help="Отступ при обрезке")
    parser.add_argument("--aggressive-clean",
                        action="store_true", help="Агрессивная очистка скана")
    parser.add_argument("--clahe-clip", type=float,
                        default=3.0, help="Параметр CLAHE для скана")

    # Параметры выравнивания
    parser.add_argument("--corner-size", type=float,
                        default=0.2, help="Размер угловых областей")
    parser.add_argument("--min-matches", type=int, default=15,
                        help="Минимальное количество совпадений")

    # Параметры детекции
    parser.add_argument("--min-defect-area", type=int,
                        default=10, help="Минимальная площадь дефекта")
    parser.add_argument("--morph-size", type=int, default=3,
                        help="Размер морфологического ядра")

    # Отладка
    parser.add_argument("--debug", action="store_true",
                        help="Режим отладки с сохранением промежуточных изображений")

    args = parser.parse_args()
    configure_logging(args.debug)

    debug_dir = "debug_output"
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        logging.info(f"Debug folder: {debug_dir}")

    try:
        # Загрузка изображений
        logging.info("Loading images...")
        ref_img = cv2.imread(args.ref)
        scan_img = cv2.imread(args.scan)

        if ref_img is None:
            raise ValueError(f"Failed to load reference image: {args.ref}")
        if scan_img is None:
            raise ValueError(f"Failed to load scan image: {args.scan}")

        # Этап 1: Раздельная обработка
        logging.info("Stage 1: Content cropping...")
        cropper = ContentCropper(
            padding=args.padding,
            aggressive_clean=args.aggressive_clean,
            clahe_clip=args.clahe_clip
        )

        ref_cropped = cropper.process_ref(ref_img)
        scan_cropped, scan_binary = cropper.process_scan(scan_img)

        if args.debug:
            save_debug_image(ref_cropped, "01_ref_cropped.png", debug_dir,
                             "Reference image after cropping")
            save_debug_image(scan_cropped, "01_scan_cropped.png", debug_dir,
                             "Scan image after cropping")
            save_debug_image(scan_binary, "01_scan_binary.png", debug_dir,
                             "Scan image after binarization")

        # Подготовка grayscale для выравнивания
        ref_gray = cv2.cvtColor(ref_cropped, cv2.COLOR_BGR2GRAY)
        scan_gray = cv2.cvtColor(scan_cropped, cv2.COLOR_BGR2GRAY)

        if args.debug:
            save_debug_pair(ref_gray, scan_gray, "01_gray_pair.png",
                            debug_dir, "Ref Gray", "Scan Gray",
                            "Grayscale images before alignment")

        # Этап 2: Выравнивание
        logging.info("Stage 2: Alignment...")
        aligner = StencilAligner(
            corner_size=args.corner_size,
            min_matches=args.min_matches
        )

        M, aligned_scan_gray = aligner.align(
            ref_gray, scan_gray, args.debug, debug_dir)

        if args.debug:
            save_debug_pair(ref_gray, aligned_scan_gray, "06_aligned_gray.png",
                            debug_dir, "Ref Gray", "Aligned Scan",
                            "Images after alignment")

        # Бинаризация после выравнивания
        logging.info("Stage 3: Binarization...")
        _, ref_binary = cv2.threshold(
            ref_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )
        _, aligned_binary = cv2.threshold(
            aligned_scan_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        if args.debug:
            save_debug_pair(ref_binary, aligned_binary, "07_binary_pair.png",
                            debug_dir, "Ref Binary", "Aligned Binary",
                            "Binary images for defect detection")

        # Этап 4: Детекция дефектов
        logging.info("Stage 4: Defect detection...")
        detector = DefectDetector(
            min_defect_area=args.min_defect_area,
            morph_size=args.morph_size
        )

        defects = detector.detect(
            ref_binary, aligned_binary, args.debug, debug_dir)

        # Этап 5: Визуализация и сохранение
        logging.info("Stage 5: Visualization...")
        result_img = visualize_results(ref_cropped, scan_cropped, M, defects)
        cv2.imwrite(args.out, result_img)

        if args.debug:
            save_debug_image(result_img, "10_final_result.png", debug_dir,
                             "Final result with defect visualization")
            # Дополнительная визуализация
            aligned_scan_color = cv2.warpAffine(
                scan_cropped, M, (ref_cropped.shape[1], ref_cropped.shape[0]))
            save_debug_pair(ref_cropped, aligned_scan_color,
                            "11_comparison.png", debug_dir,
                            "Reference", "Aligned Scan",
                            "Final alignment comparison")

        logging.info(
            f"Processing completed in {time.time()-start_time:.2f} seconds")
        logging.info(f"Result saved to: {args.out}")

    except AlignmentError as e:
        logging.error(f"Alignment failed: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
