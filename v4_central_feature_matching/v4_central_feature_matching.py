#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Усовершенствованный конвейер для анализа трафаретов:
Версия с интеллектуальной маскировкой для поиска центрального элемента в сетке.
"""

import argparse
import logging
import sys
import time
import os
import cv2
import numpy as np
from typing import Tuple, Dict, Optional, List

# ... (Код до StencilAligner остаётся без изменений) ...

# Конфигурация логирования


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=level, format=fmt)

# Утилиты для сохранения изображений


def save_output_image(image: np.ndarray, filename: str, output_dir: str, description: str = "") -> None:
    path = os.path.join(output_dir, filename)
    try:
        cv2.imwrite(path, image)
        if description:
            logging.debug(f"Saved output image {filename}: {description}")
        else:
            logging.debug(f"Saved output image: {path}")
    except Exception as e:
        logging.warning(f"Failed to save '{path}': {e}")


def save_output_pair(img1: np.ndarray, img2: np.ndarray, filename: str,
                     output_dir: str, title1: str = "Ref", title2: str = "Scan",
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

    combined[50:50+target_h, :r1.shape[1]] = r1
    combined[50:50+target_h, r1.shape[1]+gap:] = r2

    cv2.putText(combined, title1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(combined, title2,
                (r1.shape[1]+gap+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if description:
        cv2.putText(combined, description, (10, target_h+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    save_output_image(combined, filename, output_dir, description)

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
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return self._crop(image, binary)

    def process_scan(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        enhanced = self.clahe.apply(gray)
        h, w = gray.shape
        block_size = max(21, int(min(h, w) * 0.02)) | 1
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, 7
        )
        if self.aggressive_clean:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return self._crop(image, binary), binary

    def _crop(self, image: np.ndarray, binary: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.warning("No contours found for cropping")
            return image
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        p = self.padding
        return image[max(0, y-p):y+h+p, max(0, x-p):x+w+p]


# === КЛАСС STENCILALIGNER ПОЛНОСТЬЮ ПЕРЕРАБОТАН ===
class StencilAligner:
    def __init__(self, center_size: float = 0.5, min_matches: int = 15, min_element_area: int = 1000):
        self.center_size = center_size
        self.min_matches = min_matches
        self.min_element_area = min_element_area
        self.orb = cv2.ORB_create(
            nfeatures=2000, scaleFactor=1.2, scoreType=cv2.ORB_FAST_SCORE)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Поместите эту обновлённую функцию в класс StencilAligner, заменив старую

    # Поместите эту обновлённую функцию в класс StencilAligner, заменив предыдущую

    # Поместите эту обновлённую функцию в класс StencilAligner, заменив предыдущую

    def _get_centermost_element_mask(self, gray_image: np.ndarray, debug: bool = False, output_dir: str = "") -> Optional[np.ndarray]:
        """Находит все элементы в центре, выбирает центральный и возвращает маску для него."""
        h, w = gray_image.shape

        # 1. Определяем центральную ROI
        roi_w, roi_h = int(w * self.center_size), int(h * self.center_size)
        roi_x, roi_y = (w - roi_w) // 2, (h - roi_h) // 2
        central_roi = gray_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        # 2. Адаптивная бинаризация для получения всех отдельных площадок
        binary_roi = cv2.adaptiveThreshold(
            central_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=51, C=7
        )

        # 3. Сильная дилатация, чтобы "склеить" площадки одной платы в единый blob
        # Размер ядра должен быть достаточным, чтобы перекрыть зазоры внутри одной платы
        # Примерно 10-15% от размера платы (270px) -> ~27px
        dilation_kernel_size = 27
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (dilation_kernel_size, dilation_kernel_size))
        dilated_roi = cv2.dilate(binary_roi, kernel, iterations=1)

        if debug:
            save_output_image(
                dilated_roi, f"02a_dilated_roi_{'ref' if 'ref' in gray_image.sum().astype(str) else 'scan'}.png", output_dir, "Dilated ROI to find element groups")

        # 4. Поиск контуров сгруппированных элементов ("клякс")
        grouped_contours, _ = cv2.findContours(
            dilated_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Фильтрация контуров по площади
        valid_contours = [cnt for cnt in grouped_contours if cv2.contourArea(
            cnt) > self.min_element_area]
        if not valid_contours:
            logging.warning(
                f"No element groups found with area > {self.min_element_area} px in the central zone.")
            return None

        # 6. Выбор самого центрального контура
        roi_center_pt = (roi_w / 2, roi_h / 2)
        min_dist = float('inf')
        centermost_grouped_contour = None

        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = np.sqrt(
                (cx - roi_center_pt[0])**2 + (cy - roi_center_pt[1])**2)
            if dist < min_dist:
                min_dist = dist
                centermost_grouped_contour = cnt

        if centermost_grouped_contour is None:
            return None

        # 7. Создание финальной, точной маски
        # Мы не используем размытый контур, а берем его Bounding Box
        # и вырезаем им ОРИГИНАЛЬНЫЕ площадки из binary_roi
        x, y, w_cnt, h_cnt = cv2.boundingRect(centermost_grouped_contour)

        # Создаем пустую маску для ROI
        final_roi_mask = np.zeros_like(binary_roi)
        # Копируем в нее только те пиксели из оригинальной бинарной маски, что попали в BBox
        final_roi_mask[y:y+h_cnt, x:x+w_cnt] = binary_roi[y:y+h_cnt, x:x+w_cnt]

        # Переносим маску из ROI в глобальные координаты
        final_mask = np.zeros_like(gray_image)
        final_mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = final_roi_mask

        return final_mask

    def align(self, ref_gray: np.ndarray, scan_gray: np.ndarray,
              debug: bool = False, output_dir: str = "") -> Tuple[np.ndarray, np.ndarray]:

        logging.info("Creating masks for centermost elements...")
        # Передаем debug и output_dir в функцию
        ref_mask = self._get_centermost_element_mask(
            ref_gray, debug, output_dir)
        scan_mask = self._get_centermost_element_mask(
            scan_gray, debug, output_dir)

        if ref_mask is None or scan_mask is None:
            raise AlignmentError(
                "Could not isolate a central element for alignment. Try adjusting --center-size or --min-element-area.")

        if debug:
            self._visualize_search_mask(
                ref_gray, ref_mask, "02_ref_search_mask.png", output_dir, "Reference Alignment Mask")
            self._visualize_search_mask(
                scan_gray, scan_mask, "03_scan_search_mask.png", output_dir, "Scan Alignment Mask")

        # Ищем ключевые точки только в пределах масок
        ref_kp, ref_desc = self.orb.detectAndCompute(ref_gray, mask=ref_mask)
        scan_kp, scan_desc = self.orb.detectAndCompute(
            scan_gray, mask=scan_mask)

        if debug:
            self._visualize_features(
                ref_gray, ref_kp, "04_ref_features.png", output_dir, "ORB features in reference mask")
            self._visualize_features(
                scan_gray, scan_kp, "05_scan_features.png", output_dir, "ORB features in scan mask")

        if ref_desc is None or scan_desc is None or len(ref_kp) < self.min_matches or len(scan_kp) < self.min_matches:
            raise AlignmentError(
                "Not enough features found in the masked areas for matching.")

        matches = self.bf.knnMatch(ref_desc, scan_desc, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < self.min_matches:
            raise AlignmentError(
                f"Not enough good matches: {len(good)}/{self.min_matches}")

        if debug:
            self._visualize_matches(ref_gray, ref_kp, scan_gray, scan_kp, good,
                                    "06_feature_matches.png", output_dir, f"Feature matches (good: {len(good)})")

        pts_ref = np.float32([ref_kp[m.queryIdx].pt for m in good])
        pts_scan = np.float32([scan_kp[m.trainIdx].pt for m in good])

        M, _ = cv2.estimateAffinePartial2D(
            pts_scan, pts_ref, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000)
        if M is None:
            raise AlignmentError("Affine transformation estimation failed")

        aligned = cv2.warpAffine(
            scan_gray, M, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR)
        return M, aligned

    def _visualize_search_mask(self, gray: np.ndarray, mask: np.ndarray, filename: str, output_dir: str, description: str) -> None:
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Накладываем маску с полупрозрачностью
        overlay = img_color.copy()
        overlay[mask > 0] = (0, 255, 0)  # Зеленая область для поиска
        result = cv2.addWeighted(overlay, 0.4, img_color, 0.6, 0)

        cv2.putText(result, description, (10,
                    result.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        save_output_image(result, filename, output_dir, description)

    def _visualize_features(self, gray: np.ndarray, keypoints: List, filename: str, output_dir: str, description: str) -> None:
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        img_color = cv2.drawKeypoints(img_color, keypoints, None, color=(
            0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.putText(img_color, f"Features: {len(keypoints)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img_color, description, (10,
                    img_color.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_output_image(img_color, filename, output_dir, description)

    def _visualize_matches(self, ref_gray: np.ndarray, ref_kp: List, scan_gray: np.ndarray, scan_kp: List, good_matches: List, filename: str, output_dir: str, description: str) -> None:
        match_img = cv2.drawMatches(ref_gray, ref_kp, scan_gray, scan_kp, good_matches, None, matchColor=(
            0, 255, 0), singlePointColor=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        cv2.putText(match_img, f"Good matches: {len(good_matches)}", (
            10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(match_img, description, (10,
                    match_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        save_output_image(match_img, filename, output_dir, description)


# ... (Классы DefectDetector и visualize_results остаются без изменений) ...
class DefectDetector:
    def __init__(self, min_defect_area: int = 10, morph_size: int = 3):
        self.min_defect_area = min_defect_area
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size))

    def detect(self, ref_binary: np.ndarray, scan_binary: np.ndarray, debug: bool = False, output_dir: str = "") -> Dict[str, np.ndarray]:
        excess_mask = cv2.bitwise_and(scan_binary, cv2.bitwise_not(ref_binary))
        deficit_mask = cv2.bitwise_and(
            ref_binary, cv2.bitwise_not(scan_binary))

        if debug:
            save_output_image(excess_mask, "07_excess_raw.png",
                              output_dir, "Raw excess material mask")
            save_output_image(deficit_mask, "08_deficit_raw.png",
                              output_dir, "Raw deficit material mask")

        return {
            "excess": self._process_mask(excess_mask, "excess", debug, output_dir),
            "deficit": self._process_mask(deficit_mask, "deficit", debug, output_dir)
        }

    def _process_mask(self, mask: np.ndarray, defect_type: str, debug: bool, output_dir: str) -> np.ndarray:
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(cleaned)
        defect_count, total_area = 0, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_defect_area:
                cv2.drawContours(result, [cnt], -1, 255, -1)
                defect_count += 1
                total_area += area
        if debug:
            desc = f"Processed {defect_type} mask: {defect_count} defects, area: {total_area:.0f} px"
            save_output_image(
                result, f"09_{defect_type}_processed.png", output_dir, desc)
        return result

# Визуализация результатов


def visualize_results(ref: np.ndarray, scan: np.ndarray, M: np.ndarray, defects: Dict[str, np.ndarray]) -> np.ndarray:
    aligned_color = cv2.warpAffine(scan, M, (ref.shape[1], ref.shape[0]))
    result = aligned_color.copy()
    overlay = result.copy()
    overlay[defects['excess'] > 0] = [0, 0, 255]  # Red - excess
    overlay[defects['deficit'] > 0] = [255, 0, 0]  # Blue - deficit
    cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)

    legend_height = 80
    expanded_result = np.full(
        (result.shape[0] + legend_height, result.shape[1], 3), 255, dtype=np.uint8)
    expanded_result[:result.shape[0], :] = result

    excess_area = np.sum(defects['excess'] > 0)
    deficit_area = np.sum(defects['deficit'] > 0)

    cv2.putText(expanded_result, "Defect Analysis Results", (10,
                result.shape[0] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.rectangle(expanded_result, (10,
                  result.shape[0] + 40), (30, result.shape[0] + 60), (0, 0, 255), -1)
    cv2.putText(expanded_result, f"Excess Material (Red): {excess_area} px", (
        40, result.shape[0] + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.rectangle(expanded_result, (expanded_result.shape[1]//2, result.shape[0] + 40), (
        expanded_result.shape[1]//2+20, result.shape[0] + 60), (255, 0, 0), -1)
    cv2.putText(expanded_result, f"Missing Material (Blue): {deficit_area} px", (
        expanded_result.shape[1]//2+30, result.shape[0] + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return expanded_result


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Усовершенствованный анализ трафаретов с интеллектуальной маскировкой.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- Основные параметры ---
    parser.add_argument("--ref", required=True, help="Эталонное изображение")
    parser.add_argument("--scan", required=True,
                        help="Сканируемое изображение")
    parser.add_argument("--out", default="result.png",
                        help="Имя выходного файла с результатом")

    # --- Параметры обработки ---
    parser.add_argument("--padding", type=int, default=20,
                        help="Отступ при обрезке содержимого")
    parser.add_argument("--aggressive-clean", action="store_true",
                        help="Агрессивная морфологическая очистка скана")

    # --- Параметры выравнивания ---
    parser.add_argument("--center-size", type=float, default=0.5,
                        help="Размер центральной зоны для поиска элементов (доля от 0.1 до 1.0)")
    parser.add_argument("--min-element-area", type=int, default=1000,
                        help="Минимальная площадь элемента (платы) в пикселях для его учёта")
    parser.add_argument("--min-matches", type=int, default=10,
                        help="Минимальное количество совпадений для успешного выравнивания")

    # --- Параметры детекции ---
    parser.add_argument("--min-defect-area", type=int, default=10,
                        help="Минимальная площадь дефекта в пикселях")
    parser.add_argument("--morph-size", type=int, default=3,
                        help="Размер морфологического ядра для очистки дефектов")

    # --- Отладка ---
    parser.add_argument("--debug", action="store_true",
                        help="Режим отладки с сохранением промежуточных изображений")

    args = parser.parse_args()

    configure_logging(args.debug)

    output_dir = "output_imgs"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output folder: {output_dir}")

    try:
        logging.info("Loading images...")
        ref_img = cv2.imread(args.ref)
        scan_img = cv2.imread(args.scan)
        if ref_img is None:
            raise ValueError(f"Failed to load reference image: {args.ref}")
        if scan_img is None:
            raise ValueError(f"Failed to load scan image: {args.scan}")

        logging.info("Stage 1: Content cropping...")
        cropper = ContentCropper(args.padding, args.aggressive_clean)
        ref_cropped = cropper.process_ref(ref_img)
        scan_cropped, _ = cropper.process_scan(scan_img)
        if args.debug:
            save_output_image(ref_cropped, "01_ref_cropped.png",
                              output_dir, "Reference after cropping")
            save_output_image(scan_cropped, "01_scan_cropped.png",
                              output_dir, "Scan after cropping")

        ref_gray = cv2.cvtColor(ref_cropped, cv2.COLOR_BGR2GRAY)
        scan_gray = cv2.cvtColor(scan_cropped, cv2.COLOR_BGR2GRAY)

        logging.info("Stage 2: Alignment...")
        aligner = StencilAligner(
            args.center_size, args.min_matches, args.min_element_area)
        M, aligned_scan_gray = aligner.align(
            ref_gray, scan_gray, args.debug, output_dir)

        if args.debug:
            save_output_pair(ref_gray, aligned_scan_gray, "06_aligned_gray.png",
                             output_dir, "Ref Gray", "Aligned Scan", "Images after alignment")

        logging.info("Stage 3: Binarization...")
        _, ref_binary = cv2.threshold(
            ref_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, aligned_binary = cv2.threshold(
            aligned_scan_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if args.debug:
            save_output_pair(ref_binary, aligned_binary, "07_binary_pair.png", output_dir,
                             "Ref Binary", "Aligned Binary", "Binary images for defect detection")

        logging.info("Stage 4: Defect detection...")
        detector = DefectDetector(args.min_defect_area, args.morph_size)
        defects = detector.detect(
            ref_binary, aligned_binary, args.debug, output_dir)

        logging.info("Stage 5: Visualization...")
        result_img = visualize_results(ref_cropped, scan_cropped, M, defects)

        output_path = os.path.join(output_dir, args.out)
        cv2.imwrite(output_path, result_img)

        if args.debug:
            save_output_image(result_img, "10_final_result_debug.png",
                              output_dir, "Final result with defect visualization")
            aligned_scan_color = cv2.warpAffine(
                scan_cropped, M, (ref_cropped.shape[1], ref_cropped.shape[0]))
            save_output_pair(ref_cropped, aligned_scan_color, "11_comparison.png",
                             output_dir, "Reference", "Aligned Scan", "Final alignment comparison")

        logging.info(
            f"Processing completed in {time.time()-start_time:.2f} seconds")
        logging.info(f"Result saved to: {output_path}")

    except (AlignmentError, ValueError) as e:
        logging.error(f"A critical error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected processing error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
