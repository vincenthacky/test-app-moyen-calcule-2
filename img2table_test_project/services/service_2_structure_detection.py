#!/usr/bin/env python3
"""
SERVICE 2: DÉTECTION DE STRUCTURE DE TABLEAUX
Détecte lignes horizontales/verticales, identifie les cellules
et prépare les zones pour l'OCR cellule par cellule
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class CellRegion:
    """Représente une cellule du tableau"""
    x: int
    y: int
    width: int
    height: int
    row: int
    col: int
    confidence: float = 1.0

@dataclass
class TableStructure:
    """Structure complète du tableau détectée"""
    cells: List[CellRegion]
    num_rows: int
    num_cols: int
    table_bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    horizontal_lines: List[Tuple[int, int, int, int]]
    vertical_lines: List[Tuple[int, int, int, int]]
    confidence: float

class StructureDetectionService:
    """Service dédié à la détection de structure de tableaux"""

    def __init__(self):
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Paramètres de détection
        self.min_line_length = 50
        self.line_thickness_threshold = 2
        self.cell_min_size = 20
        self.merge_threshold = 10  # Distance pour fusionner lignes proches
        self.min_column_distance = 120  # Distance minimale entre colonnes (évite bruit écran/papier)
        self.min_row_distance = 35  # Distance minimale entre lignes horizontales

    def detect_table_structure(self, image_path: str) -> Optional[TableStructure]:
        """Point d'entrée principal: détecte la structure complète"""
        self.logger.info(f"🔍 Détection structure: {Path(image_path).name}")

        # Charger l'image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image non lisible: {image_path}")

        try:
            # Pipeline de détection
            horizontal_lines = self._detect_horizontal_lines(img)
            vertical_lines = self._detect_vertical_lines(img)

            self.logger.info(f"📏 Détecté: {len(horizontal_lines)} lignes H, {len(vertical_lines)} lignes V")

            if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
                self.logger.warning("Structure insuffisante détectée")
                return self._fallback_grid_detection(img)

            # Nettoyer et fusionner les lignes
            horizontal_lines = self._clean_and_merge_lines(horizontal_lines, is_horizontal=True)
            vertical_lines = self._clean_and_merge_lines(vertical_lines, is_horizontal=False)

            # CORRECTION: Filtrer les lignes trop proches (évite bruit écran/papier)
            vertical_lines = self._filter_close_lines(vertical_lines, min_dist=self.min_column_distance)
            horizontal_lines = self._filter_close_lines_horizontal(horizontal_lines, min_dist=self.min_row_distance)

            # Identifier les intersections et créer la grille
            cells = self._create_cell_grid(horizontal_lines, vertical_lines, img.shape)

            if not cells:
                return self._fallback_grid_detection(img)

            # Calculer bounding box du tableau
            table_bbox = self._calculate_table_bbox(horizontal_lines, vertical_lines)

            # Créer structure finale
            structure = TableStructure(
                cells=cells,
                num_rows=len(horizontal_lines) - 1,
                num_cols=len(vertical_lines) - 1,
                table_bbox=table_bbox,
                horizontal_lines=horizontal_lines,
                vertical_lines=vertical_lines,
                confidence=self._calculate_structure_confidence(cells, img.shape)
            )

            self.logger.info(f"✅ Structure: {structure.num_rows}x{structure.num_cols} cellules")
            return structure

        except Exception as e:
            self.logger.error(f"Erreur détection structure: {e}")
            return None

    def _detect_horizontal_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détecter les lignes horizontales (séparateurs de rangées)"""

        # Kernel horizontal pour morphologie
        kernel_length = max(img.shape[1] // 30, 15)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        # Détection par morphologie
        horizontal_lines_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Améliorer la détection avec seuillage
        _, horizontal_lines_img = cv2.threshold(horizontal_lines_img, 50, 255, cv2.THRESH_BINARY)

        # Détecter contours des lignes
        contours, _ = cv2.findContours(horizontal_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filtrer: doit être suffisamment long et fin
            if w > self.min_line_length and h <= self.line_thickness_threshold * 2:
                lines.append((x, y, x + w, y))

        # Alternative: Hough Transform pour lignes plus subtiles
        hough_lines = self._detect_lines_hough(img, orientation='horizontal')
        lines.extend(hough_lines)

        return lines

    def _detect_vertical_lines(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Détecter les lignes verticales (séparateurs de colonnes)"""

        # Kernel vertical
        kernel_length = max(img.shape[0] // 30, 15)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        # Détection par morphologie
        vertical_lines_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Seuillage
        _, vertical_lines_img = cv2.threshold(vertical_lines_img, 50, 255, cv2.THRESH_BINARY)

        # Détecter contours
        contours, _ = cv2.findContours(vertical_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filtrer: doit être suffisamment long et fin
            if h > self.min_line_length and w <= self.line_thickness_threshold * 2:
                lines.append((x, y, x, y + h))

        # Alternative: Hough Transform
        hough_lines = self._detect_lines_hough(img, orientation='vertical')
        lines.extend(hough_lines)

        return lines

    def _detect_lines_hough(self, img: np.ndarray, orientation: str) -> List[Tuple[int, int, int, int]]:
        """Détecter lignes avec Hough Transform comme méthode alternative"""

        # Préparation pour Hough
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        # Hough Transform (seuils augmentés pour réduire faux positifs)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=150,  # Seuil augmenté
            minLineLength=self.min_line_length * 2,  # Longueur minimale doublée
            maxLineGap=5  # Gap réduit
        )

        if lines is None:
            return []

        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculer angle de la ligne
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if orientation == 'horizontal' and (angle < 10 or angle > 170):
                # Ligne horizontale
                filtered_lines.append((min(x1, x2), y1, max(x1, x2), y1))
            elif orientation == 'vertical' and (80 < angle < 100):
                # Ligne verticale
                filtered_lines.append((x1, min(y1, y2), x1, max(y1, y2)))

        return filtered_lines

    def _clean_and_merge_lines(self, lines: List[Tuple[int, int, int, int]], is_horizontal: bool) -> List[Tuple[int, int, int, int]]:
        """Nettoyer et fusionner les lignes proches"""
        if not lines:
            return []

        # Trier les lignes
        if is_horizontal:
            lines.sort(key=lambda l: l[1])  # Trier par y
        else:
            lines.sort(key=lambda l: l[0])  # Trier par x

        merged = []
        current_line = lines[0]

        for next_line in lines[1:]:
            if is_horizontal:
                # Fusionner si y proche
                if abs(next_line[1] - current_line[1]) <= self.merge_threshold:
                    # Étendre la ligne
                    x_start = min(current_line[0], next_line[0])
                    x_end = max(current_line[2], next_line[2])
                    current_line = (x_start, current_line[1], x_end, current_line[3])
                else:
                    merged.append(current_line)
                    current_line = next_line
            else:
                # Fusionner si x proche
                if abs(next_line[0] - current_line[0]) <= self.merge_threshold:
                    # Étendre la ligne
                    y_start = min(current_line[1], next_line[1])
                    y_end = max(current_line[3], next_line[3])
                    current_line = (current_line[0], y_start, current_line[2], y_end)
                else:
                    merged.append(current_line)
                    current_line = next_line

        merged.append(current_line)
        return merged

    def _filter_close_lines(self, lines: List[Tuple[int, int, int, int]], min_dist: int = 40) -> List[Tuple[int, int, int, int]]:
        """
        Filtrer les lignes verticales trop proches les unes des autres.
        Évite la sur-détection causée par le bruit de l'image (grain, écran).
        """
        if not lines:
            return []

        # Trier par position X
        sorted_lines = sorted(lines, key=lambda l: l[0])

        filtered = [sorted_lines[0]]
        for line in sorted_lines[1:]:
            # Garder seulement si la distance avec la dernière ligne est suffisante
            if line[0] - filtered[-1][0] >= min_dist:
                filtered.append(line)

        self.logger.debug(f"🔧 Lignes verticales filtrées: {len(lines)} -> {len(filtered)}")
        return filtered

    def _filter_close_lines_horizontal(self, lines: List[Tuple[int, int, int, int]], min_dist: int = 25) -> List[Tuple[int, int, int, int]]:
        """
        Filtrer les lignes horizontales trop proches les unes des autres.
        Évite la sur-détection des rangées.
        """
        if not lines:
            return []

        # Trier par position Y
        sorted_lines = sorted(lines, key=lambda l: l[1])

        filtered = [sorted_lines[0]]
        for line in sorted_lines[1:]:
            # Garder seulement si la distance avec la dernière ligne est suffisante
            if line[1] - filtered[-1][1] >= min_dist:
                filtered.append(line)

        self.logger.debug(f"🔧 Lignes horizontales filtrées: {len(lines)} -> {len(filtered)}")
        return filtered

    def _create_cell_grid(self, h_lines: List[Tuple[int, int, int, int]],
                         v_lines: List[Tuple[int, int, int, int]],
                         img_shape: Tuple[int, int]) -> List[CellRegion]:
        """Créer la grille de cellules à partir des lignes"""

        if len(h_lines) < 2 or len(v_lines) < 2:
            return []

        # Extraire positions Y des lignes horizontales
        y_positions = sorted(set(line[1] for line in h_lines))

        # Extraire positions X des lignes verticales
        x_positions = sorted(set(line[0] for line in v_lines))

        cells = []
        row = 0

        for i in range(len(y_positions) - 1):
            col = 0
            y_start = y_positions[i]
            y_end = y_positions[i + 1]

            for j in range(len(x_positions) - 1):
                x_start = x_positions[j]
                x_end = x_positions[j + 1]

                # Calculer dimensions de la cellule
                width = x_end - x_start
                height = y_end - y_start

                # Filtrer cellules trop petites
                if width >= self.cell_min_size and height >= self.cell_min_size:
                    cell = CellRegion(
                        x=x_start,
                        y=y_start,
                        width=width,
                        height=height,
                        row=row,
                        col=col,
                        confidence=1.0
                    )
                    cells.append(cell)

                col += 1
            row += 1

        return cells

    def _fallback_grid_detection(self, img: np.ndarray) -> Optional[TableStructure]:
        """Méthode de fallback: grille uniforme basée sur l'analyse de contenu"""
        self.logger.info("🔄 Fallback: détection grille uniforme")

        height, width = img.shape

        # Analyser la distribution de contenu pour estimer la grille
        # Projections horizontales et verticales
        h_projection = np.sum(img == 0, axis=1)  # Pixels noirs par ligne
        v_projection = np.sum(img == 0, axis=0)  # Pixels noirs par colonne

        # Détecter espaces blancs (séparateurs probables)
        h_separators = self._find_separators(h_projection, min_gap=10)
        v_separators = self._find_separators(v_projection, min_gap=10)

        if len(h_separators) < 1 or len(v_separators) < 1:
            # Dernière option: grille fixe basée sur estimation
            estimated_rows = max(3, height // 80)
            estimated_cols = max(3, width // 100)

            h_separators = [i * (height // estimated_rows) for i in range(estimated_rows + 1)]
            v_separators = [i * (width // estimated_cols) for i in range(estimated_cols + 1)]

        # Créer cellules avec séparateurs estimés
        cells = []
        for i in range(len(h_separators) - 1):
            for j in range(len(v_separators) - 1):
                cell = CellRegion(
                    x=v_separators[j],
                    y=h_separators[i],
                    width=v_separators[j + 1] - v_separators[j],
                    height=h_separators[i + 1] - h_separators[i],
                    row=i,
                    col=j,
                    confidence=0.6  # Confiance plus faible
                )
                cells.append(cell)

        structure = TableStructure(
            cells=cells,
            num_rows=len(h_separators) - 1,
            num_cols=len(v_separators) - 1,
            table_bbox=(0, 0, width, height),
            horizontal_lines=[(0, y, width, y) for y in h_separators],
            vertical_lines=[(x, 0, x, height) for x in v_separators],
            confidence=0.6
        )

        self.logger.info(f"🔄 Fallback: {structure.num_rows}x{structure.num_cols}")
        return structure

    def _find_separators(self, projection: np.ndarray, min_gap: int = 5) -> List[int]:
        """Trouver séparateurs dans projection"""
        # Identifier zones avec peu de contenu (séparateurs)
        threshold = np.mean(projection) * 0.3
        low_content = projection < threshold

        # Trouver groupes consécutifs
        separators = [0]  # Début
        in_gap = False
        gap_start = 0

        for i, is_low in enumerate(low_content):
            if is_low and not in_gap:
                gap_start = i
                in_gap = True
            elif not is_low and in_gap:
                if i - gap_start >= min_gap:
                    separators.append((gap_start + i) // 2)
                in_gap = False

        separators.append(len(projection) - 1)  # Fin
        return sorted(list(set(separators)))

    def _calculate_table_bbox(self, h_lines: List[Tuple[int, int, int, int]],
                             v_lines: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Calculer bounding box du tableau"""
        if not h_lines or not v_lines:
            return (0, 0, 0, 0)

        # Coordonnées extrêmes
        min_x = min(line[0] for line in v_lines)
        max_x = max(line[2] for line in v_lines)
        min_y = min(line[1] for line in h_lines)
        max_y = max(line[3] for line in h_lines)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _calculate_structure_confidence(self, cells: List[CellRegion], img_shape: Tuple[int, int]) -> float:
        """Calculer confiance dans la structure détectée"""
        if not cells:
            return 0.0

        # Facteurs de confiance
        factors = []

        # 1. Régularité de la grille
        if len(cells) > 1:
            widths = [cell.width for cell in cells]
            heights = [cell.height for cell in cells]

            width_cv = np.std(widths) / np.mean(widths) if np.mean(widths) > 0 else 1.0
            height_cv = np.std(heights) / np.mean(heights) if np.mean(heights) > 0 else 1.0

            regularity = 1.0 / (1.0 + width_cv + height_cv)
            factors.append(regularity)

        # 2. Couverture de l'image
        total_area = sum(cell.width * cell.height for cell in cells)
        img_area = img_shape[0] * img_shape[1]
        coverage = min(1.0, total_area / img_area)
        factors.append(coverage)

        # 3. Taille raisonnable des cellules
        reasonable_size = sum(1 for cell in cells
                             if 20 <= cell.width <= img_shape[1] // 2
                             and 15 <= cell.height <= img_shape[0] // 2)
        size_factor = reasonable_size / len(cells) if cells else 0.0
        factors.append(size_factor)

        return np.mean(factors)

    def visualize_structure(self, image_path: str, structure: TableStructure, output_path: Optional[str] = None) -> str:
        """Visualiser la structure détectée (pour debug)"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image non lisible")

        # Dessiner lignes horizontales (rouge)
        for line in structure.horizontal_lines:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

        # Dessiner lignes verticales (bleu)
        for line in structure.vertical_lines:
            cv2.line(img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

        # Dessiner cellules (vert)
        for cell in structure.cells:
            cv2.rectangle(img, (cell.x, cell.y),
                         (cell.x + cell.width, cell.y + cell.height),
                         (0, 255, 0), 1)

            # Numéro de cellule
            cv2.putText(img, f"{cell.row},{cell.col}",
                       (cell.x + 5, cell.y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Sauvegarder
        if output_path is None:
            output_path = f"debug_structure_{Path(image_path).stem}.png"

        cv2.imwrite(output_path, img)
        return output_path

def main():
    """Test du service de détection de structure"""
    service = StructureDetectionService()

    # Test avec fichiers existants
    sample_files = Path("sample_data").glob("*.png")

    for img_path in sample_files:
        try:
            structure = service.detect_table_structure(str(img_path))

            if structure:
                print(f"✅ {img_path.name}: {structure.num_rows}x{structure.num_cols} (confiance: {structure.confidence:.2f})")

                # Créer visualisation
                debug_img = service.visualize_structure(str(img_path), structure)
                print(f"🔍 Debug: {debug_img}")

            else:
                print(f"❌ {img_path.name}: Aucune structure détectée")

        except Exception as e:
            print(f"❌ Erreur {img_path.name}: {e}")

if __name__ == "__main__":
    main()