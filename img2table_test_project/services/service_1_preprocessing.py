#!/usr/bin/env python3
"""
SERVICE 1: PRÉTRAITEMENT D'IMAGES/PDF
Redresse, améliore contraste, binarise et convertit les documents
Pour optimiser l'OCR en amont
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Tuple, Optional

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

class PreprocessingService:
    """Service dédié au prétraitement robuste d'images et PDFs"""

    def __init__(self, output_dir: str = "processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Paramètres de preprocessing
        self.min_resolution = 1000  # Résolution minimale recommandée
        self.max_resolution = 3000  # Résolution maximale (éviter surcharge)

    def process_document(self, file_path: str) -> str:
        """Point d'entrée principal: traite PDF ou image"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        self.logger.info(f"🔧 Prétraitement: {file_path.name}")

        # Traiter selon type de fichier
        if file_path.suffix.lower() == '.pdf':
            return self._process_pdf(file_path)
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Format non supporté: {file_path.suffix}")

    def _process_pdf(self, pdf_path: Path) -> str:
        """Convertir PDF en image optimisée"""
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError("pdf2image requis pour les PDFs")

        self.logger.info(f"📄 Conversion PDF: {pdf_path.name}")

        try:
            # Convertir PDF en images (première page)
            pages = pdf2image.convert_from_path(
                pdf_path,
                dpi=300,  # DPI élevé pour qualité OCR
                first_page=1,
                last_page=1,
                fmt='PNG'
            )

            if not pages:
                raise ValueError("PDF vide ou non lisible")

            # Sauvegarder temporairement
            temp_image_path = self.output_dir / f"temp_{pdf_path.stem}.png"
            pages[0].save(temp_image_path)

            # Traiter l'image convertie
            return self._process_image(temp_image_path)

        except Exception as e:
            self.logger.error(f"Erreur conversion PDF: {e}")
            raise

    def _process_image(self, image_path: Path) -> str:
        """Pipeline complet de traitement d'image"""
        self.logger.info(f"🖼️ Traitement image: {image_path.name}")

        # Charger l'image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Image non lisible: {image_path}")

        # Pipeline de preprocessing
        img = self._correct_perspective(img)
        img = self._optimize_resolution(img)
        img = self._enhance_contrast(img)
        img = self._enhance_sharpness(img)
        img = self._binarize_adaptive(img)
        img = self._remove_noise(img)

        # Sauvegarder résultat
        output_path = self.output_dir / f"processed_{image_path.name}"
        cv2.imwrite(str(output_path), img)

        self.logger.info(f"✅ Image préparée: {output_path}")
        return str(output_path)

    def _correct_perspective(self, img: np.ndarray) -> np.ndarray:
        """Corriger la perspective si l'image est prise de biais"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détecter les contours pour trouver le document
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.logger.warning("Aucun contour détecté - pas de correction perspective")
            return img

        # Trouver le plus grand rectangle (probable document)
        largest_contour = max(contours, key=cv2.contourArea)

        # Approximation polygonale
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Si on a un quadrilatère, corriger la perspective
        if len(approx) == 4:
            self.logger.info("📐 Correction perspective détectée")
            return self._apply_perspective_correction(img, approx)

        return img

    def _apply_perspective_correction(self, img: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Appliquer correction de perspective"""
        # Ordonner les coins (top-left, top-right, bottom-right, bottom-left)
        corners = corners.reshape(4, 2)
        ordered_corners = np.zeros((4, 2), dtype=np.float32)

        # Somme et différence pour identifier les coins
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)

        ordered_corners[0] = corners[np.argmin(s)]      # top-left
        ordered_corners[2] = corners[np.argmax(s)]      # bottom-right
        ordered_corners[1] = corners[np.argmin(diff)]   # top-right
        ordered_corners[3] = corners[np.argmax(diff)]   # bottom-left

        # Calculer dimensions de destination
        width = max(
            np.linalg.norm(ordered_corners[0] - ordered_corners[1]),
            np.linalg.norm(ordered_corners[2] - ordered_corners[3])
        )
        height = max(
            np.linalg.norm(ordered_corners[0] - ordered_corners[3]),
            np.linalg.norm(ordered_corners[1] - ordered_corners[2])
        )

        # Points de destination (rectangle parfait)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # Transformation de perspective
        matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        corrected = cv2.warpPerspective(img, matrix, (int(width), int(height)))

        return corrected

    def _optimize_resolution(self, img: np.ndarray) -> np.ndarray:
        """Optimiser la résolution pour OCR"""
        height, width = img.shape[:2]
        current_resolution = min(height, width)

        if current_resolution < self.min_resolution:
            # Upscale si trop petit
            scale_factor = self.min_resolution / current_resolution
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            self.logger.info(f"📈 Upscale: {width}x{height} -> {new_width}x{new_height}")
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        elif current_resolution > self.max_resolution:
            # Downscale si trop grand
            scale_factor = self.max_resolution / current_resolution
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            self.logger.info(f"📉 Downscale: {width}x{height} -> {new_width}x{new_height}")
            return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return img

    def _enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """Améliorer le contraste avec CLAHE adaptatif"""
        # Convertir en niveaux de gris si nécessaire
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        self.logger.debug("🔆 Amélioration contraste appliquée")
        return enhanced

    def _enhance_sharpness(self, img: np.ndarray) -> np.ndarray:
        """Améliorer la netteté pour l'OCR"""
        # Kernel pour augmenter la netteté
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])

        sharpened = cv2.filter2D(img, -1, kernel)

        # Mélanger avec l'original pour éviter le sur-traitement
        alpha = 0.7  # Poids de l'image nettoyée
        result = cv2.addWeighted(img, 1 - alpha, sharpened, alpha, 0)

        self.logger.debug("🔍 Amélioration netteté appliquée")
        return result

    def _binarize_adaptive(self, img: np.ndarray) -> np.ndarray:
        """Binarisation adaptative optimisée pour OCR"""
        # Binarisation adaptative Gaussian
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )

        # Alternative: binarisation Otsu pour comparaison
        _, binary_otsu = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Choisir la meilleure méthode basée sur le contraste
        adaptive_variance = np.var(binary)
        otsu_variance = np.var(binary_otsu)

        if adaptive_variance > otsu_variance:
            self.logger.debug("🎯 Binarisation adaptative sélectionnée")
            return binary
        else:
            self.logger.debug("🎯 Binarisation Otsu sélectionnée")
            return binary_otsu

    def _remove_noise(self, img: np.ndarray) -> np.ndarray:
        """Supprimer le bruit tout en préservant le texte"""
        # Morphologie pour nettoyer le bruit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Opening pour supprimer petits objets
        cleaned = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing pour fermer les gaps dans les lettres
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.logger.debug("🧹 Suppression bruit appliquée")
        return cleaned

    def get_preprocessing_stats(self, original_path: str, processed_path: str) -> dict:
        """Obtenir statistiques de prétraitement"""
        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        processed = cv2.imread(processed_path, cv2.IMREAD_GRAYSCALE)

        if original is None or processed is None:
            return {}

        return {
            'original_size': original.shape,
            'processed_size': processed.shape,
            'original_contrast': np.std(original),
            'processed_contrast': np.std(processed),
            'improvement_ratio': np.std(processed) / np.std(original) if np.std(original) > 0 else 1.0
        }

def main():
    """Test du service de prétraitement"""
    service = PreprocessingService()

    # Exemple d'utilisation
    sample_files = Path("sample_data").glob("*")

    for file_path in sample_files:
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf']:
            try:
                processed_path = service.process_document(str(file_path))
                print(f"✅ {file_path.name} -> {processed_path}")

                # Statistiques
                stats = service.get_preprocessing_stats(str(file_path), processed_path)
                if stats:
                    print(f"📊 Amélioration contraste: {stats['improvement_ratio']:.2f}x")

            except Exception as e:
                print(f"❌ Erreur {file_path.name}: {e}")

if __name__ == "__main__":
    main()