#!/usr/bin/env python3
"""
SERVICE 3: OCR CELLULE PAR CELLULE
Utilise la structure détectée pour faire OCR précis sur chaque cellule
Supporte Tesseract et Kraken/TrOCR pour manuscrit
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import services précédents
from .service_2_structure_detection import CellRegion, TableStructure

# OCR engines
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

@dataclass
class CellContent:
    """Contenu extrait d'une cellule"""
    text: str
    confidence: float
    cell_region: CellRegion
    ocr_method: str
    preprocessing_applied: List[str]

class CellOCRService:
    """Service dédié à l'OCR cellule par cellule"""

    def __init__(self, temp_dir: str = "temp_cells"):
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Dossier temporaire pour cellules
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

        # Initialiser moteurs OCR
        self.ocr_engines = {}
        self._setup_ocr_engines()

        # Configuration preprocessing par cellule
        self.cell_preprocessing_configs = {
            'text_cells': {
                'resize_factor': 2.0,
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': False
            },
            'number_cells': {
                'resize_factor': 3.0,
                'enhance_contrast': True,
                'denoise': False,
                'sharpen': True
            },
            'mixed_cells': {
                'resize_factor': 2.5,
                'enhance_contrast': True,
                'denoise': True,
                'sharpen': True
            }
        }

    def _setup_ocr_engines(self):
        """Configurer les moteurs OCR disponibles"""

        # 1. Tesseract pour texte imprimé
        if TESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                self.ocr_engines['tesseract'] = {
                    'engine': pytesseract,
                    'configs': {
                        'text': '--oem 3 --psm 8 -l fra+eng',  # Single word
                        'numbers': '--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.,',  # Numbers only
                        'mixed': '--oem 3 --psm 6 -l fra+eng'  # Uniform block
                    }
                }
                self.logger.info("✅ Tesseract configuré")
            except Exception as e:
                self.logger.warning(f"Tesseract non disponible: {e}")

        # 2. EasyOCR pour texte robuste
        if EASYOCR_AVAILABLE:
            try:
                reader = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
                self.ocr_engines['easyocr'] = {
                    'engine': reader,
                    'configs': {'default': {}}
                }
                self.logger.info("✅ EasyOCR configuré")
            except Exception as e:
                self.logger.warning(f"EasyOCR non disponible: {e}")

        if not self.ocr_engines:
            self.logger.error("❌ Aucun moteur OCR disponible!")

    def extract_cells_content(self, image_path: str, structure: TableStructure) -> List[CellContent]:
        """Extraire le contenu de toutes les cellules"""
        self.logger.info(f"📝 OCR cellule par cellule: {len(structure.cells)} cellules")

        # Charger l'image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image non lisible: {image_path}")

        results = []

        for i, cell in enumerate(structure.cells):
            try:
                # Extraire et préprocesser la cellule
                cell_img = self._extract_cell_image(img, cell)
                cell_type = self._detect_cell_type(cell_img)
                processed_cell = self._preprocess_cell(cell_img, cell_type)

                # OCR avec meilleure méthode disponible
                content = self._ocr_single_cell(processed_cell, cell_type, cell)

                if content:
                    results.append(content)
                    if i % 10 == 0:  # Log progression
                        self.logger.debug(f"Traité {i + 1}/{len(structure.cells)} cellules")

            except Exception as e:
                self.logger.warning(f"Erreur cellule ({cell.row}, {cell.col}): {e}")
                # Ajouter cellule vide en cas d'erreur
                results.append(CellContent(
                    text="",
                    confidence=0.0,
                    cell_region=cell,
                    ocr_method="error",
                    preprocessing_applied=["error"]
                ))

        self.logger.info(f"✅ OCR terminé: {len(results)} cellules traitées")
        return results

    def _extract_cell_image(self, img: np.ndarray, cell: CellRegion) -> np.ndarray:
        """Extraire l'image d'une cellule spécifique"""
        # Ajouter padding pour éviter de couper le texte
        padding = 5
        x_start = max(0, cell.x - padding)
        y_start = max(0, cell.y - padding)
        x_end = min(img.shape[1], cell.x + cell.width + padding)
        y_end = min(img.shape[0], cell.y + cell.height + padding)

        cell_img = img[y_start:y_end, x_start:x_end]

        # Vérifier taille minimale
        if cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
            # Cellule trop petite, retourner image vide
            return np.full((20, 50), 255, dtype=np.uint8)

        return cell_img

    def _detect_cell_type(self, cell_img: np.ndarray) -> str:
        """Détecter le type de contenu probable de la cellule"""

        # Analyser la densité et distribution des pixels
        if cell_img.size == 0:
            return 'empty'

        # Calculer statistiques
        non_white_pixels = np.sum(cell_img < 200)
        total_pixels = cell_img.size
        density = non_white_pixels / total_pixels

        if density < 0.02:
            return 'empty'

        # Analyser la forme du contenu
        # Détecter contours pour analyser la forme
        contours, _ = cv2.findContours(
            (cell_img < 200).astype(np.uint8) * 255,
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return 'empty'

        # Analyser les bounding boxes des contours
        boxes = [cv2.boundingRect(c) for c in contours]
        heights = [box[3] for box in boxes]
        widths = [box[2] for box in boxes]

        if not heights or not widths:
            return 'mixed'

        avg_height = np.mean(heights)
        avg_width = np.mean(widths)
        aspect_ratio = avg_width / avg_height if avg_height > 0 else 1.0

        # Heuristiques pour classification
        if avg_height < 15 and aspect_ratio < 1.5 and len(contours) <= 3:
            return 'numbers'
        elif avg_height > 20 and aspect_ratio > 2.0:
            return 'text'
        else:
            return 'mixed'

    def _preprocess_cell(self, cell_img: np.ndarray, cell_type: str) -> np.ndarray:
        """Préprocesser une cellule selon son type"""

        config = self.cell_preprocessing_configs.get(f"{cell_type}_cells",
                                                    self.cell_preprocessing_configs['mixed_cells'])

        processed = cell_img.copy()
        applied_preprocessing = []

        # 1. Redimensionnement pour améliorer OCR
        if config['resize_factor'] != 1.0:
            new_height = int(processed.shape[0] * config['resize_factor'])
            new_width = int(processed.shape[1] * config['resize_factor'])
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            applied_preprocessing.append('resize')

        # 2. Amélioration contraste
        if config['enhance_contrast']:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            processed = clahe.apply(processed)
            applied_preprocessing.append('contrast')

        # 3. Débruitage
        if config['denoise']:
            processed = cv2.bilateralFilter(processed, 9, 75, 75)
            applied_preprocessing.append('denoise')

        # 4. Amélioration netteté
        if config['sharpen']:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            processed = cv2.filter2D(processed, -1, kernel)
            applied_preprocessing.append('sharpen')

        # 5. Binarisation finale
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        applied_preprocessing.append('binarize')

        return processed

    def _ocr_single_cell(self, cell_img: np.ndarray, cell_type: str, cell_region: CellRegion) -> Optional[CellContent]:
        """Faire OCR sur une seule cellule"""

        best_result = None
        best_confidence = 0.0

        # Essayer différents moteurs OCR
        for engine_name, engine_config in self.ocr_engines.items():
            try:
                result = self._apply_ocr_engine(cell_img, engine_name, engine_config, cell_type)

                if result and result['confidence'] > best_confidence:
                    best_confidence = result['confidence']
                    best_result = {
                        'text': result['text'],
                        'confidence': result['confidence'],
                        'method': engine_name
                    }

            except Exception as e:
                self.logger.debug(f"Moteur {engine_name} échoué: {e}")

        if best_result:
            return CellContent(
                text=best_result['text'].strip(),
                confidence=best_confidence,
                cell_region=cell_region,
                ocr_method=best_result['method'],
                preprocessing_applied=[cell_type + "_optimized"]
            )

        return None

    def _apply_ocr_engine(self, cell_img: np.ndarray, engine_name: str,
                         engine_config: dict, cell_type: str) -> Optional[Dict]:
        """Appliquer un moteur OCR spécifique"""

        if engine_name == 'tesseract':
            return self._tesseract_ocr(cell_img, engine_config, cell_type)
        elif engine_name == 'easyocr':
            return self._easyocr_ocr(cell_img, engine_config)

        return None

    def _tesseract_ocr(self, cell_img: np.ndarray, engine_config: dict, cell_type: str) -> Optional[Dict]:
        """OCR avec Tesseract"""

        # Choisir configuration selon type
        if cell_type == 'numbers':
            config = engine_config['configs']['numbers']
        elif cell_type == 'text':
            config = engine_config['configs']['text']
        else:
            config = engine_config['configs']['mixed']

        # Sauvegarder temporairement
        temp_path = self.temp_dir / f"temp_cell_{np.random.randint(10000)}.png"
        cv2.imwrite(str(temp_path), cell_img)

        try:
            # OCR avec données de confiance
            data = pytesseract.image_to_data(
                str(temp_path), config=config,
                output_type=pytesseract.Output.DICT
            )

            # Extraire texte et confiance
            words = []
            confidences = []

            for i, word in enumerate(data['text']):
                if word.strip() and int(data['conf'][i]) > 0:
                    words.append(word.strip())
                    confidences.append(int(data['conf'][i]))

            if words:
                text = ' '.join(words)
                avg_confidence = np.mean(confidences) / 100.0  # Normaliser 0-1
                return {'text': text, 'confidence': avg_confidence}

        finally:
            # Nettoyer fichier temporaire
            if temp_path.exists():
                temp_path.unlink()

        return None

    def _easyocr_ocr(self, cell_img: np.ndarray, engine_config: dict) -> Optional[Dict]:
        """OCR avec EasyOCR"""

        try:
            reader = engine_config['engine']

            # Convertir pour EasyOCR
            if len(cell_img.shape) == 3:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
            else:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2RGB)

            # OCR
            results = reader.readtext(cell_img, paragraph=False)

            if results:
                # Combiner tout le texte détecté
                texts = []
                confidences = []

                for (bbox, text, conf) in results:
                    if text.strip() and conf > 0.3:
                        texts.append(text.strip())
                        confidences.append(conf)

                if texts:
                    combined_text = ' '.join(texts)
                    avg_confidence = np.mean(confidences)
                    return {'text': combined_text, 'confidence': avg_confidence}

        except Exception as e:
            self.logger.debug(f"EasyOCR erreur: {e}")

        return None

    def create_cell_extraction_report(self, cell_contents: List[CellContent],
                                    structure: TableStructure) -> Dict:
        """Créer rapport détaillé de l'extraction"""

        report = {
            'total_cells': len(structure.cells),
            'extracted_cells': len([c for c in cell_contents if c.text.strip()]),
            'empty_cells': len([c for c in cell_contents if not c.text.strip()]),
            'average_confidence': np.mean([c.confidence for c in cell_contents]),
            'ocr_methods_used': {},
            'cell_types_detected': {},
            'extraction_quality': {}
        }

        # Analyser méthodes OCR utilisées
        for content in cell_contents:
            method = content.ocr_method
            report['ocr_methods_used'][method] = report['ocr_methods_used'].get(method, 0) + 1

        # Analyser qualité par confiance
        high_conf = len([c for c in cell_contents if c.confidence > 0.8])
        med_conf = len([c for c in cell_contents if 0.5 <= c.confidence <= 0.8])
        low_conf = len([c for c in cell_contents if 0 < c.confidence < 0.5])

        report['extraction_quality'] = {
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf,
            'failed': len(cell_contents) - high_conf - med_conf - low_conf
        }

        return report

    def export_cell_images_debug(self, image_path: str, structure: TableStructure,
                                output_dir: str = "debug_cells"):
        """Exporter images de cellules individuelles pour debug"""
        debug_dir = Path(output_dir)
        debug_dir.mkdir(exist_ok=True)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        for cell in structure.cells:
            cell_img = self._extract_cell_image(img, cell)

            # Sauvegarder cellule originale
            original_path = debug_dir / f"cell_{cell.row}_{cell.col}_original.png"
            cv2.imwrite(str(original_path), cell_img)

            # Sauvegarder cellule préprocessée
            cell_type = self._detect_cell_type(cell_img)
            processed_cell = self._preprocess_cell(cell_img, cell_type)
            processed_path = debug_dir / f"cell_{cell.row}_{cell.col}_processed_{cell_type}.png"
            cv2.imwrite(str(processed_path), processed_cell)

        self.logger.info(f"🔍 Images debug sauvées dans {debug_dir}")

def main():
    """Test du service OCR cellule par cellule"""

    # Import du service de structure
    from service_2_structure_detection import StructureDetectionService

    structure_service = StructureDetectionService()
    ocr_service = CellOCRService()

    # Test avec images existantes
    sample_files = Path("sample_data").glob("*.png")

    for img_path in sample_files:
        try:
            print(f"\n🔍 Test OCR: {img_path.name}")

            # 1. Détecter structure
            structure = structure_service.detect_table_structure(str(img_path))
            if not structure:
                print(f"❌ Aucune structure détectée")
                continue

            print(f"📊 Structure: {structure.num_rows}x{structure.num_cols}")

            # 2. Extraire contenu cellules
            cell_contents = ocr_service.extract_cells_content(str(img_path), structure)

            # 3. Rapport
            report = ocr_service.create_cell_extraction_report(cell_contents, structure)
            print(f"✅ Extraction: {report['extracted_cells']}/{report['total_cells']} cellules")
            print(f"📈 Confiance moyenne: {report['average_confidence']:.2f}")

            # 4. Aperçu contenu
            print("📝 Aperçu contenu:")
            for content in cell_contents[:10]:  # Premiers 10
                if content.text.strip():
                    print(f"  ({content.cell_region.row},{content.cell_region.col}): '{content.text}' (conf: {content.confidence:.2f})")

            # 5. Export debug
            ocr_service.export_cell_images_debug(str(img_path), structure)

        except Exception as e:
            print(f"❌ Erreur {img_path.name}: {e}")

if __name__ == "__main__":
    main()