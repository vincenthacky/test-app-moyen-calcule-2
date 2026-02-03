#!/usr/bin/env python3
"""
EXTRACTION OPTIMISÉE AVEC OCR PERFORMANT
Configuration OCR spécialement optimisée pour tableaux de notes
Preprocessing d'image pour améliorer la qualité d'extraction
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image as PILImage, ImageEnhance, ImageFilter

try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except Exception:
    IMG2TABLE_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class OptimizedOCRExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Vérifier disponibilité Tesseract
        self.tesseract_available = self._check_tesseract()

    def _check_tesseract(self):
        """Vérifier si Tesseract fonctionne"""
        try:
            if PYTESSERACT_AVAILABLE:
                version = pytesseract.get_tesseract_version()
                print(f"✅ Tesseract {version}")
                return True
        except Exception as e:
            print(f"❌ Tesseract inaccessible: {e}")
        return False

    def preprocess_image_for_ocr(self, image_path):
        """Préprocesser l'image pour améliorer l'OCR"""
        print("🔧 Préprocessing image pour OCR...")

        try:
            # Charger l'image
            img = cv2.imread(str(image_path))

            # Convertir en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Augmenter la résolution si nécessaire
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale_factor = max(2.0, 1000 / min(height, width))
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"  📐 Redimensionné: {width}x{height} -> {new_width}x{new_height}")

            # Amélioration du contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Débruitage
            denoised = cv2.medianBlur(enhanced, 3)

            # Amélioration netteté
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            # Sauvegarder l'image préprocessée
            processed_path = self.output_dir / f"processed_{Path(image_path).name}"
            cv2.imwrite(str(processed_path), binary)
            print(f"  💾 Image préprocessée: {processed_path}")

            return str(processed_path)

        except Exception as e:
            print(f"  ❌ Erreur preprocessing: {e}")
            return str(image_path)  # Retourner l'original si échec

    def extract_with_optimized_tesseract(self, image_path):
        """Extraction avec Tesseract optimisé"""
        if not self.tesseract_available:
            return None

        print(f"🔍 Tesseract optimisé: {Path(image_path).name}")

        try:
            # Préprocesser l'image
            processed_image = self.preprocess_image_for_ocr(image_path)

            # Charger image
            img = PILImage.open(processed_image)

            # Configurations optimisées pour différents types de tableaux
            configs = [
                {
                    'name': 'table_optimized',
                    'config': '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ .,/()-'
                },
                {
                    'name': 'french_names',
                    'config': '--oem 3 --psm 4 -l fra+eng'
                },
                {
                    'name': 'numbers_focused',
                    'config': '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,'
                }
            ]

            best_result = None
            max_content = 0

            for config in configs:
                try:
                    print(f"  🧪 Test config: {config['name']}")

                    # Extraction données avec position
                    data = pytesseract.image_to_data(
                        img,
                        config=config['config'],
                        output_type=pytesseract.Output.DICT,
                        lang='fra+eng'
                    )

                    # Construire tableau à partir des données
                    table_data = self._build_table_from_positioned_data(data)

                    if table_data and len(table_data) > 1:  # Au moins header + 1 ligne
                        content_count = sum(len(str(cell).strip()) for row in table_data for cell in row)

                        print(f"    📊 {len(table_data)} lignes, {content_count} caractères")

                        if content_count > max_content:
                            max_content = content_count
                            df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                            best_result = {
                                'method': f'tesseract_{config["name"]}',
                                'dataframe': df,
                                'confidence': 'high',
                                'char_count': content_count,
                                'config_used': config['name']
                            }

                except Exception as e:
                    print(f"    ❌ Config {config['name']} échouée: {e}")

            if best_result:
                print(f"  ✅ Meilleur résultat: {best_result['config_used']}")
                print(f"  📋 Données extraites:")
                df = best_result['dataframe']
                print(df.head().to_string(index=False, max_cols=6))

            return best_result

        except Exception as e:
            print(f"  ❌ Erreur Tesseract: {e}")
            return None

    def extract_with_img2table_enhanced(self, image_path):
        """Extraction avec img2table + OCR amélioré"""
        if not IMG2TABLE_AVAILABLE or not self.tesseract_available:
            return None

        print(f"📊 img2table avec OCR amélioré: {Path(image_path).name}")

        try:
            # Préprocesser l'image
            processed_image = self.preprocess_image_for_ocr(image_path)

            # Configuration OCR optimisée
            ocr = TesseractOCR(
                n_threads=2,
                lang="fra+eng",
                psm=6,  # Uniform block of text
                oem=3   # Default, based on what is available
            )

            # Document avec image préprocessée
            doc = Img2TableImage(src=processed_image, detect_rotation=True)

            # Extraction avec différents paramètres
            extraction_params = [
                {
                    'implicit_rows': True,
                    'borderless_tables': True,
                    'min_confidence': 20
                },
                {
                    'implicit_rows': True,
                    'borderless_tables': False,
                    'min_confidence': 30
                },
                {
                    'implicit_rows': False,
                    'borderless_tables': True,
                    'min_confidence': 40
                }
            ]

            best_extraction = None
            max_cells_with_content = 0

            for i, params in enumerate(extraction_params):
                try:
                    print(f"  🧪 Paramètres {i+1}: {params}")

                    tables = doc.extract_tables(ocr=ocr, **params)

                    if tables:
                        table = tables[0]  # Premier tableau
                        df = table.df

                        # Nettoyer
                        df_cleaned = self._clean_ocr_dataframe(df)

                        if not df_cleaned.empty:
                            # Compter cellules avec contenu
                            content_cells = (df_cleaned != '').sum().sum()
                            print(f"    📊 {df_cleaned.shape}, {content_cells} cellules avec contenu")

                            if content_cells > max_cells_with_content:
                                max_cells_with_content = content_cells
                                best_extraction = {
                                    'method': 'img2table_enhanced',
                                    'dataframe': df_cleaned,
                                    'confidence': 'high',
                                    'bbox': {
                                        'x1': table.bbox.x1, 'y1': table.bbox.y1,
                                        'x2': table.bbox.x2, 'y2': table.bbox.y2
                                    },
                                    'content_cells': content_cells,
                                    'params_used': i+1
                                }

                except Exception as e:
                    print(f"    ❌ Paramètres {i+1} échoués: {e}")

            if best_extraction:
                print(f"  ✅ Meilleur: paramètres {best_extraction['params_used']}")
                df = best_extraction['dataframe']
                print(f"  📋 Résultat final:")
                print(df.head().to_string(index=False, max_cols=6))

            return best_extraction

        except Exception as e:
            print(f"  ❌ Erreur img2table: {e}")
            return None

    def _build_table_from_positioned_data(self, data):
        """Construire tableau à partir des données Tesseract avec positions"""
        words = []

        # Filtrer et organiser les mots par position
        for i, text in enumerate(data['text']):
            conf = int(data['conf'][i])
            if text.strip() and conf > 20:  # Seuil de confiance bas pour plus de données
                words.append({
                    'text': text.strip(),
                    'x': data['left'][i] + data['width'][i] // 2,
                    'y': data['top'][i] + data['height'][i] // 2,
                    'conf': conf,
                    'left': data['left'][i],
                    'top': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                })

        if len(words) < 3:
            return None

        # Trier par position Y puis X
        words.sort(key=lambda w: (w['y'], w['x']))

        # Grouper en lignes avec tolérance plus fine
        rows = []
        current_row = []
        last_y = -1
        y_tolerance = 25  # Tolérance pour même ligne

        for word in words:
            if last_y == -1 or abs(word['y'] - last_y) <= y_tolerance:
                current_row.append(word)
                # Moyenne pondérée des Y
                if last_y == -1:
                    last_y = word['y']
                else:
                    last_y = (last_y + word['y']) // 2
            else:
                if current_row:
                    # Trier la ligne par position X
                    current_row.sort(key=lambda w: w['x'])
                    row_texts = []

                    # Regrouper mots proches en cellules
                    cells = []
                    current_cell = []
                    last_x = -1

                    for word in current_row:
                        if last_x == -1 or abs(word['x'] - last_x) <= 50:  # Mots de même cellule
                            current_cell.append(word['text'])
                            last_x = word['x']
                        else:
                            if current_cell:
                                cells.append(' '.join(current_cell))
                            current_cell = [word['text']]
                            last_x = word['x']

                    if current_cell:
                        cells.append(' '.join(current_cell))

                    rows.append(cells)

                current_row = [word]
                last_y = word['y']

        # Ajouter dernière ligne
        if current_row:
            current_row.sort(key=lambda w: w['x'])
            cells = []
            current_cell = []
            last_x = -1

            for word in current_row:
                if last_x == -1 or abs(word['x'] - last_x) <= 50:
                    current_cell.append(word['text'])
                    last_x = word['x']
                else:
                    if current_cell:
                        cells.append(' '.join(current_cell))
                    current_cell = [word['text']]
                    last_x = word['x']

            if current_cell:
                cells.append(' '.join(current_cell))

            rows.append(cells)

        # Égaliser nombre de colonnes
        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows if len(rows) >= 2 else None

    def _clean_ocr_dataframe(self, df):
        """Nettoyer DataFrame OCR"""
        if df.empty:
            return df

        # Remplacer None/NaN
        df = df.fillna('')

        # Nettoyer le texte
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Supprimer caractères parasites courants de l'OCR
            df[col] = df[col].str.replace(r'[|_\[\]\'\"]+', '', regex=True)

        # Supprimer lignes vides
        mask = (df != '').any(axis=1)
        df = df[mask].reset_index(drop=True)

        # Supprimer colonnes vides
        mask_cols = (df != '').any(axis=0)
        df = df.loc[:, mask_cols]

        return df

    def process_image_optimized(self, image_path):
        """Traitement optimisé d'une image"""
        print(f"\n🎯 Traitement optimisé: {Path(image_path).name}")

        results = []

        # 1. Essayer img2table amélioré
        result1 = self.extract_with_img2table_enhanced(image_path)
        if result1:
            results.append(result1)

        # 2. Essayer Tesseract optimisé
        result2 = self.extract_with_optimized_tesseract(image_path)
        if result2:
            results.append(result2)

        # Choisir le meilleur résultat
        if results:
            best = max(results, key=lambda r: r.get('content_cells', r.get('char_count', 0)))
            print(f"\n✅ Meilleur résultat: {best['method']}")
            return best

        return None

    def export_optimized_results(self, results, filename="optimized_extraction"):
        """Export résultats optimisés"""
        if not results:
            return []

        files_created = []

        # Export technique
        result_data = {
            'file': results.get('source_file', 'unknown'),
            'method': results['method'],
            'confidence': results['confidence'],
            'shape': {
                'rows': results['dataframe'].shape[0],
                'cols': results['dataframe'].shape[1]
            },
            'data': results['dataframe'].to_dict('records'),
            'metadata': {
                'config_used': results.get('config_used', results.get('params_used', 'unknown')),
                'content_quality': results.get('content_cells', results.get('char_count', 0))
            }
        }

        # JSON
        json_file = self.output_dir / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        files_created.append(json_file)
        print(f"📄 Export JSON: {json_file}")

        # CSV
        csv_file = self.output_dir / f"{filename}.csv"
        results['dataframe'].to_csv(csv_file, index=False, encoding='utf-8')
        files_created.append(csv_file)
        print(f"📊 Export CSV: {csv_file}")

        # Rapport détaillé
        report_file = self.output_dir / f"{filename}_detailed_report.txt"
        self._generate_detailed_report(result_data, report_file)
        files_created.append(report_file)

        return files_created

    def _generate_detailed_report(self, data, output_path):
        """Générer rapport détaillé"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT D'EXTRACTION OPTIMISÉE ===\n\n")
            f.write(f"Fichier: {data['file']}\n")
            f.write(f"Méthode: {data['method']}\n")
            f.write(f"Configuration: {data['metadata']['config_used']}\n")
            f.write(f"Confiance: {data['confidence']}\n")
            f.write(f"Dimensions: {data['shape']['rows']} x {data['shape']['cols']}\n")
            f.write(f"Qualité contenu: {data['metadata']['content_quality']}\n\n")

            f.write("DONNÉES EXTRAITES:\n")
            for i, row in enumerate(data['data'], 1):
                f.write(f"Ligne {i}: {list(row.values())}\n")

def main():
    """Fonction principale"""
    print("🚀 EXTRACTION OCR OPTIMISÉE")
    print("=" * 50)

    extractor = OptimizedOCRExtractor()

    if not extractor.tesseract_available:
        print("❌ Tesseract requis!")
        return

    # Chercher fichiers
    sample_dir = Path("sample_data")
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

    if not image_files:
        print("❌ Aucun fichier trouvé!")
        return

    # Traiter chaque image
    for img_path in image_files:
        result = extractor.process_image_optimized(str(img_path))

        if result:
            # Ajouter info fichier source
            result['source_file'] = img_path.name

            # Export
            files = extractor.export_optimized_results(result)

            print(f"\n✅ EXTRACTION OPTIMISÉE TERMINÉE!")
            print(f"📁 {len(files)} fichier(s) créé(s)")
            print(f"📊 Résultat: {result['dataframe'].shape[0]} lignes x {result['dataframe'].shape[1]} colonnes")
        else:
            print("\n❌ Aucune extraction réussie")

if __name__ == "__main__":
    main()