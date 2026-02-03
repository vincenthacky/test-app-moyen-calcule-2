#!/usr/bin/env python3
"""
Projet img2table OPTIMISÉ - Extraction COMPLÈTE avec multiple OCR
Extrait le VRAI TEXTE des tableaux avec fallbacks OCR multiples
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image as PILImage

# Imports conditionnels pour OCR
try:
    from img2table.document import Image, PDF
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except Exception as e:
    print(f"⚠️ img2table non disponible: {e}")
    IMG2TABLE_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

class AdvancedTableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Initialiser multiple OCR
        self.ocr_engines = {}
        self._setup_ocr_engines()

    def _setup_ocr_engines(self):
        """Configurer tous les moteurs OCR disponibles"""

        # 1. img2table + Tesseract
        if IMG2TABLE_AVAILABLE:
            try:
                self.ocr_engines['img2table_tesseract'] = TesseractOCR(n_threads=2, lang="fra+eng")
                print("✅ img2table + Tesseract configuré")
            except Exception as e:
                print(f"⚠️ img2table + Tesseract non disponible: {e}")

        # 2. EasyOCR (très performant)
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_engines['easyocr'] = easyocr.Reader(['fr', 'en'], gpu=False)
                print("✅ EasyOCR configuré")
            except Exception as e:
                print(f"⚠️ EasyOCR non disponible: {e}")

        # 3. Pytesseract direct
        if PYTESSERACT_AVAILABLE:
            try:
                # Test si tesseract est accessible
                pytesseract.get_tesseract_version()
                self.ocr_engines['pytesseract'] = True
                print("✅ Pytesseract configuré")
            except Exception as e:
                print(f"⚠️ Pytesseract non disponible: {e}")

        if not self.ocr_engines:
            print("❌ AUCUN moteur OCR disponible!")
            print("Installation recommandée:")
            print("  - brew install tesseract tesseract-lang")
            print("  - pip install easyocr pytesseract")

    def extract_with_img2table(self, file_path, file_type="image"):
        """Extraction avec img2table + OCR intégré"""
        if not IMG2TABLE_AVAILABLE or 'img2table_tesseract' not in self.ocr_engines:
            return None

        print(f"📊 Extraction img2table: {file_path}")

        try:
            if file_type == "image":
                doc = Image(src=file_path, detect_rotation=True)
            else:
                doc = PDF(src=file_path, pages=[0])

            # Extraction avec OCR
            extracted_tables = doc.extract_tables(
                ocr=self.ocr_engines['img2table_tesseract'],
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=50
            )

            results = []
            for i, table in enumerate(extracted_tables):
                df = table.df
                print(f"  📋 Tableau {i+1}: {df.shape[0]}x{df.shape[1]}")

                # Nettoyer les données
                df_cleaned = self._clean_dataframe(df)

                result = {
                    'method': 'img2table',
                    'table_id': i+1,
                    'bbox': {
                        'x1': table.bbox.x1, 'y1': table.bbox.y1,
                        'x2': table.bbox.x2, 'y2': table.bbox.y2
                    },
                    'shape': {'rows': df_cleaned.shape[0], 'cols': df_cleaned.shape[1]},
                    'dataframe': df_cleaned,
                    'confidence': 'high'
                }
                results.append(result)

            return results

        except Exception as e:
            print(f"  ❌ Erreur img2table: {e}")
            return None

    def extract_with_easyocr(self, image_path):
        """Extraction avec EasyOCR (méthode fallback performante)"""
        if 'easyocr' not in self.ocr_engines:
            return None

        print(f"🔍 Extraction EasyOCR: {image_path}")

        try:
            # Charger l'image
            img = cv2.imread(str(image_path))
            if img is None:
                return None

            # OCR avec EasyOCR
            reader = self.ocr_engines['easyocr']
            results = reader.readtext(img, paragraph=False)

            # Convertir en structure tabulaire
            table_data = self._ocr_to_table_structure(results, img.shape)

            if table_data:
                df = pd.DataFrame(table_data)
                df_cleaned = self._clean_dataframe(df)

                result = {
                    'method': 'easyocr',
                    'table_id': 1,
                    'bbox': {'x1': 0, 'y1': 0, 'x2': img.shape[1], 'y2': img.shape[0]},
                    'shape': {'rows': df_cleaned.shape[0], 'cols': df_cleaned.shape[1]},
                    'dataframe': df_cleaned,
                    'confidence': 'medium'
                }
                return [result]

        except Exception as e:
            print(f"  ❌ Erreur EasyOCR: {e}")

        return None

    def extract_with_pytesseract(self, image_path):
        """Extraction avec Pytesseract direct (dernier fallback)"""
        if 'pytesseract' not in self.ocr_engines:
            return None

        print(f"📝 Extraction Pytesseract: {image_path}")

        try:
            # Charger l'image
            img = PILImage.open(image_path)

            # Configuration Tesseract pour tableaux
            config = '--psm 6 -l fra+eng'

            # Extraction données + structure
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

            # Convertir en tableau structuré
            table_data = self._tesseract_data_to_table(data)

            if table_data:
                df = pd.DataFrame(table_data)
                df_cleaned = self._clean_dataframe(df)

                result = {
                    'method': 'pytesseract',
                    'table_id': 1,
                    'bbox': {'x1': 0, 'y1': 0, 'x2': img.width, 'y2': img.height},
                    'shape': {'rows': df_cleaned.shape[0], 'cols': df_cleaned.shape[1]},
                    'dataframe': df_cleaned,
                    'confidence': 'low'
                }
                return [result]

        except Exception as e:
            print(f"  ❌ Erreur Pytesseract: {e}")

        return None

    def _ocr_to_table_structure(self, ocr_results, img_shape):
        """Convertir résultats OCR en structure tabulaire"""
        if not ocr_results:
            return None

        # Extraire texte et positions
        texts = []
        for (bbox, text, conf) in ocr_results:
            if conf > 0.5 and text.strip():  # Filtrer par confiance
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                x_center = (bbox[0][0] + bbox[2][0]) / 2
                texts.append({
                    'text': text.strip(),
                    'x': x_center,
                    'y': y_center,
                    'confidence': conf
                })

        if not texts:
            return None

        # Trier par position (y puis x)
        texts.sort(key=lambda t: (t['y'], t['x']))

        # Grouper en lignes approximatives
        rows = []
        current_row = []
        last_y = -1
        y_threshold = img_shape[0] * 0.02  # 2% de la hauteur

        for item in texts:
            if last_y == -1 or abs(item['y'] - last_y) < y_threshold:
                current_row.append(item)
                last_y = item['y']
            else:
                if current_row:
                    # Trier la ligne par x
                    current_row.sort(key=lambda t: t['x'])
                    rows.append([t['text'] for t in current_row])
                current_row = [item]
                last_y = item['y']

        # Ajouter la dernière ligne
        if current_row:
            current_row.sort(key=lambda t: t['x'])
            rows.append([t['text'] for t in current_row])

        # Égaliser le nombre de colonnes
        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows

    def _tesseract_data_to_table(self, data):
        """Convertir données Tesseract en tableau"""
        words = []

        for i, text in enumerate(data['text']):
            if text.strip() and int(data['conf'][i]) > 30:
                words.append({
                    'text': text.strip(),
                    'x': data['left'][i] + data['width'][i] // 2,
                    'y': data['top'][i] + data['height'][i] // 2,
                    'confidence': int(data['conf'][i])
                })

        if not words:
            return None

        # Grouper en lignes
        words.sort(key=lambda w: (w['y'], w['x']))

        rows = []
        current_row = []
        last_y = -1
        y_threshold = 15

        for word in words:
            if last_y == -1 or abs(word['y'] - last_y) < y_threshold:
                current_row.append(word)
                last_y = word['y']
            else:
                if current_row:
                    current_row.sort(key=lambda w: w['x'])
                    rows.append([w['text'] for w in current_row])
                current_row = [word]
                last_y = word['y']

        if current_row:
            current_row.sort(key=lambda w: w['x'])
            rows.append([w['text'] for w in current_row])

        # Égaliser colonnes
        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows

    def _clean_dataframe(self, df):
        """Nettoyer et optimiser le DataFrame"""
        if df.empty:
            return df

        # Remplacer None/nan par chaîne vide
        df = df.fillna('')

        # Nettoyer le texte
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                # Supprimer lignes vides
                df = df[df[col] != '']

        # Supprimer lignes entièrement vides
        df = df.dropna(how='all').reset_index(drop=True)

        return df

    def extract_table_intelligent(self, file_path, file_type="image"):
        """Extraction intelligente avec fallbacks multiples"""
        print(f"🧠 Extraction intelligente: {file_path}")

        results = None

        # 1. Essayer img2table (le plus précis pour la structure)
        if results is None:
            results = self.extract_with_img2table(file_path, file_type)

        # 2. Fallback EasyOCR (bon pour le texte)
        if (results is None or self._results_are_empty(results)) and file_type == "image":
            print("🔄 Fallback vers EasyOCR...")
            results = self.extract_with_easyocr(file_path)

        # 3. Fallback Pytesseract (dernier recours)
        if (results is None or self._results_are_empty(results)) and file_type == "image":
            print("🔄 Fallback vers Pytesseract...")
            results = self.extract_with_pytesseract(file_path)

        return results

    def _results_are_empty(self, results):
        """Vérifier si les résultats sont vides"""
        if not results:
            return True
        for result in results:
            df = result['dataframe']
            if not df.empty and not df.astype(str).eq('').all().all():
                return False
        return True

    def format_educational_data(self, results):
        """Formater spécialement pour données éducatives (notes élèves)"""
        if not results:
            return None

        formatted_results = []

        for result in results:
            df = result['dataframe']

            # Essayer de détecter structure notes d'élèves
            if df.shape[1] >= 3:  # Au moins 3 colonnes
                educational_format = {
                    'source_method': result['method'],
                    'confidence': result['confidence'],
                    'detected_structure': 'student_grades',
                    'students': []
                }

                # Supposer: Col 0 = N°, Col 1 = Nom, Col 2+ = Notes
                for idx, row in df.iterrows():
                    if idx == 0:  # Header probable
                        educational_format['headers'] = row.tolist()
                        continue

                    student_data = {
                        'numero': row.iloc[0] if len(row) > 0 else '',
                        'nom': row.iloc[1] if len(row) > 1 else '',
                        'donnees': row.iloc[2:].tolist() if len(row) > 2 else []
                    }
                    educational_format['students'].append(student_data)

                formatted_results.append(educational_format)

        return formatted_results

    def export_results(self, results, filename_prefix="extraction_complete"):
        """Exporter les résultats optimisés"""
        if not results:
            print("❌ Aucun résultat à exporter")
            return

        # Format standard
        export_data = {
            'extraction_summary': {
                'total_tables': len(results),
                'methods_used': [r['method'] for r in results],
                'best_confidence': max([r['confidence'] for r in results], default='none')
            },
            'tables': []
        }

        combined_df = pd.DataFrame()

        for result in results:
            table_info = {
                'method': result['method'],
                'table_id': result['table_id'],
                'confidence': result['confidence'],
                'bbox': result['bbox'],
                'shape': result['shape'],
                'data': result['dataframe'].to_dict('records')
            }
            export_data['tables'].append(table_info)

            if not combined_df.empty:
                combined_df = pd.concat([combined_df, result['dataframe']], ignore_index=True)
            else:
                combined_df = result['dataframe'].copy()

        # Format éducatif spécialisé
        educational_data = self.format_educational_data(results)

        # Exports
        json_path = self.output_dir / f"{filename_prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON standard: {json_path}")

        if educational_data:
            edu_json_path = self.output_dir / f"{filename_prefix}_educational.json"
            with open(edu_json_path, 'w', encoding='utf-8') as f:
                json.dump(educational_data, f, ensure_ascii=False, indent=2)
            print(f"📚 JSON éducatif: {edu_json_path}")

        csv_path = self.output_dir / f"{filename_prefix}.csv"
        combined_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 CSV: {csv_path}")

        return json_path, csv_path

def main():
    """Fonction principale optimisée"""
    print("🚀 EXTRACTION OPTIMISÉE - img2table + OCR Multiple")
    print("=" * 60)

    extractor = AdvancedTableExtractor()
    sample_dir = Path("sample_data")

    # Chercher fichiers
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))
    pdf_files = list(sample_dir.glob("*.pdf"))

    if not image_files and not pdf_files:
        print("❌ Aucun fichier trouvé!")
        return

    all_results = []

    # Traiter images
    for img_path in image_files:
        print(f"\n🖼️ Traitement: {img_path.name}")
        results = extractor.extract_table_intelligent(str(img_path), "image")
        if results:
            all_results.extend(results)

    # Traiter PDFs
    for pdf_path in pdf_files:
        print(f"\n📄 Traitement: {pdf_path.name}")
        results = extractor.extract_table_intelligent(str(pdf_path), "pdf")
        if results:
            all_results.extend(results)

    # Export final
    if all_results:
        print(f"\n📊 Export de {len(all_results)} tableau(x) extrait(s)")
        extractor.export_results(all_results)

        print("\n✅ EXTRACTION COMPLETE!")
        print("📁 Vérifiez le dossier 'output/' pour tous les résultats")

        # Résumé des méthodes utilisées
        methods = [r['method'] for r in all_results]
        print(f"🔧 Méthodes utilisées: {', '.join(set(methods))}")

    else:
        print("\n❌ Aucune extraction réussie")
        print("💡 Vérifiez que les moteurs OCR sont installés")

if __name__ == "__main__":
    main()