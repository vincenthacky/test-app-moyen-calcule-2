#!/usr/bin/env python3
"""
Version simplifiée avec Tesseract OCR fonctionnel
Extraction RÉELLE du texte des tableaux
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image as PILImage

# Imports pour OCR
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
    print("✅ Pytesseract disponible")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("❌ Pytesseract non disponible")

try:
    from img2table.document import Image, PDF
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
    print("✅ img2table disponible")
except Exception as e:
    IMG2TABLE_AVAILABLE = False
    print(f"❌ img2table: {e}")

class SimpleTableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Test Tesseract
        if PYTESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                print(f"✅ Tesseract version: {version}")
                self.tesseract_ok = True
            except Exception as e:
                print(f"❌ Tesseract non accessible: {e}")
                self.tesseract_ok = False
        else:
            self.tesseract_ok = False

    def extract_with_img2table_tesseract(self, image_path):
        """Extraction avec img2table + Tesseract"""
        if not IMG2TABLE_AVAILABLE or not self.tesseract_ok:
            return None

        print(f"📊 img2table + Tesseract: {image_path}")

        try:
            # Configurer OCR
            ocr = TesseractOCR(n_threads=1, lang="fra+eng")

            # Document
            doc = Image(src=image_path, detect_rotation=True)

            # Extraction
            tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True
            )

            if not tables:
                print("  ❌ Aucun tableau détecté")
                return None

            results = []
            for i, table in enumerate(tables):
                df = table.df
                print(f"  📋 Tableau {i+1}: {df.shape}")
                print(f"  📝 Aperçu:")
                print(df.head().to_string(index=False, max_cols=6))

                results.append({
                    'method': 'img2table+tesseract',
                    'table_id': i+1,
                    'dataframe': df,
                    'bbox': {
                        'x1': table.bbox.x1, 'y1': table.bbox.y1,
                        'x2': table.bbox.x2, 'y2': table.bbox.y2
                    }
                })

            return results

        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            return None

    def extract_with_pytesseract_direct(self, image_path):
        """Extraction directe avec Pytesseract"""
        if not self.tesseract_ok:
            return None

        print(f"📝 Pytesseract direct: {image_path}")

        try:
            # Charger image
            img = PILImage.open(image_path)
            img_cv = cv2.imread(str(image_path))

            # Configuration optimisée pour tableaux
            config = r'--oem 3 --psm 6 -l fra+eng'

            # Extraction du texte brut
            text = pytesseract.image_to_string(img, config=config)
            print(f"  📄 Texte brut extrait ({len(text)} caractères)")

            if len(text.strip()) < 10:
                print("  ❌ Pas assez de texte extrait")
                return None

            # Extraction avec structure (données + positions)
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

            # Analyser structure tabulaire
            table_data = self._build_table_from_tesseract_data(data)

            if not table_data:
                # Fallback: structure basique du texte brut
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                if len(lines) >= 2:
                    # Essayer de détecter structure colonnes
                    table_data = self._parse_text_lines_to_table(lines)

            if table_data:
                df = pd.DataFrame(table_data)
                print(f"  📊 DataFrame: {df.shape}")
                print(f"  🔍 Contenu:")
                print(df.head().to_string(index=False))

                return [{
                    'method': 'pytesseract_direct',
                    'table_id': 1,
                    'dataframe': df,
                    'text_raw': text[:500] + "..." if len(text) > 500 else text
                }]

        except Exception as e:
            print(f"  ❌ Erreur Pytesseract: {e}")

        return None

    def _build_table_from_tesseract_data(self, data):
        """Construire tableau à partir des données Tesseract"""
        # Filtrer mots avec bonne confiance
        words = []
        for i, text in enumerate(data['text']):
            conf = int(data['conf'][i])
            if text.strip() and conf > 30:
                words.append({
                    'text': text.strip(),
                    'x': data['left'][i] + data['width'][i] // 2,
                    'y': data['top'][i] + data['height'][i] // 2,
                    'conf': conf
                })

        if len(words) < 3:
            return None

        # Trier par position Y puis X
        words.sort(key=lambda w: (w['y'], w['x']))

        # Grouper en lignes (tolérance Y)
        rows = []
        current_row = []
        last_y = -1
        y_tolerance = 20

        for word in words:
            if last_y == -1 or abs(word['y'] - last_y) <= y_tolerance:
                current_row.append(word)
                last_y = (last_y + word['y']) // 2 if last_y != -1 else word['y']
            else:
                if current_row:
                    # Trier ligne par X
                    current_row.sort(key=lambda w: w['x'])
                    rows.append([w['text'] for w in current_row])
                current_row = [word]
                last_y = word['y']

        # Dernière ligne
        if current_row:
            current_row.sort(key=lambda w: w['x'])
            rows.append([w['text'] for w in current_row])

        # Égaliser colonnes
        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows if len(rows) >= 2 else None

    def _parse_text_lines_to_table(self, lines):
        """Parser basique lignes de texte en tableau"""
        table_rows = []

        for line in lines:
            # Essayer différents séparateurs
            if '\t' in line:
                cols = line.split('\t')
            elif '  ' in line:  # Double espace
                cols = [col.strip() for col in line.split('  ') if col.strip()]
            elif len(line.split()) >= 3:  # Au moins 3 mots
                cols = line.split()
            else:
                continue

            if len(cols) >= 2:
                table_rows.append(cols)

        # Égaliser colonnes
        if table_rows:
            max_cols = max(len(row) for row in table_rows)
            for row in table_rows:
                while len(row) < max_cols:
                    row.append('')

        return table_rows if len(table_rows) >= 2 else None

    def process_file(self, file_path):
        """Traiter un fichier avec fallbacks"""
        print(f"\n🔍 Traitement: {file_path}")

        results = None

        # 1. Essayer img2table + Tesseract
        if not results:
            results = self.extract_with_img2table_tesseract(file_path)

        # 2. Fallback Pytesseract direct
        if not results:
            print("🔄 Fallback Pytesseract direct...")
            results = self.extract_with_pytesseract_direct(file_path)

        return results

    def export_results(self, all_results):
        """Exporter tous les résultats"""
        if not all_results:
            print("❌ Aucun résultat à exporter")
            return

        # Données standard
        export_data = {
            'extraction_info': {
                'total_files_processed': len(all_results),
                'successful_extractions': len([r for r in all_results if r]),
                'methods_used': list(set(r['method'] for results in all_results if results for r in results))
            },
            'extractions': []
        }

        # Données éducatives spécifiques
        educational_data = []

        combined_df = pd.DataFrame()

        for file_results in all_results:
            if not file_results:
                continue

            for result in file_results:
                df = result['dataframe']

                # Ajouter aux données standard
                export_data['extractions'].append({
                    'method': result['method'],
                    'table_id': result['table_id'],
                    'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
                    'data': df.to_dict('records')
                })

                # Format éducatif si possible
                if df.shape[1] >= 4 and df.shape[0] >= 2:  # Structure type notes
                    edu_entry = {
                        'method': result['method'],
                        'detected_format': 'student_grades',
                        'students': []
                    }

                    for idx, row in df.iterrows():
                        if idx == 0:  # Headers
                            edu_entry['headers'] = row.tolist()
                        else:
                            edu_entry['students'].append({
                                'numero': row.iloc[0] if len(row) > 0 else '',
                                'nom': row.iloc[1] if len(row) > 1 else '',
                                'note': row.iloc[2] if len(row) > 2 else '',
                                'coefficient': row.iloc[3] if len(row) > 3 else '',
                                'note_ponderee': row.iloc[4] if len(row) > 4 else ''
                            })

                    educational_data.append(edu_entry)

                # Combiner DataFrames
                if not combined_df.empty:
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                else:
                    combined_df = df.copy()

        # Exports
        json_path = self.output_dir / "extraction_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON: {json_path}")

        if educational_data:
            edu_json_path = self.output_dir / "extraction_educational.json"
            with open(edu_json_path, 'w', encoding='utf-8') as f:
                json.dump(educational_data, f, ensure_ascii=False, indent=2)
            print(f"📚 JSON éducatif: {edu_json_path}")

        if not combined_df.empty:
            csv_path = self.output_dir / "extraction_complete.csv"
            combined_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"💾 CSV: {csv_path}")

def main():
    """Fonction principale"""
    print("🚀 EXTRACTION SIMPLIFIÉE AVEC OCR")
    print("=" * 50)

    extractor = SimpleTableExtractor()

    if not extractor.tesseract_ok:
        print("❌ Tesseract non disponible!")
        print("Installation: brew install tesseract tesseract-lang")
        print("Et vérifiez le PATH")
        return

    # Chercher fichiers
    sample_dir = Path("sample_data")
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

    if not image_files:
        print("❌ Aucun fichier image trouvé!")
        return

    all_results = []

    # Traiter chaque fichier
    for img_path in image_files:
        results = extractor.process_file(str(img_path))
        all_results.append(results)

    # Export final
    extractor.export_results(all_results)

    print(f"\n✅ Traitement terminé!")
    print("📁 Consultez le dossier 'output/' pour les résultats")

if __name__ == "__main__":
    main()