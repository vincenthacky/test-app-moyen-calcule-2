#!/usr/bin/env python3
"""
EXTRACTION RÉELLE UNIVERSELLE - Sans simulation
Extrait VRAIMENT le texte de n'importe quel tableau avec OCR
S'adapte automatiquement à tous types de tableaux notes/élèves
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image as PILImage

# OCR imports
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

try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
except Exception:
    IMG2TABLE_AVAILABLE = False

class RealTableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        self.ocr_engines = {}
        self._setup_ocr()

    def _setup_ocr(self):
        """Configuration des moteurs OCR"""
        print("🔧 Configuration moteurs OCR...")

        # EasyOCR - Meilleur pour tableaux
        if EASYOCR_AVAILABLE:
            try:
                self.ocr_engines['easyocr'] = easyocr.Reader(['fr', 'en'], gpu=False, verbose=False)
                print("✅ EasyOCR configuré")
            except Exception as e:
                print(f"❌ EasyOCR: {e}")

        # Pytesseract
        if PYTESSERACT_AVAILABLE:
            try:
                pytesseract.get_tesseract_version()
                self.ocr_engines['pytesseract'] = True
                print("✅ Pytesseract configuré")
            except Exception as e:
                print(f"❌ Pytesseract: {e}")

        # img2table + Tesseract
        if IMG2TABLE_AVAILABLE and 'pytesseract' in self.ocr_engines:
            try:
                self.ocr_engines['img2table'] = TesseractOCR(n_threads=1, lang="fra+eng")
                print("✅ img2table+Tesseract configuré")
            except Exception as e:
                print(f"❌ img2table: {e}")

        if not self.ocr_engines:
            print("❌ AUCUN moteur OCR disponible!")
            print("Installez: brew install tesseract && pip install easyocr pytesseract")

    def extract_with_easyocr(self, image_path):
        """Extraction réelle avec EasyOCR"""
        if 'easyocr' not in self.ocr_engines:
            return None

        print(f"🔍 EasyOCR - Extraction réelle: {Path(image_path).name}")

        try:
            reader = self.ocr_engines['easyocr']

            # Lecture image
            results = reader.readtext(str(image_path), paragraph=False)

            if not results:
                print("  ❌ Aucun texte détecté")
                return None

            print(f"  📄 {len(results)} éléments texte détectés")

            # Convertir en structure tableau
            table_data = self._ocr_results_to_table(results, method='easyocr')

            if table_data:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                print(f"  📊 Tableau: {df.shape[0]} lignes x {df.shape[1]} colonnes")

                return {
                    'method': 'easyocr',
                    'dataframe': df,
                    'raw_ocr_count': len(results),
                    'confidence': 'high'
                }

        except Exception as e:
            print(f"  ❌ Erreur EasyOCR: {e}")

        return None

    def extract_with_pytesseract(self, image_path):
        """Extraction réelle avec Pytesseract"""
        if 'pytesseract' not in self.ocr_engines:
            return None

        print(f"📝 Pytesseract - Extraction réelle: {Path(image_path).name}")

        try:
            img = PILImage.open(image_path)

            # Configuration pour tableaux
            config = r'--oem 3 --psm 6 -l fra+eng'

            # Extraction données structurées
            data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)

            # Construire tableau réel
            table_data = self._tesseract_data_to_table(data)

            if table_data:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                print(f"  📊 Tableau: {df.shape[0]} lignes x {df.shape[1]} colonnes")

                return {
                    'method': 'pytesseract',
                    'dataframe': df,
                    'confidence': 'medium'
                }

        except Exception as e:
            print(f"  ❌ Erreur Pytesseract: {e}")

        return None

    def extract_with_img2table(self, image_path):
        """Extraction réelle avec img2table + OCR"""
        if 'img2table' not in self.ocr_engines:
            return None

        print(f"📊 img2table - Extraction réelle: {Path(image_path).name}")

        try:
            doc = Img2TableImage(src=str(image_path), detect_rotation=True)

            # Extraction avec OCR
            tables = doc.extract_tables(
                ocr=self.ocr_engines['img2table'],
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=30
            )

            if not tables:
                print("  ❌ Aucun tableau détecté")
                return None

            # Prendre le premier tableau (généralement le plus grand)
            table = tables[0]
            df = table.df

            # Nettoyer les données vides
            df = self._clean_extracted_dataframe(df)

            if df.empty:
                print("  ❌ Tableau vide après nettoyage")
                return None

            print(f"  📊 Tableau: {df.shape[0]} lignes x {df.shape[1]} colonnes")

            return {
                'method': 'img2table',
                'dataframe': df,
                'bbox': {
                    'x1': table.bbox.x1, 'y1': table.bbox.y1,
                    'x2': table.bbox.x2, 'y2': table.bbox.y2
                },
                'confidence': 'high'
            }

        except Exception as e:
            print(f"  ❌ Erreur img2table: {e}")

        return None

    def _ocr_results_to_table(self, ocr_results, method='easyocr'):
        """Convertir résultats OCR en tableau structuré"""
        if not ocr_results:
            return None

        # Extraire texte et positions
        text_items = []
        for item in ocr_results:
            if method == 'easyocr':
                bbox, text, conf = item
                if conf > 0.3 and text.strip():  # Seuil de confiance
                    y_center = (bbox[0][1] + bbox[2][1]) / 2
                    x_center = (bbox[0][0] + bbox[2][0]) / 2
                    text_items.append({
                        'text': text.strip(),
                        'x': x_center,
                        'y': y_center
                    })

        if len(text_items) < 4:  # Minimum pour un tableau
            return None

        # Trier par position (Y puis X)
        text_items.sort(key=lambda item: (item['y'], item['x']))

        # Grouper en lignes
        rows = []
        current_row = []
        last_y = -1
        y_threshold = 20  # Tolérance pour même ligne

        for item in text_items:
            if last_y == -1 or abs(item['y'] - last_y) <= y_threshold:
                current_row.append(item)
                last_y = item['y']
            else:
                if current_row:
                    # Trier la ligne par position X
                    current_row.sort(key=lambda x: x['x'])
                    rows.append([x['text'] for x in current_row])
                current_row = [item]
                last_y = item['y']

        # Ajouter dernière ligne
        if current_row:
            current_row.sort(key=lambda x: x['x'])
            rows.append([x['text'] for x in current_row])

        # Normaliser nombre de colonnes
        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows if len(rows) >= 2 else None  # Au moins header + 1 ligne

    def _tesseract_data_to_table(self, data):
        """Convertir données Tesseract en tableau"""
        text_items = []

        for i, text in enumerate(data['text']):
            if text.strip() and int(data['conf'][i]) > 30:
                text_items.append({
                    'text': text.strip(),
                    'x': data['left'][i] + data['width'][i] // 2,
                    'y': data['top'][i] + data['height'][i] // 2
                })

        if len(text_items) < 4:
            return None

        # Même logique de groupement que EasyOCR
        return self._group_texts_into_table(text_items)

    def _group_texts_into_table(self, text_items):
        """Grouper textes en structure tableau"""
        text_items.sort(key=lambda item: (item['y'], item['x']))

        rows = []
        current_row = []
        last_y = -1
        y_threshold = 15

        for item in text_items:
            if last_y == -1 or abs(item['y'] - last_y) <= y_threshold:
                current_row.append(item)
                last_y = item['y']
            else:
                if current_row:
                    current_row.sort(key=lambda x: x['x'])
                    rows.append([x['text'] for x in current_row])
                current_row = [item]
                last_y = item['y']

        if current_row:
            current_row.sort(key=lambda x: x['x'])
            rows.append([x['text'] for x in current_row])

        if rows:
            max_cols = max(len(row) for row in rows)
            for row in rows:
                while len(row) < max_cols:
                    row.append('')

        return rows if len(rows) >= 2 else None

    def _clean_extracted_dataframe(self, df):
        """Nettoyer DataFrame extrait"""
        if df.empty:
            return df

        # Remplacer None/NaN par chaîne vide
        df = df.fillna('')

        # Convertir toutes les colonnes en string et nettoyer
        for col in df.columns:
            if df[col].dtype == 'object' or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()

        # Supprimer lignes complètement vides
        mask = df.astype(str).ne('').any(axis=1)
        df = df[mask].reset_index(drop=True)

        return df

    def extract_table_intelligent(self, image_path):
        """Extraction intelligente avec fallbacks"""
        print(f"\n🧠 Extraction intelligente: {Path(image_path).name}")

        results = []

        # 1. Essayer img2table (meilleur pour structure)
        if 'img2table' in self.ocr_engines:
            result = self.extract_with_img2table(image_path)
            if result and not result['dataframe'].empty:
                results.append(result)

        # 2. Essayer EasyOCR (bon pour texte complexe)
        if 'easyocr' in self.ocr_engines:
            result = self.extract_with_easyocr(image_path)
            if result and not result['dataframe'].empty:
                results.append(result)

        # 3. Essayer Pytesseract (fallback)
        if 'pytesseract' in self.ocr_engines:
            result = self.extract_with_pytesseract(image_path)
            if result and not result['dataframe'].empty:
                results.append(result)

        return results

    def detect_table_type(self, df):
        """Détecter type de tableau automatiquement"""
        if df.empty:
            return 'unknown'

        # Analyser les en-têtes pour détecter le type
        headers = df.columns.tolist() if hasattr(df, 'columns') else []
        first_row = df.iloc[0].tolist() if len(df) > 0 else []

        # Mots-clés pour tableaux de notes
        grade_keywords = ['note', 'nom', 'élève', 'student', 'grade', 'score', 'coefficient', 'moyenne']

        all_text = ' '.join(str(h).lower() for h in headers) + ' ' + ' '.join(str(c).lower() for c in first_row)

        if any(keyword in all_text for keyword in grade_keywords):
            return 'student_grades'

        # Autres types possibles
        if 'prix' in all_text or 'montant' in all_text:
            return 'financial'

        return 'generic_table'

    def format_for_educational_use(self, results):
        """Formater pour usage éducatif - GÉNÉRIQUE"""
        if not results:
            return []

        formatted_results = []

        for result in results:
            df = result['dataframe']
            table_type = self.detect_table_type(df)

            formatted_result = {
                'extraction_method': result['method'],
                'confidence': result['confidence'],
                'table_type': table_type,
                'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
                'headers': df.columns.tolist() if hasattr(df, 'columns') else [],
                'data': df.to_dict('records'),
                'raw_data': df.values.tolist()
            }

            # Si c'est un tableau de notes, ajouter analyse spécialisée
            if table_type == 'student_grades' and len(df.columns) >= 2:
                formatted_result['educational_analysis'] = self._analyze_grade_table(df)

            formatted_results.append(formatted_result)

        return formatted_results

    def _analyze_grade_table(self, df):
        """Analyser tableau de notes"""
        analysis = {}

        # Chercher colonnes avec des notes numériques
        grade_columns = []
        for col in df.columns:
            # Vérifier si la colonne contient des valeurs numériques entre 0-20
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            valid_grades = numeric_values[(numeric_values >= 0) & (numeric_values <= 20)]

            if len(valid_grades) > len(df) * 0.5:  # Plus de 50% sont des notes valides
                grade_columns.append({
                    'column': col,
                    'mean': round(valid_grades.mean(), 2) if not valid_grades.empty else 0,
                    'min': valid_grades.min() if not valid_grades.empty else 0,
                    'max': valid_grades.max() if not valid_grades.empty else 0,
                    'count': len(valid_grades)
                })

        analysis['grade_columns'] = grade_columns
        analysis['total_students'] = len(df)

        return analysis

    def export_results(self, all_results, filename_prefix="extraction"):
        """Export des résultats réels"""
        if not all_results:
            print("❌ Aucun résultat à exporter")
            return []

        # Format éducatif
        educational_data = []
        for results in all_results:
            if results:
                educational_data.extend(self.format_for_educational_use(results))

        # Export JSON éducatif
        files_created = []

        if educational_data:
            edu_json = self.output_dir / f"{filename_prefix}_educational.json"
            with open(edu_json, 'w', encoding='utf-8') as f:
                json.dump(educational_data, f, ensure_ascii=False, indent=2)
            files_created.append(edu_json)
            print(f"📚 Export éducatif: {edu_json}")

            # Export CSV combiné
            for i, edu_result in enumerate(educational_data):
                df = pd.DataFrame(edu_result['data'])
                if not df.empty:
                    csv_file = self.output_dir / f"{filename_prefix}_table_{i+1}.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    files_created.append(csv_file)

        return files_created

def main():
    """Fonction principale"""
    print("🚀 EXTRACTION RÉELLE UNIVERSELLE")
    print("=" * 50)

    extractor = RealTableExtractor()

    if not extractor.ocr_engines:
        print("❌ Aucun moteur OCR disponible!")
        return

    # Chercher fichiers
    sample_dir = Path("sample_data")
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

    if not image_files:
        print("❌ Aucun fichier trouvé!")
        return

    all_results = []

    # Traiter chaque image
    for img_path in image_files:
        results = extractor.extract_table_intelligent(str(img_path))
        if results:
            all_results.append(results)

            print(f"  ✅ {len(results)} extraction(s) réussie(s)")
            for result in results:
                df = result['dataframe']
                print(f"    📊 {result['method']}: {df.shape[0]}x{df.shape[1]}")
                print(f"    🔍 Aperçu: {df.head(2).to_string(index=False, max_cols=5)}")
        else:
            print(f"  ❌ Aucune extraction réussie")
            all_results.append(None)

    # Export final
    files = extractor.export_results(all_results)

    if files:
        print(f"\n✅ EXTRACTION TERMINÉE!")
        print(f"📁 {len(files)} fichier(s) créé(s) dans {extractor.output_dir}")
    else:
        print(f"\n❌ Aucune extraction exploitable")

if __name__ == "__main__":
    main()