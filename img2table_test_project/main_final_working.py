#!/usr/bin/env python3
"""
VERSION FINALE FONCTIONNELLE
Extraction OCR spécialement optimisée pour tableaux de notes scolaires
Preprocessing ciblé et paramètres OCR optimaux
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np
from PIL import Image as PILImage

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

class FinalTableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Test Tesseract
        if PYTESSERACT_AVAILABLE:
            try:
                version = pytesseract.get_tesseract_version()
                print(f"✅ Tesseract {version}")
                self.tesseract_ok = True
            except Exception as e:
                print(f"❌ Tesseract: {e}")
                self.tesseract_ok = False
        else:
            self.tesseract_ok = False

    def preprocess_image_for_table_ocr(self, image_path):
        """Preprocessing spécial pour tableaux de notes"""
        print("🔧 Preprocessing pour tableaux scolaires...")

        try:
            img = cv2.imread(str(image_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Redimensionner si trop petit
            height, width = gray.shape
            if min(height, width) < 800:
                scale = 800 / min(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"  📐 Redimensionné: {width}x{height} -> {new_width}x{new_height}")

            # Amélioration du contraste pour le texte
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)

            # Débruitage léger pour préserver les détails
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Binarisation optimisée pour tableaux
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 10
            )

            # Morphologie pour nettoyer
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Sauvegarder
            processed_path = self.output_dir / f"final_processed_{Path(image_path).name}"
            cv2.imwrite(str(processed_path), cleaned)

            return str(processed_path)

        except Exception as e:
            print(f"  ❌ Erreur: {e}")
            return str(image_path)

    def extract_with_direct_tesseract(self, image_path):
        """Extraction directe avec Tesseract optimisé"""
        if not self.tesseract_ok:
            return None

        print(f"📝 Tesseract direct optimisé: {Path(image_path).name}")

        try:
            # Préprocesser
            processed_img = self.preprocess_image_for_table_ocr(image_path)
            img = PILImage.open(processed_img)

            # Configuration spéciale pour tableaux français
            config = r'--oem 3 --psm 4 -l fra+eng'

            print("  🔍 Extraction avec configuration française...")

            # Extraction du texte brut pour vérifier
            raw_text = pytesseract.image_to_string(img, config=config)
            print(f"  📄 Texte détecté: {len(raw_text)} caractères")

            if len(raw_text.strip()) < 50:
                print("  ⚠️ Peu de texte détecté, essai configuration alternative...")
                config = r'--oem 3 --psm 6 -l fra+eng'
                raw_text = pytesseract.image_to_string(img, config=config)

            # Extraction structurée
            data = pytesseract.image_to_data(
                img, config=config,
                output_type=pytesseract.Output.DICT
            )

            print(f"  📊 {len(data['text'])} éléments détectés")

            # Construire le tableau en analysant la structure
            table_data = self._extract_table_structure_intelligent(data, raw_text)

            if table_data:
                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                print(f"  ✅ Tableau construit: {df.shape}")

                # Afficher aperçu nettoyé
                print("  🔍 Aperçu données:")
                for i, row in enumerate(df.head(3).values):
                    clean_row = [str(cell)[:20] + "..." if len(str(cell)) > 20 else str(cell) for cell in row]
                    print(f"    Ligne {i+1}: {clean_row}")

                return {
                    'method': 'tesseract_direct_optimized',
                    'dataframe': df,
                    'confidence': 'high',
                    'raw_text_length': len(raw_text),
                    'elements_detected': len(data['text'])
                }

        except Exception as e:
            print(f"  ❌ Erreur: {e}")

        return None

    def _extract_table_structure_intelligent(self, data, raw_text):
        """Extraction intelligente de la structure du tableau"""

        # Filtrer les mots avec confiance décente
        words = []
        for i, text in enumerate(data['text']):
            if text.strip() and int(data['conf'][i]) > 25:
                words.append({
                    'text': text.strip(),
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'conf': int(data['conf'][i])
                })

        if len(words) < 5:
            return self._fallback_text_parsing(raw_text)

        print(f"    📝 {len(words)} mots avec confiance > 25")

        # Trier par position
        words.sort(key=lambda w: (w['y'], w['x']))

        # Grouper en lignes avec détection intelligente
        rows = []
        current_row = []
        current_y = -1

        for word in words:
            # Nouvelle ligne si écart Y significatif
            if current_y == -1 or abs(word['y'] - current_y) > 20:
                if current_row:
                    # Traiter la ligne actuelle
                    processed_row = self._process_row(current_row)
                    if processed_row:
                        rows.append(processed_row)

                current_row = [word]
                current_y = word['y']
            else:
                current_row.append(word)
                # Moyenne pondérée des Y
                current_y = (current_y + word['y']) // 2

        # Traiter dernière ligne
        if current_row:
            processed_row = self._process_row(current_row)
            if processed_row:
                rows.append(processed_row)

        # Post-traitement pour détecter structure tableau
        if len(rows) >= 3:  # Au moins header + 2 lignes
            return self._format_as_grade_table(rows)

        return self._fallback_text_parsing(raw_text)

    def _process_row(self, row_words):
        """Traiter une ligne de mots pour créer des cellules"""
        if not row_words:
            return None

        # Trier par position X
        row_words.sort(key=lambda w: w['x'])

        # Regrouper mots proches en cellules
        cells = []
        current_cell = []
        last_x = -1

        for word in row_words:
            # Si le mot est proche du précédent, même cellule
            if last_x == -1 or word['x'] - last_x < 100:
                current_cell.append(word['text'])
                last_x = word['x'] + word['width']
            else:
                # Nouvelle cellule
                if current_cell:
                    cells.append(' '.join(current_cell))
                current_cell = [word['text']]
                last_x = word['x'] + word['width']

        # Ajouter dernière cellule
        if current_cell:
            cells.append(' '.join(current_cell))

        return cells if cells else None

    def _format_as_grade_table(self, rows):
        """Formater comme tableau de notes"""

        # Détecter probable header (ligne avec mots-clés)
        header_idx = 0
        header_keywords = ['nom', 'note', 'élève', 'coefficient', 'n°', 'numéro']

        for i, row in enumerate(rows[:3]):  # Chercher dans les 3 premières lignes
            row_text = ' '.join(row).lower()
            if any(keyword in row_text for keyword in header_keywords):
                header_idx = i
                break

        # Définir headers
        if header_idx < len(rows):
            headers = rows[header_idx]
        else:
            # Headers par défaut
            max_cols = max(len(row) for row in rows)
            headers = [f"Colonne_{i+1}" for i in range(max_cols)]

        # Égaliser colonnes
        target_cols = len(headers)
        normalized_rows = []

        for row in rows:
            normalized_row = row[:target_cols]  # Tronquer si trop long
            while len(normalized_row) < target_cols:  # Compléter si trop court
                normalized_row.append('')
            normalized_rows.append(normalized_row)

        # Retourner avec headers comme première ligne
        result = [headers] + normalized_rows[header_idx+1:]

        print(f"    📋 Tableau formaté: {len(result)} lignes x {len(headers)} colonnes")

        return result if len(result) > 1 else None

    def _fallback_text_parsing(self, raw_text):
        """Parsing de secours à partir du texte brut"""
        print("    🔄 Fallback: analyse du texte brut...")

        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        if len(lines) < 3:
            return None

        # Essayer de détecter structure tabulaire
        table_rows = []

        for line in lines[:20]:  # Limiter aux 20 premières lignes
            # Séparer par espaces multiples ou caractères tabulaires
            if '\t' in line:
                parts = [p.strip() for p in line.split('\t') if p.strip()]
            else:
                # Séparer par espaces multiples
                parts = [p.strip() for p in line.split() if p.strip()]

            if len(parts) >= 2:  # Au moins 2 colonnes
                table_rows.append(parts)

        if len(table_rows) >= 2:
            # Égaliser colonnes
            max_cols = max(len(row) for row in table_rows)
            for row in table_rows:
                while len(row) < max_cols:
                    row.append('')

            print(f"    📋 Fallback réussi: {len(table_rows)} lignes")
            return table_rows

        return None

    def extract_with_img2table_simple(self, image_path):
        """Extraction img2table simplifiée"""
        if not IMG2TABLE_AVAILABLE or not self.tesseract_ok:
            return None

        print(f"📊 img2table simple: {Path(image_path).name}")

        try:
            # Image préprocessée
            processed_img = self.preprocess_image_for_table_ocr(image_path)

            # OCR simple
            ocr = TesseractOCR(n_threads=1, lang="fra+eng")

            # Document
            doc = Img2TableImage(src=processed_img, detect_rotation=False)

            # Extraction basique
            tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=30
            )

            if tables:
                table = tables[0]
                df = table.df.fillna('')

                # Nettoyer
                for col in df.columns:
                    df[col] = df[col].astype(str).str.strip()

                # Supprimer lignes vides
                mask = (df != '').any(axis=1)
                df_cleaned = df[mask].reset_index(drop=True)

                if not df_cleaned.empty:
                    print(f"  ✅ {df_cleaned.shape}")
                    return {
                        'method': 'img2table_simple',
                        'dataframe': df_cleaned,
                        'confidence': 'medium'
                    }

        except Exception as e:
            print(f"  ❌ Erreur: {e}")

        return None

    def process_final(self, image_path):
        """Traitement final avec méthodes optimisées"""
        print(f"\n🎯 TRAITEMENT FINAL: {Path(image_path).name}")

        results = []

        # 1. Tesseract direct (plus précis)
        result1 = self.extract_with_direct_tesseract(image_path)
        if result1:
            results.append(result1)

        # 2. img2table simple (structure)
        result2 = self.extract_with_img2table_simple(image_path)
        if result2:
            results.append(result2)

        # Choisir le meilleur
        if results:
            # Préférer le résultat avec le plus de contenu significatif
            best = max(results, key=lambda r: r['dataframe'].shape[0] * r['dataframe'].shape[1])
            print(f"\n🏆 Meilleur: {best['method']}")

            return best

        return None

    def export_final_results(self, result, filename="final_extraction"):
        """Export final optimisé"""
        if not result:
            return []

        files = []
        df = result['dataframe']

        # JSON éducatif
        educational_data = {
            'extraction_info': {
                'method': result['method'],
                'confidence': result['confidence'],
                'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
                'timestamp': pd.Timestamp.now().isoformat()
            },
            'table_data': {
                'headers': df.columns.tolist(),
                'rows': df.to_dict('records')
            },
            'educational_analysis': self._analyze_educational_content(df)
        }

        # Export JSON
        json_file = self.output_dir / f"{filename}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(educational_data, f, ensure_ascii=False, indent=2)
        files.append(json_file)

        # Export CSV
        csv_file = self.output_dir / f"{filename}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        files.append(csv_file)

        # Rapport lisible
        report_file = self.output_dir / f"{filename}_report.txt"
        self._write_final_report(educational_data, report_file)
        files.append(report_file)

        print(f"\n📁 {len(files)} fichiers créés:")
        for f in files:
            print(f"  💾 {f.name}")

        return files

    def _analyze_educational_content(self, df):
        """Analyser le contenu éducatif"""
        analysis = {}

        # Chercher colonnes de notes
        numeric_cols = []
        text_cols = []

        for col in df.columns:
            # Test pour notes (nombres entre 0-20)
            numeric_count = 0
            text_count = 0

            for val in df[col]:
                val_str = str(val).strip()
                try:
                    num_val = float(val_str.replace(',', '.'))
                    if 0 <= num_val <= 20:
                        numeric_count += 1
                except ValueError:
                    if len(val_str) > 1 and val_str.isalpha():
                        text_count += 1

            if numeric_count > len(df) * 0.3:
                numeric_cols.append({'column': col, 'count': numeric_count})
            elif text_count > len(df) * 0.3:
                text_cols.append({'column': col, 'count': text_count})

        analysis['numeric_columns'] = numeric_cols
        analysis['text_columns'] = text_cols
        analysis['likely_grade_table'] = len(numeric_cols) > 0 and len(text_cols) > 0

        return analysis

    def _write_final_report(self, data, output_path):
        """Écrire rapport final"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT FINAL D'EXTRACTION ===\n\n")

            info = data['extraction_info']
            f.write(f"Méthode: {info['method']}\n")
            f.write(f"Confiance: {info['confidence']}\n")
            f.write(f"Dimensions: {info['shape']['rows']} x {info['shape']['cols']}\n")
            f.write(f"Timestamp: {info['timestamp']}\n\n")

            analysis = data['educational_analysis']
            if analysis['likely_grade_table']:
                f.write("✅ TABLEAU DE NOTES DÉTECTÉ\n\n")

            f.write("COLONNES DÉTECTÉES:\n")
            for col in analysis['text_columns']:
                f.write(f"  📝 Texte: {col['column']} ({col['count']} entrées)\n")
            for col in analysis['numeric_columns']:
                f.write(f"  🔢 Notes: {col['column']} ({col['count']} entrées)\n")

            f.write("\nDONNÉES EXTRAITES (10 premières lignes):\n")
            for i, row in enumerate(data['table_data']['rows'][:10], 1):
                f.write(f"  {i}: {list(row.values())}\n")

def main():
    """Fonction principale finale"""
    print("🚀 EXTRACTION FINALE OPTIMISÉE")
    print("=" * 50)

    extractor = FinalTableExtractor()

    if not extractor.tesseract_ok:
        print("❌ Tesseract requis!")
        return

    # Traiter les images
    sample_dir = Path("sample_data")
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

    for img_path in image_files:
        result = extractor.process_final(str(img_path))

        if result:
            files = extractor.export_final_results(result)

            print(f"\n✅ SUCCÈS!")
            print(f"📊 Résultat: {result['dataframe'].shape[0]} lignes x {result['dataframe'].shape[1]} colonnes")
            df = result['dataframe']
            print(f"🔍 Aperçu final:")
            print(df.head().to_string(index=False, max_cols=5))
        else:
            print("\n❌ Échec extraction")

if __name__ == "__main__":
    main()