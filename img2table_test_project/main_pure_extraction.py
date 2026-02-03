#!/usr/bin/env python3
"""
EXTRACTION PURE SANS SIMULATION
Utilise uniquement img2table + OCR intégré pour extraire LE VRAI CONTENU
Aucune donnée hardcodée - extraction 100% automatique
"""

import os
import sys
from pathlib import Path
import pandas as pd
import json
import cv2
import numpy as np

# Import img2table avec gestion d'erreur
try:
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR
    IMG2TABLE_AVAILABLE = True
    print("✅ img2table disponible")
except Exception as e:
    IMG2TABLE_AVAILABLE = False
    print(f"❌ img2table non disponible: {e}")

class PureTableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Tester si Tesseract est accessible
        self.tesseract_available = self._test_tesseract()

    def _test_tesseract(self):
        """Tester si Tesseract est disponible"""
        try:
            import subprocess
            result = subprocess.run(['tesseract', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ Tesseract accessible via système")
                return True
        except Exception:
            pass

        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract via pytesseract: {version}")
            return True
        except Exception as e:
            print(f"❌ Tesseract non accessible: {e}")
            return False

    def extract_with_img2table_ocr(self, image_path):
        """Extraction avec img2table + OCR réel"""
        if not IMG2TABLE_AVAILABLE or not self.tesseract_available:
            print("  ⚠️ OCR non disponible - extraction structure seule")
            return self.extract_structure_only(image_path)

        print(f"📊 img2table + OCR: {Path(image_path).name}")

        try:
            # Configuration OCR
            ocr = TesseractOCR(
                n_threads=1,
                lang="fra+eng",  # Français et anglais
                psm=6  # Assume uniform block of text
            )

            # Document
            doc = Img2TableImage(src=str(image_path), detect_rotation=True)

            # Extraction avec OCR
            tables = doc.extract_tables(
                ocr=ocr,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=30  # Seuil de confiance OCR
            )

            if not tables:
                print("  ❌ Aucun tableau détecté")
                return None

            table = tables[0]  # Premier tableau
            df = table.df

            print(f"  📋 Tableau extrait: {df.shape[0]} x {df.shape[1]}")

            # Nettoyer les données extraites
            df_cleaned = self._clean_ocr_dataframe(df)

            if df_cleaned.empty:
                print("  ⚠️ Tableau vide après nettoyage")
                return None

            print("  🔍 Données extraites:")
            print(df_cleaned.head().to_string(index=False, max_cols=6))

            return {
                'method': 'img2table_with_ocr',
                'dataframe': df_cleaned,
                'original_shape': df.shape,
                'cleaned_shape': df_cleaned.shape,
                'bbox': {
                    'x1': table.bbox.x1, 'y1': table.bbox.y1,
                    'x2': table.bbox.x2, 'y2': table.bbox.y2
                },
                'confidence': 'high'
            }

        except Exception as e:
            print(f"  ❌ Erreur extraction OCR: {e}")
            print("  🔄 Tentative extraction structure seule...")
            return self.extract_structure_only(image_path)

    def extract_structure_only(self, image_path):
        """Extraction structure seule (sans OCR)"""
        if not IMG2TABLE_AVAILABLE:
            return None

        print(f"📐 Structure seule: {Path(image_path).name}")

        try:
            doc = Img2TableImage(src=str(image_path), detect_rotation=True)

            # Extraction structure uniquement
            tables = doc.extract_tables(
                implicit_rows=True,
                borderless_tables=True
            )

            if not tables:
                print("  ❌ Aucune structure détectée")
                return None

            table = tables[0]
            df_structure = table.df

            print(f"  📏 Structure détectée: {df_structure.shape[0]} x {df_structure.shape[1]}")

            # Les cellules sont vides (None) mais la structure est détectée
            # On garde les dimensions réelles détectées

            # Créer un DataFrame avec les dimensions correctes mais vide
            empty_df = pd.DataFrame(
                index=range(df_structure.shape[0]),
                columns=range(df_structure.shape[1])
            ).fillna('')

            return {
                'method': 'img2table_structure_only',
                'dataframe': empty_df,
                'shape': df_structure.shape,
                'bbox': {
                    'x1': table.bbox.x1, 'y1': table.bbox.y1,
                    'x2': table.bbox.x2, 'y2': table.bbox.y2
                },
                'confidence': 'medium',
                'note': 'Structure détectée mais contenu non extrait (OCR requis)'
            }

        except Exception as e:
            print(f"  ❌ Erreur extraction structure: {e}")
            return None

    def _clean_ocr_dataframe(self, df):
        """Nettoyer DataFrame issu d'OCR"""
        if df.empty:
            return df

        # Remplacer None/NaN par chaîne vide
        df_cleaned = df.fillna('')

        # Convertir en string et nettoyer
        for col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

        # Supprimer lignes complètement vides
        mask = (df_cleaned != '').any(axis=1)
        df_cleaned = df_cleaned[mask].reset_index(drop=True)

        # Supprimer colonnes complètement vides
        mask_cols = (df_cleaned != '').any(axis=0)
        df_cleaned = df_cleaned.loc[:, mask_cols]

        return df_cleaned

    def analyze_extracted_content(self, result):
        """Analyser le contenu extrait"""
        if not result or result['dataframe'].empty:
            return None

        df = result['dataframe']

        analysis = {
            'extraction_method': result['method'],
            'confidence': result['confidence'],
            'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
            'content_analysis': {}
        }

        # Analyser le type de contenu
        if df.shape[1] >= 3:  # Au moins 3 colonnes
            # Vérifier s'il y a des patterns de notes/évaluations
            has_numeric_data = False
            has_names = False

            for col_idx, col in enumerate(df.columns):
                col_data = df[col].astype(str)

                # Chercher patterns numériques (notes potentielles)
                numeric_count = 0
                for value in col_data:
                    try:
                        num_val = float(value)
                        if 0 <= num_val <= 20:  # Range typique notes
                            numeric_count += 1
                    except ValueError:
                        pass

                if numeric_count > len(col_data) * 0.3:  # 30% sont des notes valides
                    has_numeric_data = True
                    analysis['content_analysis'][f'column_{col_idx}'] = {
                        'type': 'numeric_grades',
                        'valid_count': numeric_count,
                        'percentage': round(numeric_count / len(col_data) * 100, 1)
                    }

                # Chercher patterns de noms
                text_count = 0
                for value in col_data:
                    if isinstance(value, str) and len(value) > 2 and value.replace(' ', '').isalpha():
                        text_count += 1

                if text_count > len(col_data) * 0.5:  # 50% sont du texte
                    has_names = True
                    analysis['content_analysis'][f'column_{col_idx}'] = {
                        'type': 'text_names',
                        'valid_count': text_count,
                        'percentage': round(text_count / len(col_data) * 100, 1)
                    }

            # Déterminer le type de tableau
            if has_numeric_data and has_names:
                analysis['table_type'] = 'student_grades'
            elif has_numeric_data:
                analysis['table_type'] = 'numeric_data'
            elif has_names:
                analysis['table_type'] = 'text_data'
            else:
                analysis['table_type'] = 'mixed_or_unknown'

        return analysis

    def process_image(self, image_path):
        """Traiter une image complètement"""
        print(f"\n🎯 Traitement: {Path(image_path).name}")

        # Extraction principale
        extraction_result = self.extract_with_img2table_ocr(image_path)

        if not extraction_result:
            print("  ❌ Échec extraction")
            return None

        # Analyse du contenu
        content_analysis = self.analyze_extracted_content(extraction_result)

        # Résultat complet (éviter DataFrame dans JSON)
        df = extraction_result['dataframe']
        complete_result = {
            'file': Path(image_path).name,
            'extraction': {
                'method': extraction_result['method'],
                'confidence': extraction_result['confidence'],
                'shape': {'rows': df.shape[0], 'cols': df.shape[1]},
                'bbox': extraction_result.get('bbox', {}),
                'note': extraction_result.get('note', '')
            },
            'analysis': content_analysis,
            'raw_data': df.to_dict('records') if not df.empty else [],
            'data_preview': df.head().to_dict('records') if not df.empty else [],
            'summary': {
                'method_used': extraction_result['method'],
                'confidence': extraction_result['confidence'],
                'rows_extracted': df.shape[0],
                'cols_extracted': df.shape[1],
                'has_content': not df.empty and not (df == '').all().all()
            }
        }

        return complete_result

    def export_pure_results(self, results, filename="pure_extraction"):
        """Exporter résultats purs"""
        if not results:
            print("❌ Aucun résultat à exporter")
            return []

        files_created = []

        # Export JSON technique complet
        tech_json = self.output_dir / f"{filename}_technical.json"
        with open(tech_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        files_created.append(tech_json)
        print(f"📄 Export technique: {tech_json}")

        # Export données seules (pour chaque fichier traité)
        for i, result in enumerate(results):
            if result and result['raw_data']:
                # Convertir raw_data en DataFrame pour export CSV
                df = pd.DataFrame(result['raw_data'])
                if not df.empty:
                    csv_file = self.output_dir / f"{filename}_data_{i+1}.csv"
                    df.to_csv(csv_file, index=False, encoding='utf-8')
                    files_created.append(csv_file)
                    print(f"📊 Export données: {csv_file}")

        # Export rapport lisible
        report_file = self.output_dir / f"{filename}_report.txt"
        self._generate_report(results, report_file)
        files_created.append(report_file)

        return files_created

    def _generate_report(self, results, output_path):
        """Générer rapport lisible"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT EXTRACTION PURE ===\n\n")

            for i, result in enumerate(results, 1):
                if not result:
                    f.write(f"FICHIER {i}: Échec extraction\n\n")
                    continue

                f.write(f"FICHIER {i}: {result['file']}\n")
                f.write(f"Méthode: {result['summary']['method_used']}\n")
                f.write(f"Confiance: {result['summary']['confidence']}\n")
                f.write(f"Contenu extrait: {'Oui' if result['summary']['has_content'] else 'Non'}\n")
                f.write(f"Dimensions: {result['summary']['rows_extracted']} x {result['summary']['cols_extracted']}\n")

                if result['analysis']:
                    f.write(f"Type détecté: {result['analysis'].get('table_type', 'inconnu')}\n")

                # Données extraites
                if result['raw_data'] and len(result['raw_data']) > 0:
                    f.write("\nDONNÉES EXTRAITES:\n")
                    for j, row in enumerate(result['raw_data'][:10]):  # Limite à 10 lignes
                        f.write(f"  Ligne {j+1}: {list(row.values())}\n")
                    if len(result['raw_data']) > 10:
                        f.write(f"  ... et {len(result['raw_data']) - 10} lignes supplémentaires\n")

                f.write("\n" + "="*50 + "\n\n")

def main():
    """Fonction principale"""
    print("🚀 EXTRACTION PURE SANS SIMULATION")
    print("=" * 50)

    if not IMG2TABLE_AVAILABLE:
        print("❌ img2table requis!")
        print("Installation: pip install img2table")
        return

    extractor = PureTableExtractor()

    # Chercher fichiers
    sample_dir = Path("sample_data")
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))

    if not image_files:
        print("❌ Aucun fichier trouvé!")
        return

    all_results = []

    # Traiter chaque image
    for img_path in image_files:
        result = extractor.process_image(str(img_path))
        all_results.append(result)

    # Filtrer résultats valides
    valid_results = [r for r in all_results if r is not None]

    if valid_results:
        # Export
        files = extractor.export_pure_results(valid_results)

        print(f"\n✅ EXTRACTION PURE TERMINÉE!")
        print(f"📁 {len(files)} fichier(s) créé(s)")
        print(f"📊 {len(valid_results)} extraction(s) réussie(s)")

        # Résumé des méthodes utilisées
        methods = [r['summary']['method_used'] for r in valid_results]
        content_extracted = [r['summary']['has_content'] for r in valid_results]

        print(f"🔧 Méthodes: {', '.join(set(methods))}")
        print(f"📄 Contenu extrait: {sum(content_extracted)}/{len(content_extracted)} fichiers")

    else:
        print("\n❌ Aucune extraction réussie")
        print("💡 Vérifiez que Tesseract est installé pour l'OCR")

if __name__ == "__main__":
    main()