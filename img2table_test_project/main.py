#!/usr/bin/env python3
"""
Projet de test img2table - Extraction de tableaux à partir d'images/PDF
Convertit les tableaux extraits en DataFrame pandas et exporte en JSON/CSV
"""

import os
import sys
from pathlib import Path
import pandas as pd
from img2table.document import Image, PDF
from img2table.ocr import TesseractOCR
import json

class TableExtractor:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Initialiser OCR (optionnel - img2table peut fonctionner sans OCR pour certains cas)
        try:
            self.ocr = TesseractOCR(n_threads=1, lang="fra+eng")
        except Exception as e:
            print(f"⚠️ OCR Tesseract non disponible: {e}")
            print("Fonctionnement sans OCR (extraction structure uniquement)")
            self.ocr = None

    def extract_from_image(self, image_path):
        """Extraire tableau à partir d'une image"""
        print(f"📸 Extraction à partir de l'image: {image_path}")

        try:
            # Créer document img2table
            doc = Image(src=image_path, detect_rotation=True)

            # Extraire les tableaux
            if self.ocr:
                extracted_tables = doc.extract_tables(ocr=self.ocr, implicit_rows=True, borderless_tables=True)
            else:
                extracted_tables = doc.extract_tables(implicit_rows=True, borderless_tables=True)

            print(f"✅ {len(extracted_tables)} tableau(x) détecté(s)")
            return self._process_extracted_tables(extracted_tables, "image")

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de l'image: {e}")
            return None

    def extract_from_pdf(self, pdf_path):
        """Extraire tableau à partir d'un PDF"""
        print(f"📄 Extraction à partir du PDF: {pdf_path}")

        try:
            # Créer document img2table
            doc = PDF(src=pdf_path, pages=[0])  # Premier page uniquement

            # Extraire les tableaux
            if self.ocr:
                extracted_tables = doc.extract_tables(ocr=self.ocr, implicit_rows=True, borderless_tables=True)
            else:
                extracted_tables = doc.extract_tables(implicit_rows=True, borderless_tables=True)

            print(f"✅ {len(extracted_tables)} tableau(x) détecté(s)")
            return self._process_extracted_tables(extracted_tables, "pdf")

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction du PDF: {e}")
            return None

    def _process_extracted_tables(self, extracted_tables, source_type):
        """Traiter les tableaux extraits et les convertir en DataFrame"""
        if not extracted_tables:
            print("❌ Aucun tableau trouvé")
            return None

        results = []

        for i, table in enumerate(extracted_tables):
            print(f"\n🔍 Analyse du tableau {i+1}:")
            print(f"  - Position: ({table.bbox.x1}, {table.bbox.y1}) -> ({table.bbox.x2}, {table.bbox.y2})")
            print(f"  - Dimensions: {table.bbox.x2 - table.bbox.x1}x{table.bbox.y2 - table.bbox.y1}")

            # Convertir en DataFrame
            try:
                df = table.df
                print(f"  - DataFrame créé: {df.shape[0]} lignes x {df.shape[1]} colonnes")
                print(f"  - Aperçu des données:")
                print(df.head().to_string(index=False))

                # Sauvegarder ce tableau
                result = {
                    'table_id': i+1,
                    'source_type': source_type,
                    'bbox': {
                        'x1': table.bbox.x1,
                        'y1': table.bbox.y1,
                        'x2': table.bbox.x2,
                        'y2': table.bbox.y2
                    },
                    'shape': {
                        'rows': df.shape[0],
                        'cols': df.shape[1]
                    },
                    'dataframe': df
                }
                results.append(result)

            except Exception as e:
                print(f"  ❌ Erreur conversion DataFrame: {e}")

        return results

    def export_results(self, results, filename_prefix="result"):
        """Exporter les résultats en JSON et CSV"""
        if not results:
            print("❌ Aucun résultat à exporter")
            return

        # Préparer les données pour l'export
        export_data = {
            'extraction_summary': {
                'total_tables': len(results),
                'source_type': results[0]['source_type'] if results else None
            },
            'tables': []
        }

        combined_df = pd.DataFrame()

        for result in results:
            table_info = {
                'table_id': result['table_id'],
                'bbox': result['bbox'],
                'shape': result['shape'],
                'data': result['dataframe'].to_dict('records')
            }
            export_data['tables'].append(table_info)

            # Combiner tous les DataFrames
            if not combined_df.empty:
                combined_df = pd.concat([combined_df, result['dataframe']], ignore_index=True)
            else:
                combined_df = result['dataframe'].copy()

        # Export JSON
        json_path = self.output_dir / f"{filename_prefix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON exporté: {json_path}")

        # Export CSV
        csv_path = self.output_dir / f"{filename_prefix}.csv"
        combined_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"💾 CSV exporté: {csv_path}")

        return json_path, csv_path

def main():
    """Fonction principale"""
    print("🚀 Démarrage du test img2table")
    print("=" * 50)

    extractor = TableExtractor()
    sample_dir = Path("sample_data")

    # Chercher fichiers de test
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpeg"))
    pdf_files = list(sample_dir.glob("*.pdf"))

    if not image_files and not pdf_files:
        print("❌ Aucun fichier de test trouvé dans sample_data/")
        print("Créez un fichier image (.jpg/.png) ou PDF contenant un tableau")
        return

    all_results = []

    # Traiter les images
    for img_path in image_files:
        print(f"\n🔄 Traitement: {img_path.name}")
        results = extractor.extract_from_image(str(img_path))
        if results:
            all_results.extend(results)

    # Traiter les PDFs
    for pdf_path in pdf_files:
        print(f"\n🔄 Traitement: {pdf_path.name}")
        results = extractor.extract_from_pdf(str(pdf_path))
        if results:
            all_results.extend(results)

    # Exporter tous les résultats
    if all_results:
        print(f"\n📊 Export de {len(all_results)} tableau(x)")
        extractor.export_results(all_results)

        print("\n✅ Extraction terminée avec succès!")
        print("📁 Consultez le dossier 'output/' pour les résultats")
    else:
        print("\n❌ Aucun tableau extrait")

if __name__ == "__main__":
    main()