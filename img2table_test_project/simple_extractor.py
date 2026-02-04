#!/usr/bin/env python3
"""
EXTRACTEUR SIMPLIFIÉ DE TABLEAUX
Utilise img2table directement pour une extraction fiable
"""

import argparse
from pathlib import Path
from img2table.document import Image, PDF
from img2table.ocr import EasyOCR, TesseractOCR
import pandas as pd
import json
from datetime import datetime


def extract_table(input_file: str, output_dir: str = "output", ocr_engine: str = "easyocr"):
    """
    Extrait les tableaux d'une image ou PDF

    Args:
        input_file: Chemin vers l'image ou PDF
        output_dir: Répertoire de sortie
        ocr_engine: 'easyocr' ou 'tesseract'
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"📄 Extraction de: {input_path.name}")

    # Configurer OCR
    if ocr_engine == "easyocr":
        ocr = EasyOCR(lang=['fr', 'en'])
        print("🔧 OCR: EasyOCR (français + anglais)")
    else:
        ocr = TesseractOCR(lang='fra+eng')
        print("🔧 OCR: Tesseract (français + anglais)")

    # Charger document
    if input_path.suffix.lower() == '.pdf':
        doc = PDF(src=str(input_path))
    else:
        doc = Image(src=str(input_path))

    # Extraire tableaux
    print("🔍 Détection des tableaux...")
    tables = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=True,
        min_confidence=50
    )

    print(f"✅ {len(tables)} tableau(x) trouvé(s)")

    results = {
        'source_file': input_path.name,
        'extraction_timestamp': datetime.now().isoformat(),
        'ocr_engine': ocr_engine,
        'tables_count': len(tables),
        'tables': []
    }

    base_name = input_path.stem

    for i, table in enumerate(tables):
        df = table.df

        # Nettoyer les données
        df = clean_dataframe(df)

        print(f"\n📊 Tableau {i+1}: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        print(df.to_string())

        # Exporter CSV
        csv_path = output_path / f"{base_name}_table_{i+1}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"💾 CSV: {csv_path}")

        # Exporter Excel
        excel_path = output_path / f"{base_name}_table_{i+1}.xlsx"
        df.to_excel(excel_path, index=False)
        print(f"💾 Excel: {excel_path}")

        results['tables'].append({
            'table_index': i + 1,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'headers': list(df.columns),
            'data': df.to_dict('records'),
            'csv_file': str(csv_path),
            'excel_file': str(excel_path)
        })

    # Exporter JSON avec métadonnées
    json_path = output_path / f"{base_name}_extraction.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📋 JSON: {json_path}")

    return results


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le DataFrame extrait"""
    # Remplacer les retours à la ligne par des espaces
    df = df.replace(r'\n', ' ', regex=True)

    # Supprimer espaces multiples
    df = df.replace(r'\s+', ' ', regex=True)

    # Nettoyer les espaces en début/fin
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # Renommer colonnes si elles sont numériques
    if all(isinstance(c, int) for c in df.columns):
        # Première ligne comme en-tête si elle contient du texte
        first_row = df.iloc[0]
        if first_row.notna().any():
            new_headers = []
            for val in first_row:
                if pd.notna(val):
                    # Nettoyer l'en-tête
                    header = str(val).replace('\n', ' ').strip()
                    new_headers.append(header)
                else:
                    new_headers.append(f"Col_{len(new_headers)+1}")
            df.columns = new_headers
            df = df.iloc[1:].reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser(description='Extraction simplifiée de tableaux')
    parser.add_argument('input_file', help='Image ou PDF à traiter')
    parser.add_argument('--output-dir', '-o', default='output', help='Répertoire de sortie')
    parser.add_argument('--ocr', choices=['easyocr', 'tesseract'], default='easyocr',
                       help='Moteur OCR à utiliser')

    args = parser.parse_args()

    results = extract_table(args.input_file, args.output_dir, args.ocr)

    print(f"\n✅ Extraction terminée: {results['tables_count']} tableau(x)")


if __name__ == "__main__":
    main()
