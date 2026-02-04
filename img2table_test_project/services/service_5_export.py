#!/usr/bin/env python3
"""
SERVICE 5: EXPORT VERS FORMATS MULTIPLES
Exporte les données vers DataFrame pandas, JSON et CSV
Avec options de formatage et validation
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import csv
from openpyxl.styles import Font

class ExportService:
    """Service d'export vers formats multiples avec options avancées"""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def export_table_data(self,
                         clean_data: Dict[str, Any],
                         base_filename: str,
                         export_formats: List[str] = ['csv', 'json', 'excel']) -> Dict[str, str]:
        """
        Exporte les données nettoyées vers multiple formats

        Args:
            clean_data: Données nettoyées du service 4
            base_filename: Nom de base pour les fichiers de sortie
            export_formats: Liste des formats à exporter ('csv', 'json', 'excel', 'parquet')

        Returns:
            Dict avec chemins des fichiers créés
        """
        self.logger.info(f"📤 Export données: {base_filename}")

        exported_files = {}

        # Créer DataFrame à partir des données nettoyées
        df = self._create_dataframe(clean_data)

        if df is None or df.empty:
            self.logger.warning("Aucune donnée valide à exporter")
            return exported_files

        # Export selon formats demandés
        if 'csv' in export_formats:
            csv_path = self._export_to_csv(df, base_filename)
            exported_files['csv'] = csv_path

        if 'json' in export_formats:
            json_path = self._export_to_json(clean_data, df, base_filename)
            exported_files['json'] = json_path

        if 'excel' in export_formats:
            excel_path = self._export_to_excel(df, base_filename)
            exported_files['excel'] = excel_path

        if 'parquet' in export_formats:
            parquet_path = self._export_to_parquet(df, base_filename)
            exported_files['parquet'] = parquet_path

        # Générer rapport d'export
        report_path = self._generate_export_report(clean_data, df, base_filename, exported_files)
        exported_files['report'] = report_path

        self.logger.info(f"✅ Export terminé: {len(exported_files)} fichiers créés")
        return exported_files

    def _create_dataframe(self, clean_data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Créer DataFrame à partir des données nettoyées"""
        try:
            table_data = clean_data.get('table_data', [])
            if not table_data:
                return None

            # Extraire en-têtes et données
            headers = clean_data.get('headers', [])

            # Si pas d'en-têtes détectés, créer des noms génériques
            if not headers:
                max_cols = max(len(row) for row in table_data) if table_data else 0
                headers = [f"Colonne_{i+1}" for i in range(max_cols)]

            # Créer DataFrame
            df = pd.DataFrame(table_data, columns=headers[:len(table_data[0]) if table_data else 0])

            # Nettoyer les valeurs vides
            df = df.replace('', pd.NA)

            self.logger.info(f"📊 DataFrame créé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df

        except Exception as e:
            self.logger.error(f"Erreur création DataFrame: {e}")
            return None

    def _export_to_csv(self, df: pd.DataFrame, base_filename: str) -> str:
        """Export CSV avec encoding UTF-8"""
        csv_path = self.output_dir / f"{base_filename}.csv"

        try:
            df.to_csv(csv_path,
                     index=False,
                     encoding='utf-8-sig',  # BOM pour Excel
                     quoting=csv.QUOTE_MINIMAL,
                     escapechar='\\')

            self.logger.info(f"📄 CSV exporté: {csv_path}")
            return str(csv_path)

        except Exception as e:
            self.logger.error(f"Erreur export CSV: {e}")
            return ""

    def _export_to_json(self, clean_data: Dict[str, Any], df: pd.DataFrame, base_filename: str) -> str:
        """Export JSON avec métadonnées complètes"""
        json_path = self.output_dir / f"{base_filename}.json"

        try:
            # Structure JSON complète avec métadonnées
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "source_file": clean_data.get('source_file', 'unknown'),
                    "extraction_method": clean_data.get('extraction_method', 'pipeline'),
                    "confidence_score": clean_data.get('confidence_score', 0.0),
                    "table_dimensions": {
                        "rows": df.shape[0],
                        "columns": df.shape[1]
                    },
                    "data_quality": clean_data.get('quality_metrics', {})
                },
                "headers": list(df.columns),
                "data": df.to_dict('records'),
                "raw_extraction_info": {
                    "original_cell_count": clean_data.get('original_cell_count', 0),
                    "cleaned_cell_count": clean_data.get('cleaned_cell_count', 0),
                    "correction_count": clean_data.get('correction_count', 0)
                }
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📋 JSON exporté: {json_path}")
            return str(json_path)

        except Exception as e:
            self.logger.error(f"Erreur export JSON: {e}")
            return ""

    def _export_to_excel(self, df: pd.DataFrame, base_filename: str) -> str:
        """Export Excel avec formatage"""
        excel_path = self.output_dir / f"{base_filename}.xlsx"

        try:
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Feuille principale avec données
                df.to_excel(writer, sheet_name='Données', index=False)

                # Obtenir le worksheet pour formatage
                worksheet = writer.sheets['Données']

                # CORRECTION: Formatage des en-têtes avec openpyxl.styles.Font
                for col in range(1, len(df.columns) + 1):
                    cell = worksheet.cell(row=1, column=col)
                    cell.font = Font(bold=True)

                # Auto-ajuster largeur colonnes
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width

            self.logger.info(f"📊 Excel exporté: {excel_path}")
            return str(excel_path)

        except Exception as e:
            self.logger.error(f"Erreur export Excel: {e}")
            # Fallback vers export CSV simple
            return self._export_to_csv(df, f"{base_filename}_fallback")

    def _export_to_parquet(self, df: pd.DataFrame, base_filename: str) -> str:
        """Export Parquet pour traitement big data"""
        parquet_path = self.output_dir / f"{base_filename}.parquet"

        try:
            df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            self.logger.info(f"🗄️ Parquet exporté: {parquet_path}")
            return str(parquet_path)

        except ImportError:
            self.logger.warning("PyArrow non disponible pour export Parquet")
            return ""
        except Exception as e:
            self.logger.error(f"Erreur export Parquet: {e}")
            return ""

    def _generate_export_report(self,
                               clean_data: Dict[str, Any],
                               df: pd.DataFrame,
                               base_filename: str,
                               exported_files: Dict[str, str]) -> str:
        """Générer rapport détaillé d'export"""
        report_path = self.output_dir / f"{base_filename}_export_report.txt"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT D'EXPORT DÉTAILLÉ ===\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Fichier source: {clean_data.get('source_file', 'unknown')}\n")
                f.write(f"Méthode extraction: {clean_data.get('extraction_method', 'pipeline')}\n")
                f.write(f"Score confiance: {clean_data.get('confidence_score', 0.0):.2f}\n\n")

                f.write("DIMENSIONS DU TABLEAU:\n")
                f.write(f"  Lignes: {df.shape[0]}\n")
                f.write(f"  Colonnes: {df.shape[1]}\n")
                f.write(f"  Cellules totales: {df.shape[0] * df.shape[1]}\n\n")

                f.write("QUALITÉ DES DONNÉES:\n")
                quality_metrics = clean_data.get('quality_metrics', {})
                for metric, value in quality_metrics.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")

                f.write("COLONNES DÉTECTÉES:\n")
                for i, col in enumerate(df.columns, 1):
                    non_null_count = df[col].notna().sum()
                    f.write(f"  {i}: '{col}' ({non_null_count} valeurs non-vides)\n")
                f.write("\n")

                f.write("ÉCHANTILLON DE DONNÉES (5 premières lignes):\n")
                for i, (idx, row) in enumerate(df.head().iterrows()):
                    f.write(f"  {i+1}: {list(row.values)}\n")
                f.write("\n")

                f.write("FICHIERS EXPORTÉS:\n")
                for format_type, file_path in exported_files.items():
                    if file_path and format_type != 'report':
                        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                        f.write(f"  {format_type.upper()}: {file_path} ({file_size} bytes)\n")

                f.write("\n=== FIN DU RAPPORT ===\n")

            self.logger.info(f"📋 Rapport d'export: {report_path}")
            return str(report_path)

        except Exception as e:
            self.logger.error(f"Erreur génération rapport: {e}")
            return ""

    def validate_export_quality(self, exported_files: Dict[str, str]) -> Dict[str, bool]:
        """Valider la qualité des exports"""
        validation_results = {}

        for format_type, file_path in exported_files.items():
            if not file_path or format_type == 'report':
                continue

            try:
                file_path_obj = Path(file_path)

                # Vérifier existence et taille
                exists = file_path_obj.exists()
                size_ok = file_path_obj.stat().st_size > 0 if exists else False

                # Validation spécifique par format
                content_ok = False
                if format_type == 'csv' and exists:
                    df_test = pd.read_csv(file_path_obj)
                    content_ok = not df_test.empty
                elif format_type == 'json' and exists:
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        content_ok = 'data' in data and len(data['data']) > 0
                elif format_type == 'excel' and exists:
                    df_test = pd.read_excel(file_path_obj)
                    content_ok = not df_test.empty

                validation_results[format_type] = exists and size_ok and content_ok

            except Exception as e:
                self.logger.warning(f"Erreur validation {format_type}: {e}")
                validation_results[format_type] = False

        return validation_results

def main():
    """Test du service d'export"""
    service = ExportService()

    # Données d'exemple pour test
    sample_data = {
        'source_file': 'test_image.jpg',
        'extraction_method': 'pipeline_complet',
        'confidence_score': 0.85,
        'headers': ['Nom', 'Note1', 'Note2', 'Moyenne'],
        'table_data': [
            ['Kouassi Yao', '15', '12', '13.5'],
            ['Traoré Aicha', '18', '16', '17'],
            ['Koné Ibrahim', '14', '15', '14.5']
        ],
        'quality_metrics': {
            'completeness': 0.95,
            'accuracy': 0.88,
            'consistency': 0.92
        },
        'original_cell_count': 12,
        'cleaned_cell_count': 12,
        'correction_count': 3
    }

    # Test export
    exported_files = service.export_table_data(
        sample_data,
        'test_extraction',
        ['csv', 'json', 'excel']
    )

    # Validation
    validation = service.validate_export_quality(exported_files)

    print("📤 RÉSULTATS D'EXPORT:")
    for format_type, file_path in exported_files.items():
        status = "✅" if validation.get(format_type, False) else "❌"
        print(f"  {status} {format_type.upper()}: {file_path}")

if __name__ == "__main__":
    main()