#!/usr/bin/env python3
"""
PIPELINE ORCHESTRATEUR PRINCIPAL
Coordonne les 5 services pour extraction complète de tableaux
Architecture robuste pour traitement universel d'images/PDF
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import des 5 services
from services.service_1_preprocessing import PreprocessingService
from services.service_2_structure_detection import StructureDetectionService
from services.service_3_ocr_cell_by_cell import CellOCRService
from services.service_4_data_cleaning import DataCleaningService
from services.service_5_export import ExportService

class TableExtractionPipeline:
    """Pipeline orchestrateur pour extraction complète de tableaux"""

    def __init__(self,
                 output_dir: str = "output",
                 enable_detailed_logging: bool = True,
                 fallback_on_errors: bool = True):
        """
        Initialiser le pipeline avec tous les services

        Args:
            output_dir: Répertoire de sortie pour tous les fichiers
            enable_detailed_logging: Activer logs détaillés
            fallback_on_errors: Permettre fallback en cas d'erreur
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.fallback_on_errors = fallback_on_errors

        # Configuration logging
        log_level = logging.DEBUG if enable_detailed_logging else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialisation des 5 services
        self.logger.info("🚀 Initialisation pipeline extraction")

        self.preprocessing_service = PreprocessingService(str(self.output_dir / "preprocessed"))
        self.structure_service = StructureDetectionService()
        self.ocr_service = CellOCRService()
        self.cleaning_service = DataCleaningService()
        self.export_service = ExportService(str(self.output_dir))

        # Métriques pipeline
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'services_executed': [],
            'errors_encountered': [],
            'files_created': []
        }

    def process_document(self,
                        input_file: str,
                        export_formats: List[str] = ['csv', 'json', 'excel'],
                        custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Traitement complet d'un document image/PDF

        Args:
            input_file: Chemin vers fichier d'entrée
            export_formats: Formats d'export souhaités
            custom_config: Configuration personnalisée

        Returns:
            Résultats complets du pipeline avec chemins de fichiers
        """
        self.pipeline_stats['start_time'] = datetime.now()
        self.logger.info(f"📋 DÉMARRAGE PIPELINE: {Path(input_file).name}")

        pipeline_results = {
            'success': False,
            'input_file': input_file,
            'processed_files': {},
            'extraction_data': {},
            'export_files': {},
            'pipeline_stats': {},
            'errors': []
        }

        try:
            # ÉTAPE 1: Prétraitement
            self.logger.info("🔧 ÉTAPE 1/5: Prétraitement")
            preprocessed_path = self._execute_service_with_fallback(
                self.preprocessing_service.process_document,
                input_file,
                service_name="preprocessing"
            )

            if not preprocessed_path:
                raise Exception("Échec prétraitement - arrêt pipeline")

            pipeline_results['processed_files']['preprocessed'] = preprocessed_path

            # ÉTAPE 2: Détection structure
            self.logger.info("🔍 ÉTAPE 2/5: Détection structure")
            table_structure = self._execute_service_with_fallback(
                self.structure_service.detect_table_structure,
                preprocessed_path,
                service_name="structure_detection"
            )

            if not table_structure:
                raise Exception("Échec détection structure - arrêt pipeline")

            pipeline_results['extraction_data']['structure'] = table_structure

            # ÉTAPE 3: OCR cellule par cellule
            self.logger.info("👁️ ÉTAPE 3/5: OCR cellule par cellule")
            ocr_results = self._execute_service_with_fallback(
                self.ocr_service.extract_cells_content,
                preprocessed_path,
                service_name="ocr_extraction",
                structure=table_structure
            )

            if not ocr_results:
                raise Exception("Échec OCR - arrêt pipeline")

            pipeline_results['extraction_data']['ocr_raw'] = ocr_results

            # ÉTAPE 4: Nettoyage et validation
            self.logger.info("🧹 ÉTAPE 4/5: Nettoyage données")
            cleaned_data = self._execute_service_with_fallback(
                self.cleaning_service.clean_table_data,
                ocr_results,
                service_name="data_cleaning"
            )

            if not cleaned_data:
                raise Exception("Échec nettoyage - arrêt pipeline")

            pipeline_results['extraction_data']['cleaned'] = cleaned_data

            # ÉTAPE 5: Export multi-format
            self.logger.info("📤 ÉTAPE 5/5: Export données")
            base_filename = Path(input_file).stem
            exported_files = self._execute_service_with_fallback(
                self.export_service.export_table_data,
                cleaned_data,
                service_name="export",
                base_filename=base_filename,
                export_formats=export_formats
            )

            if exported_files:
                pipeline_results['export_files'] = exported_files

            # Finalisation
            self.pipeline_stats['end_time'] = datetime.now()
            self.pipeline_stats['total_duration'] = (
                self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']
            ).total_seconds()

            pipeline_results['success'] = True
            pipeline_results['pipeline_stats'] = self.pipeline_stats.copy()

            # Génération rapport final
            report_path = self._generate_pipeline_report(pipeline_results)
            pipeline_results['export_files']['pipeline_report'] = report_path

            self.logger.info(f"✅ PIPELINE TERMINÉ: {self.pipeline_stats['total_duration']:.2f}s")
            return pipeline_results

        except Exception as e:
            self.logger.error(f"❌ ERREUR PIPELINE: {e}")
            pipeline_results['errors'].append(str(e))

            # Générer rapport d'erreur même en cas d'échec
            error_report_path = self._generate_error_report(pipeline_results, str(e))
            pipeline_results['export_files']['error_report'] = error_report_path

            return pipeline_results

    def _execute_service_with_fallback(self,
                                      service_method,
                                      *args,
                                      service_name: str,
                                      **kwargs) -> Any:
        """Exécuter service avec gestion d'erreur et fallback"""
        try:
            self.logger.debug(f"🔄 Exécution service: {service_name}")
            start_time = time.time()

            result = service_method(*args, **kwargs)

            execution_time = time.time() - start_time
            self.pipeline_stats['services_executed'].append({
                'service': service_name,
                'execution_time': execution_time,
                'status': 'success'
            })

            self.logger.info(f"✅ {service_name} terminé: {execution_time:.2f}s")
            return result

        except Exception as e:
            error_msg = f"Erreur {service_name}: {e}"
            self.logger.error(f"❌ {error_msg}")

            self.pipeline_stats['errors_encountered'].append(error_msg)
            self.pipeline_stats['services_executed'].append({
                'service': service_name,
                'execution_time': 0,
                'status': 'failed',
                'error': str(e)
            })

            if self.fallback_on_errors and service_name in ['structure_detection', 'ocr_extraction']:
                self.logger.warning(f"🔄 Tentative fallback pour {service_name}")
                return self._attempt_fallback(service_name, *args, **kwargs)

            return None

    def _attempt_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Tenter méthodes de fallback pour services critiques"""
        try:
            if service_name == 'structure_detection':
                # Fallback: détection simple basée sur lignes
                self.logger.info("🔧 Fallback détection structure simple")
                return self.structure_service.detect_simple_grid(*args, **kwargs)

            elif service_name == 'ocr_extraction':
                # Fallback: OCR global sans structure cellule par cellule
                self.logger.info("🔧 Fallback OCR global")
                return self.ocr_service.extract_global_text(*args, **kwargs)

        except Exception as fallback_error:
            self.logger.error(f"❌ Échec fallback {service_name}: {fallback_error}")
            return None

    def _generate_pipeline_report(self, pipeline_results: Dict[str, Any]) -> str:
        """Générer rapport détaillé du pipeline"""
        report_path = self.output_dir / "pipeline_execution_report.txt"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT D'EXÉCUTION PIPELINE ===\n\n")

                f.write(f"Fichier traité: {pipeline_results['input_file']}\n")
                f.write(f"Succès: {'✅ OUI' if pipeline_results['success'] else '❌ NON'}\n")
                f.write(f"Durée totale: {pipeline_results.get('pipeline_stats', {}).get('total_duration', 0):.2f}s\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

                # Détail services
                f.write("EXÉCUTION DES SERVICES:\n")
                for service_info in self.pipeline_stats.get('services_executed', []):
                    status_icon = "✅" if service_info['status'] == 'success' else "❌"
                    f.write(f"  {status_icon} {service_info['service']}: {service_info['execution_time']:.2f}s\n")
                    if service_info.get('error'):
                        f.write(f"    Erreur: {service_info['error']}\n")
                f.write("\n")

                # Fichiers créés
                f.write("FICHIERS CRÉÉS:\n")
                for file_type, file_path in pipeline_results.get('export_files', {}).items():
                    if file_path:
                        f.write(f"  📄 {file_type}: {file_path}\n")
                f.write("\n")

                # Données extraites
                if pipeline_results.get('extraction_data', {}).get('cleaned'):
                    cleaned = pipeline_results['extraction_data']['cleaned']
                    f.write("RÉSUMÉ EXTRACTION:\n")
                    f.write(f"  Lignes détectées: {len(cleaned.get('table_data', []))}\n")
                    f.write(f"  Colonnes: {len(cleaned.get('headers', []))}\n")
                    f.write(f"  Score confiance: {cleaned.get('confidence_score', 0):.2f}\n")
                    f.write(f"  Corrections appliquées: {cleaned.get('correction_count', 0)}\n")

                f.write("\n=== FIN RAPPORT ===\n")

            return str(report_path)

        except Exception as e:
            self.logger.error(f"Erreur génération rapport: {e}")
            return ""

    def _generate_error_report(self, pipeline_results: Dict[str, Any], error: str) -> str:
        """Générer rapport en cas d'erreur pipeline"""
        error_report_path = self.output_dir / "pipeline_error_report.txt"

        try:
            with open(error_report_path, 'w', encoding='utf-8') as f:
                f.write("=== RAPPORT D'ERREUR PIPELINE ===\n\n")

                f.write(f"Fichier: {pipeline_results['input_file']}\n")
                f.write(f"Erreur principale: {error}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")

                f.write("ERREURS RENCONTRÉES:\n")
                for err in pipeline_results.get('errors', []):
                    f.write(f"  ❌ {err}\n")

                f.write("\nSERVICES EXÉCUTÉS AVEC SUCCÈS:\n")
                for service_info in self.pipeline_stats.get('services_executed', []):
                    if service_info['status'] == 'success':
                        f.write(f"  ✅ {service_info['service']}\n")

                f.write("\n=== RECOMMANDATIONS ===\n")
                f.write("1. Vérifier qualité image d'entrée\n")
                f.write("2. S'assurer que le tableau est bien visible\n")
                f.write("3. Réessayer avec image prétraitée manuellement\n")

            return str(error_report_path)

        except Exception as e:
            self.logger.error(f"Erreur génération rapport d'erreur: {e}")
            return ""

    def process_multiple_documents(self,
                                  file_list: List[str],
                                  export_formats: List[str] = ['csv', 'json']) -> Dict[str, Any]:
        """Traiter multiple documents en lot"""
        self.logger.info(f"📦 TRAITEMENT LOT: {len(file_list)} fichiers")

        batch_results = {
            'total_files': len(file_list),
            'successful': 0,
            'failed': 0,
            'results': {},
            'batch_summary': {}
        }

        for file_path in file_list:
            try:
                self.logger.info(f"📋 Traitement: {Path(file_path).name}")
                result = self.process_document(file_path, export_formats)

                batch_results['results'][file_path] = result
                if result['success']:
                    batch_results['successful'] += 1
                else:
                    batch_results['failed'] += 1

            except Exception as e:
                self.logger.error(f"Erreur traitement {file_path}: {e}")
                batch_results['failed'] += 1

        # Générer résumé lot
        success_rate = (batch_results['successful'] / batch_results['total_files']) * 100
        batch_results['batch_summary'] = {
            'success_rate': success_rate,
            'total_duration': sum(
                r.get('pipeline_stats', {}).get('total_duration', 0)
                for r in batch_results['results'].values()
            )
        }

        self.logger.info(f"📊 LOT TERMINÉ: {success_rate:.1f}% succès")
        return batch_results

def main():
    """Test du pipeline complet"""
    pipeline = TableExtractionPipeline()

    # Tester avec image échantillon
    test_file = "sample_data/ma_photo.jpeg"

    if Path(test_file).exists():
        print(f"🧪 Test pipeline avec: {test_file}")

        results = pipeline.process_document(
            test_file,
            export_formats=['csv', 'json', 'excel']
        )

        if results['success']:
            print("✅ Pipeline exécuté avec succès!")
            print(f"📄 Fichiers créés: {len(results['export_files'])}")
            for file_type, path in results['export_files'].items():
                print(f"  - {file_type}: {path}")
        else:
            print("❌ Échec pipeline")
            print(f"Erreurs: {results['errors']}")

    else:
        print(f"⚠️ Fichier test non trouvé: {test_file}")
        print("Placez une image dans sample_data/ pour tester")

if __name__ == "__main__":
    main()