#!/usr/bin/env python3
"""
TEST SIMPLE DU PIPELINE COMPLET
Version simplifiée qui simule les données OCR pour tester l'architecture
"""

from services.service_1_preprocessing import PreprocessingService
from services.service_2_structure_detection import StructureDetectionService
from services.service_4_data_cleaning import DataCleaningService
from services.service_5_export import ExportService
from pathlib import Path

def test_pipeline_simple():
    """Test simplifié du pipeline sans OCR réel"""
    print("🧪 TEST PIPELINE SIMPLIFIÉ")

    # Service 1: Prétraitement
    print("\n🔧 ÉTAPE 1/5: Prétraitement")
    preprocessing = PreprocessingService("output/preprocessed")

    input_file = "sample_data/ma_photo.jpeg"
    if not Path(input_file).exists():
        print(f"❌ Fichier non trouvé: {input_file}")
        return

    preprocessed_path = preprocessing.process_document(input_file)
    print(f"✅ Preprocessed: {preprocessed_path}")

    # Service 2: Structure
    print("\n🔍 ÉTAPE 2/5: Détection structure")
    structure_service = StructureDetectionService()
    structure = structure_service.detect_table_structure(preprocessed_path)
    print(f"✅ Structure: {structure.num_rows}x{structure.num_cols} cellules")

    # Service 3: OCR simulé
    print("\n👁️ ÉTAPE 3/5: OCR simulé")
    # Simuler données OCR basées sur la structure détectée
    simulated_ocr_data = []
    sample_names = ["Kouassi Yao", "Traoré Aicha", "Koné Ibrahim", "Ouattara Karim"]
    sample_grades = ["15", "12", "18", "14", "16", "11"]

    # Créer données simulées pour la structure détectée
    for row in range(structure.num_rows):
        row_data = []
        for col in range(structure.num_cols):
            if row == 0:  # En-tête
                if col == 0:
                    text = "Nom"
                else:
                    text = f"Note{col}"
            else:  # Données
                if col == 0:
                    text = sample_names[(row-1) % len(sample_names)]
                else:
                    text = sample_grades[(row*col) % len(sample_grades)]

            # Simuler structure CellRegion
            from services.service_2_structure_detection import CellRegion
            cell_region = CellRegion(
                x=col*100,
                y=row*50,
                width=100,
                height=50,
                row=row,
                col=col,
                confidence=0.9
            )

            # Simuler structure CellContent
            cell_content = type('CellContent', (), {
                'text': text,
                'confidence': 0.85,
                'cell_region': cell_region,
                'ocr_method': 'simulated',
                'preprocessing_applied': ['test']
            })()

            simulated_ocr_data.append(cell_content)

    print(f"✅ OCR simulé: {len(simulated_ocr_data)} cellules")

    # Service 4: Nettoyage
    print("\n🧹 ÉTAPE 4/5: Nettoyage données")
    cleaning_service = DataCleaningService()

    cleaned_data = cleaning_service.clean_extracted_data(simulated_ocr_data)
    print(f"✅ Données nettoyées: {len(cleaned_data)} cellules")

    # Service 5: Export
    print("\n📤 ÉTAPE 5/5: Export données")
    export_service = ExportService("output")

    # Adapter format pour export: convertir liste de cellules en structure tableau
    table_data = []
    headers = []

    # Organiser cellules par position
    max_row = max(cell.cell_region.row for cell in cleaned_data)
    max_col = max(cell.cell_region.col for cell in cleaned_data)

    # Créer matrice de données
    matrix = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cleaned_data:
        matrix[cell.cell_region.row][cell.cell_region.col] = cell.text

    # Extraire en-têtes (première ligne) et données
    if matrix:
        headers = matrix[0] if matrix[0] else [f"Col_{i}" for i in range(max_col + 1)]
        table_data = matrix[1:] if len(matrix) > 1 else []

    # Structure pour export
    export_data = {
        'source_file': input_file,
        'extraction_method': 'pipeline_test',
        'confidence_score': 0.85,
        'headers': headers,
        'table_data': table_data,
        'quality_metrics': {
            'completeness': 0.95,
            'accuracy': 0.88,
            'consistency': 0.92
        },
        'original_cell_count': len(simulated_ocr_data),
        'cleaned_cell_count': len(cleaned_data),
        'correction_count': sum(1 for cell in cleaned_data if hasattr(cell, '_corrected') and cell._corrected)
    }

    exported_files = export_service.export_table_data(
        export_data,
        "test_pipeline",
        ['csv', 'json', 'excel']
    )

    print("\n🎉 PIPELINE TERMINÉ!")
    print("\n📄 FICHIERS CRÉÉS:")
    for format_type, file_path in exported_files.items():
        if file_path:
            print(f"  ✅ {format_type.upper()}: {file_path}")

    # Validation finale
    print("\n🔍 VALIDATION:")
    validation = export_service.validate_export_quality(exported_files)
    for format_type, is_valid in validation.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {format_type.upper()}: {'Valide' if is_valid else 'Erreur'}")

if __name__ == "__main__":
    test_pipeline_simple()