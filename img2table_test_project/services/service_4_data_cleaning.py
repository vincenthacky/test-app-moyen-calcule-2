#!/usr/bin/env python3
"""
SERVICE 4: NETTOYAGE ET NORMALISATION
Nettoie les données OCR, corrige erreurs, normalise les types
et valide la cohérence pour données éducatives
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Import du service précédent
from .service_3_ocr_cell_by_cell import CellContent

@dataclass
class CleaningRule:
    """Règle de nettoyage"""
    name: str
    pattern: str
    replacement: str
    applies_to: List[str]  # ['text', 'numbers', 'all']
    priority: int = 1

@dataclass
class ValidationRule:
    """Règle de validation"""
    name: str
    column_type: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_patterns: Optional[List[str]] = None
    required: bool = False

class DataCleaningService:
    """Service dédié au nettoyage et normalisation des données extraites"""

    def __init__(self):
        # Configuration logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Règles de nettoyage prédéfinies
        self.cleaning_rules = self._setup_cleaning_rules()

        # Règles de validation pour données éducatives
        self.validation_rules = self._setup_validation_rules()

        # Dictionnaire de corrections communes
        self.correction_dict = self._setup_correction_dictionary()

    def _setup_cleaning_rules(self) -> List[CleaningRule]:
        """Définir règles de nettoyage des erreurs OCR communes"""
        return [
            # Corrections caractères OCR
            CleaningRule("fix_ocr_o_to_0", r"[oO](?=\d|$|\s)", "0", ["numbers"], 1),
            CleaningRule("fix_ocr_l_to_1", r"[lI](?=\d|$|\s)", "1", ["numbers"], 1),
            CleaningRule("fix_ocr_s_to_5", r"[sS](?=\d|$|\s)", "5", ["numbers"], 2),
            CleaningRule("fix_ocr_g_to_9", r"[g](?=\d|$|\s)", "9", ["numbers"], 2),
            CleaningRule("fix_ocr_b_to_6", r"[b](?=\d|$|\s)", "6", ["numbers"], 2),

            # Corrections espaces et ponctuation
            CleaningRule("remove_extra_spaces", r"\s+", " ", ["all"], 1),
            CleaningRule("remove_leading_trailing", r"^\s+|\s+$", "", ["all"], 1),
            CleaningRule("fix_comma_decimal", r"(\d),(\d)", r"\1.\2", ["numbers"], 1),

            # Corrections caractères spéciaux
            CleaningRule("remove_brackets", r"[\[\](){}]", "", ["text", "numbers"], 2),
            CleaningRule("fix_quotes", r"[''`´]", "'", ["text"], 2),
            CleaningRule("remove_artifacts", r"[|_\\\/]", "", ["all"], 3),

            # Corrections noms propres
            CleaningRule("fix_accents_ocr", r"[àáâãäå]", "a", ["text"], 3),
            CleaningRule("fix_e_accents", r"[èéêë]", "e", ["text"], 3),
            CleaningRule("fix_i_accents", r"[ìíîï]", "i", ["text"], 3),
            CleaningRule("fix_o_accents", r"[òóôõö]", "o", ["text"], 3),
            CleaningRule("fix_u_accents", r"[ùúûü]", "u", ["text"], 3),

            # Corrections spécifiques notes
            CleaningRule("normalize_note_format", r"(\d+)\s*[/\\]\s*(\d+)", r"\1/\2", ["numbers"], 1),
        ]

    def _setup_validation_rules(self) -> Dict[str, ValidationRule]:
        """Définir règles de validation pour données éducatives"""
        return {
            'note_sur_20': ValidationRule(
                name="Note sur 20",
                column_type="grade",
                min_value=0,
                max_value=20,
                allowed_patterns=[r"^\d{1,2}(\.\d{1,2})?$", r"^\d{1,2}/20$"]
            ),
            'coefficient': ValidationRule(
                name="Coefficient",
                column_type="coefficient",
                min_value=1,
                max_value=10,
                allowed_patterns=[r"^\d{1,2}$"]
            ),
            'student_name': ValidationRule(
                name="Nom étudiant",
                column_type="name",
                allowed_patterns=[r"^[A-Za-zÀ-ÿ\s\-']{2,50}$"],
                required=True
            ),
            'student_number': ValidationRule(
                name="Numéro étudiant",
                column_type="number",
                min_value=1,
                max_value=999,
                allowed_patterns=[r"^\d{1,3}$"]
            )
        }

    def _setup_correction_dictionary(self) -> Dict[str, str]:
        """Dictionnaire de corrections spécifiques"""
        return {
            # Corrections noms fréquents
            'KOUASS I': 'KOUASSI',
            'TRAORE': 'TRAORÉ',
            'KONE': 'KONÉ',
            'OUATTARA': 'OUATTARA',
            'DIALLO': 'DIALLO',
            'SANGARE': 'SANGARÉ',
            'CAMARA': 'CAMARA',
            'TOURE': 'TOURÉ',

            # Corrections termes éducatifs
            'COEFFIC IENT': 'COEFFICIENT',
            'NOTE PONDEREE': 'NOTE PONDÉRÉE',
            'MOYENNE': 'MOYENNE',
            'TOTAL': 'TOTAL',

            # Corrections chiffres mal lus
            'O': '0',
            'l': '1',
            'Z': '2',
            'S': '5',
            'G': '6',
            'T': '7',
            'B': '8',
            'g': '9'
        }

    def clean_table_data(self, cell_contents: List[CellContent]) -> Dict[str, Any]:
        """
        Méthode principale pour nettoyer et structurer les données du tableau
        Compatible avec le pipeline orchestrateur
        """
        # Nettoyer les cellules
        cleaned_contents = self.clean_extracted_data(cell_contents)

        # Structurer en DataFrame
        df = self.validate_and_structure_data(cleaned_contents)

        # Retourner format attendu par le pipeline
        return {
            'table_data': df.values.tolist() if not df.empty else [],
            'headers': df.columns.tolist() if not df.empty else [],
            'dataframe': df,
            'confidence_score': self._calculate_confidence(cleaned_contents),
            'correction_count': sum(1 for orig, clean in zip(cell_contents, cleaned_contents)
                                   if orig.text != clean.text)
        }

    def _calculate_confidence(self, contents: List[CellContent]) -> float:
        """Calculer score de confiance moyen"""
        if not contents:
            return 0.0
        return sum(c.confidence for c in contents) / len(contents)

    def clean_extracted_data(self, cell_contents: List[CellContent]) -> List[CellContent]:
        """Nettoyer toutes les données extraites"""
        self.logger.info(f"🧹 Nettoyage de {len(cell_contents)} cellules")

        cleaned_contents = []

        for content in cell_contents:
            cleaned_content = self._clean_single_cell(content)
            cleaned_contents.append(cleaned_content)

        # Statistiques de nettoyage
        changes = sum(1 for orig, cleaned in zip(cell_contents, cleaned_contents)
                     if orig.text != cleaned.text)

        self.logger.info(f"✅ Nettoyage terminé: {changes} cellules modifiées")
        return cleaned_contents

    def _clean_single_cell(self, content: CellContent) -> CellContent:
        """Nettoyer le contenu d'une seule cellule"""
        if not content.text.strip():
            return content

        original_text = content.text
        cleaned_text = original_text

        # Détecter le type de contenu
        content_type = self._detect_content_type(cleaned_text)

        # Appliquer règles de nettoyage par priorité
        sorted_rules = sorted(self.cleaning_rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            if content_type in rule.applies_to or 'all' in rule.applies_to:
                cleaned_text = re.sub(rule.pattern, rule.replacement, cleaned_text)

        # Appliquer dictionnaire de corrections
        cleaned_text = self._apply_correction_dictionary(cleaned_text, content_type)

        # Post-traitement selon type
        cleaned_text = self._post_process_by_type(cleaned_text, content_type)

        # Créer nouveau contenu nettoyé
        return CellContent(
            text=cleaned_text.strip(),
            confidence=content.confidence,
            cell_region=content.cell_region,
            ocr_method=content.ocr_method,
            preprocessing_applied=content.preprocessing_applied + ["cleaned"]
        )

    def _detect_content_type(self, text: str) -> str:
        """Détecter le type de contenu d'une cellule"""
        text = text.strip()

        if not text:
            return 'empty'

        # Notes (chiffres avec ou sans décimales)
        if re.match(r'^\d{1,2}(\.\d{1,2})?(/\d+)?$', text):
            return 'numbers'

        # Noms (lettres, espaces, tirets, apostrophes)
        if re.match(r'^[A-Za-zÀ-ÿ\s\-\']{2,}$', text):
            return 'text'

        # Mixte ou autre
        return 'mixed'

    def _apply_correction_dictionary(self, text: str, content_type: str) -> str:
        """Appliquer le dictionnaire de corrections"""
        text_upper = text.upper()

        for incorrect, correct in self.correction_dict.items():
            if incorrect.upper() in text_upper:
                # Remplacer en préservant la casse
                text = re.sub(re.escape(incorrect), correct, text, flags=re.IGNORECASE)

        return text

    def _post_process_by_type(self, text: str, content_type: str) -> str:
        """Post-traitement spécialisé selon le type"""

        if content_type == 'numbers':
            return self._post_process_numbers(text)
        elif content_type == 'text':
            return self._post_process_text(text)
        else:
            return text.strip()

    def _post_process_numbers(self, text: str) -> str:
        """Post-traitement spécialisé pour les nombres"""
        # Normaliser format décimal
        text = re.sub(r'(\d),(\d)', r'\1.\2', text)

        # Supprimer caractères non numériques sauf . et /
        text = re.sub(r'[^\d\./]', '', text)

        # Valider format note
        if '/' in text:
            parts = text.split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                note, total = int(parts[0]), int(parts[1])
                if total == 20:  # Note sur 20
                    return f"{note}/20" if 0 <= note <= 20 else text
                else:
                    return text
        else:
            # Nombre simple
            try:
                num = float(text)
                if 0 <= num <= 20:  # Probable note
                    return f"{num:.1f}".rstrip('0').rstrip('.')
            except ValueError:
                pass

        return text

    def _post_process_text(self, text: str) -> str:
        """Post-traitement spécialisé pour le texte"""
        # Capitaliser proprement les noms
        words = text.split()
        capitalized_words = []

        for word in words:
            if len(word) > 1:
                # Noms propres : première lettre majuscule
                capitalized_words.append(word[0].upper() + word[1:].lower())
            else:
                capitalized_words.append(word.upper())

        return ' '.join(capitalized_words)

    def validate_and_structure_data(self, cell_contents: List[CellContent]) -> pd.DataFrame:
        """Structurer et valider les données en DataFrame"""
        self.logger.info(f"📊 Structuration en DataFrame")

        # Créer structure de base
        df_data = self._create_dataframe_structure(cell_contents)

        if df_data.empty:
            self.logger.warning("Aucune donnée structurable")
            return df_data

        # Identifier types de colonnes
        column_types = self._identify_column_types(df_data)

        # Appliquer validation et corrections
        validated_df = self._apply_validation_rules(df_data, column_types)

        # Rapport de validation
        self._generate_validation_report(df_data, validated_df, column_types)

        return validated_df

    def _create_dataframe_structure(self, cell_contents: List[CellContent]) -> pd.DataFrame:
        """Créer DataFrame à partir des cellules"""

        if not cell_contents:
            return pd.DataFrame()

        # Déterminer dimensions du tableau
        max_row = max(content.cell_region.row for content in cell_contents)
        max_col = max(content.cell_region.col for content in cell_contents)

        # Créer matrice vide
        data_matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        # Remplir matrice
        for content in cell_contents:
            row, col = content.cell_region.row, content.cell_region.col
            data_matrix[row][col] = content.text

        # Identifier header et données
        if len(data_matrix) > 0:
            headers = data_matrix[0] if data_matrix[0] else [f"Col_{i}" for i in range(len(data_matrix[0]))]
            data_rows = data_matrix[1:] if len(data_matrix) > 1 else []

            # Nettoyer headers
            cleaned_headers = []
            for i, header in enumerate(headers):
                if header.strip():
                    cleaned_headers.append(header.strip())
                else:
                    cleaned_headers.append(f"Colonne_{i}")

            # Créer DataFrame
            if data_rows:
                df = pd.DataFrame(data_rows, columns=cleaned_headers)
            else:
                df = pd.DataFrame(columns=cleaned_headers)

            # Supprimer lignes/colonnes vides
            df = df.dropna(how='all').dropna(axis=1, how='all')

            return df

        return pd.DataFrame()

    def _identify_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identifier le type de chaque colonne"""
        column_types = {}

        for col in df.columns:
            col_data = df[col].dropna().astype(str)

            if len(col_data) == 0:
                column_types[col] = 'empty'
                continue

            # Analyser contenu
            numeric_count = 0
            name_count = 0
            mixed_count = 0

            for value in col_data:
                value_clean = value.strip()
                if not value_clean:
                    continue

                # Test nombre/note
                if re.match(r'^\d{1,2}(\.\d{1,2})?(/\d+)?$', value_clean):
                    numeric_count += 1
                # Test nom
                elif re.match(r'^[A-Za-zÀ-ÿ\s\-\']{2,}$', value_clean):
                    name_count += 1
                else:
                    mixed_count += 1

            total_values = len(col_data)

            # Classifier par majorité
            if numeric_count / total_values > 0.6:
                if any(val for val in col_data if '/' in val and '20' in val):
                    column_types[col] = 'grade'
                elif any(val for val in col_data if val.isdigit() and 1 <= int(val) <= 10):
                    column_types[col] = 'coefficient'
                else:
                    column_types[col] = 'number'
            elif name_count / total_values > 0.6:
                if any(keyword in col.lower() for keyword in ['nom', 'name', 'élève', 'student']):
                    column_types[col] = 'name'
                else:
                    column_types[col] = 'text'
            else:
                column_types[col] = 'mixed'

        return column_types

    def _apply_validation_rules(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Appliquer règles de validation"""

        validated_df = df.copy()

        for col, col_type in column_types.items():
            if col_type in ['grade', 'coefficient', 'number']:
                validated_df[col] = validated_df[col].apply(
                    lambda x: self._validate_numeric_value(x, col_type)
                )
            elif col_type == 'name':
                validated_df[col] = validated_df[col].apply(
                    self._validate_name_value
                )

        return validated_df

    def _validate_numeric_value(self, value: str, value_type: str) -> str:
        """Valider et corriger valeurs numériques"""
        value = str(value).strip()

        if not value:
            return value

        # Pour les notes
        if value_type == 'grade':
            # Note sur 20
            if '/' in value:
                try:
                    note_part = value.split('/')[0]
                    note = float(note_part)
                    return f"{note:.1f}/20" if 0 <= note <= 20 else value
                except ValueError:
                    return value
            else:
                try:
                    note = float(value)
                    return f"{note:.1f}" if 0 <= note <= 20 else value
                except ValueError:
                    return value

        # Pour les coefficients
        elif value_type == 'coefficient':
            try:
                coef = int(float(value))
                return str(coef) if 1 <= coef <= 10 else value
            except ValueError:
                return value

        return value

    def _validate_name_value(self, value: str) -> str:
        """Valider et corriger noms"""
        value = str(value).strip()

        if not value:
            return value

        # Supprimer caractères non alphabétiques sauf espaces, tirets, apostrophes
        cleaned = re.sub(r"[^A-Za-zÀ-ÿ\s\-']", "", value)

        # Capitaliser correctement
        words = cleaned.split()
        capitalized = []

        for word in words:
            if len(word) > 0:
                if len(word) == 1:
                    capitalized.append(word.upper())
                else:
                    capitalized.append(word[0].upper() + word[1:].lower())

        return ' '.join(capitalized)

    def _generate_validation_report(self, original_df: pd.DataFrame,
                                  validated_df: pd.DataFrame,
                                  column_types: Dict[str, str]) -> Dict:
        """Générer rapport de validation"""

        changes = 0
        for col in original_df.columns:
            if col in validated_df.columns:
                changes += (original_df[col] != validated_df[col]).sum()

        report = {
            'total_cells': original_df.size,
            'changes_made': changes,
            'column_types_detected': column_types,
            'validation_summary': {
                'grades_found': sum(1 for t in column_types.values() if t == 'grade'),
                'names_found': sum(1 for t in column_types.values() if t == 'name'),
                'numbers_found': sum(1 for t in column_types.values() if t in ['number', 'coefficient'])
            }
        }

        self.logger.info(f"📋 Validation: {changes} corrections appliquées")
        return report

def main():
    """Test du service de nettoyage"""

    # Données de test simulées (normalement viennent du service OCR)
    from service_2_structure_detection import CellRegion

    test_contents = [
        CellContent("N° ", 0.9, CellRegion(0, 0, 50, 30, 0, 0), "tesseract", []),
        CellContent("Nom de l'élève", 0.95, CellRegion(50, 0, 150, 30, 0, 1), "tesseract", []),
        CellContent("Note /2O", 0.8, CellRegion(200, 0, 100, 30, 0, 2), "tesseract", []),
        CellContent("1", 0.9, CellRegion(0, 30, 50, 30, 1, 0), "tesseract", []),
        CellContent("KOUASS I Yao", 0.85, CellRegion(50, 30, 150, 30, 1, 1), "tesseract", []),
        CellContent("l5.5", 0.7, CellRegion(200, 30, 100, 30, 1, 2), "tesseract", []),
    ]

    cleaning_service = DataCleaningService()

    print("🧪 Test du service de nettoyage")

    # 1. Nettoyage
    cleaned_contents = cleaning_service.clean_extracted_data(test_contents)

    print("\n📝 Avant/Après nettoyage:")
    for orig, clean in zip(test_contents, cleaned_contents):
        if orig.text != clean.text:
            print(f"  '{orig.text}' -> '{clean.text}'")

    # 2. Structuration
    df = cleaning_service.validate_and_structure_data(cleaned_contents)

    print(f"\n📊 DataFrame structuré:")
    print(df.to_string(index=False))

    # 3. Types détectés
    column_types = cleaning_service._identify_column_types(df)
    print(f"\n🏷️ Types de colonnes: {column_types}")

if __name__ == "__main__":
    main()