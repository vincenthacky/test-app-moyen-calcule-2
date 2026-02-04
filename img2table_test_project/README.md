# Extracteur de Tableaux

Outil simple pour extraire des tableaux depuis des images ou PDFs.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Utilisation

```bash
# Extraction avec EasyOCR (recommandé)
python simple_extractor.py mon_image.jpeg

# Extraction avec Tesseract
python simple_extractor.py mon_image.jpeg --ocr tesseract

# Spécifier le dossier de sortie
python simple_extractor.py mon_image.jpeg --output-dir resultats/
```

## Formats supportés

**Entrée :** JPEG, PNG, PDF

**Sortie :** CSV, Excel (.xlsx), JSON

## Exemple

```bash
python simple_extractor.py ma_photo.jpeg
```

Résultat :
```
📄 Extraction de: ma_photo.jpeg
🔧 OCR: EasyOCR (français + anglais)
🔍 Détection des tableaux...
✅ 1 tableau(x) trouvé(s)

📊 Tableau 1: 14 lignes × 5 colonnes
💾 CSV: output/ma_photo_table_1.csv
💾 Excel: output/ma_photo_table_1.xlsx
📋 JSON: output/ma_photo_extraction.json
```

## Structure du projet

```
img2table_test_project/
├── simple_extractor.py   # Script principal
├── requirements.txt      # Dépendances
├── sample_data/          # Images de test
└── output/               # Résultats
```

## Dépendances

- img2table
- easyocr
- pandas
- openpyxl
