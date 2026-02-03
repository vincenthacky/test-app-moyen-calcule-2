# 🧪 Projet Test img2table - Extraction de Tableaux

Ce mini-projet teste la bibliothèque **img2table** pour extraire des tableaux à partir d'images et de PDFs, puis les convertir en pandas DataFrames et les exporter en JSON/CSV.

## 📁 Structure du projet

```
img2table_test_project/
├── README.md               # Ce fichier
├── requirements.txt        # Dépendances Python
├── main.py                # Script principal
├── sample_data/           # Données de test
│   └── ma_photo.jpeg      # Image test avec tableau
├── output/                # Résultats d'extraction
│   ├── result.json        # Export JSON
│   └── result.csv         # Export CSV
└── venv/                  # Environnement virtuel
```

## 🚀 Installation et utilisation

### 1. Créer l'environnement virtuel
```bash
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'extraction
```bash
python main.py
```

## 📊 Ce que fait le script

1. **Détection automatique** des fichiers image (.jpg, .png) et PDF dans `sample_data/`
2. **Extraction des tableaux** avec img2table
3. **Conversion** en pandas DataFrame
4. **Analyse** de la structure détectée (position, dimensions)
5. **Export** en JSON et CSV dans le dossier `output/`

## 🔍 Résultats détaillés

Le script affiche :
- Nombre de tableaux détectés
- Position et dimensions de chaque tableau
- Aperçu des données extraites
- Informations sur la conversion DataFrame

## 📝 Formats d'export

### JSON
```json
{
  "extraction_summary": {
    "total_tables": 1,
    "source_type": "image"
  },
  "tables": [
    {
      "table_id": 1,
      "bbox": {"x1": 100, "y1": 150, "x2": 700, "y2": 450},
      "shape": {"rows": 8, "cols": 5},
      "data": [...]
    }
  ]
}
```

### CSV
Données combinées de tous les tableaux en format tabulaire.

## 🎯 Objectif d'évaluation

Ce projet permet d'évaluer si **img2table** convient pour une plateforme éducative de gestion de notes en testant :

- ✅ Précision de détection des tableaux
- ✅ Qualité de l'extraction des cellules
- ✅ Facilité d'intégration avec pandas
- ✅ Format des données extraites
- ✅ Performance sur différents types d'images

## 🔧 Dépendances principales

- **img2table** : Extraction de tableaux
- **pandas** : Manipulation des données
- **opencv-python** : Traitement d'images
- **PyMuPDF** : Support PDF
- **Pillow** : Gestion d'images

## ⚠️ Notes techniques

- OCR Tesseract est optionnel (fonctionne sans)
- Le script traite la première page des PDFs
- Les tableaux sans bordures sont supportés
- Format de sortie compatible avec les workflows pandas