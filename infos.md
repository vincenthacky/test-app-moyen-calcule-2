# 🧪 Prompt Claude — Création d’un Mini-Projet Python avec img2table (Test Réel d’Extraction de Tableaux)

## 🎯 Objectif
Tu dois CRÉER un mini-projet Python fonctionnel qui utilise la bibliothèque open-source **img2table** pour :
- Prendre une image ou un PDF contenant un tableau (liste d’élèves + notes)
- Extraire les cellules du tableau
- Générer un **pandas DataFrame**
- Exporter les résultats en :
  - JSON
  - CSV
- Me montrer exactement **ce que le package extrait, comment il l’extrait et sous quelle forme**

Je veux pouvoir tester ce projet localement et évaluer si `img2table` est adapté à une plateforme éducative de gestion de notes.

---

## 📁 Partie 1 — Structure du Projet

Tu dois générer une structure claire comme ceci :
img2table_test_project/
├── README.md
├── requirements.txt
├── main.py
├── sample_data/
│ ├── table_notes.jpg
│ └── table_notes.pdf
├── output/
│ ├── result.json
│ └── result.csv















//nouvelle ligne


Si je devais construire un système robuste, voici ma méthode étape par étape :
1️⃣ Prétraitement des images / PDF
Utiliser OpenCV pour :
Redresser les images (correction de perspective)
Améliorer contraste et luminosité
Binarisation pour mieux faire ressortir le texte
Convertir les PDF en images avec pdf2image
Pourquoi : un OCR fonctionne beaucoup mieux si l’image est nette et contrastée.
2️⃣ Détection de structure (tables, lignes, colonnes)
Utiliser LayoutParser ou OpenCV pour détecter :
Lignes horizontales → séparer les rangées d’élèves
Lignes verticales → séparer les colonnes matières/notes
Détecter automatiquement le “bloc tableau” sur l’image
Cela permet à l’OCR de lire cellule par cellule, ce qui augmente la précision.
3️⃣ Reconnaissance de texte
Tesseract → pour texte imprimé
Kraken ou TrOCR → pour manuscrit
Faire un OCR cellule par cellule plutôt que toute l’image d’un coup
Cela réduit énormément les erreurs et permet de relier correctement le texte à la colonne.
4️⃣ Nettoyage et normalisation
Utiliser Pandas + Regex pour :
Supprimer espaces inutiles
Corriger les caractères mal reconnus
Convertir les notes en chiffres (int/float)
Vérifier les valeurs incohérentes (ex : note > 20 ou < 0)
5️⃣ Conversion en DataFrame / JSON / CSV
