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
