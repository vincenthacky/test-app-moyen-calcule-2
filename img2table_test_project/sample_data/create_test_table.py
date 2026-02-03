#!/usr/bin/env python3
"""
Script pour créer un tableau de test (liste d'élèves + notes)
Génère une image PNG avec un tableau structuré
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_student_table():
    """Créer une image avec un tableau d'élèves et leurs notes"""

    # Dimensions
    width, height = 800, 600
    background_color = (255, 255, 255)  # Blanc

    # Créer l'image
    img = Image.new('RGB', (width, height), background_color)
    draw = ImageDraw.Draw(img)

    # Couleurs
    header_color = (70, 130, 180)  # Bleu
    text_color = (0, 0, 0)  # Noir
    border_color = (0, 0, 0)  # Noir

    # Police (utiliser police système)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()

    # Titre
    title = "RELEVÉ DE NOTES - CLASSE 3ème A"
    title_bbox = draw.textbbox((0, 0), title, font=font_large)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 30), title, font=font_large, fill=text_color)

    # Position du tableau
    table_x = 100
    table_y = 100
    col_width = 150
    row_height = 40

    # Données du tableau
    headers = ["Nom", "Prénom", "Math", "Français", "Moyenne"]
    students = [
        ["DUPONT", "Marie", "16.5", "14.0", "15.25"],
        ["MARTIN", "Pierre", "12.0", "16.5", "14.25"],
        ["BERNARD", "Sophie", "18.0", "15.5", "16.75"],
        ["MOREAU", "Lucas", "14.5", "13.0", "13.75"],
        ["SIMON", "Emma", "17.0", "18.0", "17.5"],
        ["DUBOIS", "Thomas", "11.5", "12.5", "12.0"],
        ["LAURENT", "Chloé", "15.5", "17.0", "16.25"],
        ["MICHEL", "Antoine", "13.0", "14.5", "13.75"]
    ]

    # Dessiner le tableau
    num_cols = len(headers)
    num_rows = len(students) + 1  # +1 pour l'en-tête

    # Calculer la largeur totale du tableau
    table_width = num_cols * col_width

    # Dessiner les lignes horizontales
    for i in range(num_rows + 1):
        y = table_y + i * row_height
        draw.line([(table_x, y), (table_x + table_width, y)], fill=border_color, width=2)

    # Dessiner les lignes verticales
    for i in range(num_cols + 1):
        x = table_x + i * col_width
        draw.line([(x, table_y), (x, table_y + num_rows * row_height)], fill=border_color, width=2)

    # Dessiner l'en-tête
    for i, header in enumerate(headers):
        x = table_x + i * col_width + 10
        y = table_y + 10

        # Fond coloré pour l'en-tête
        header_rect = [
            table_x + i * col_width + 2,
            table_y + 2,
            table_x + (i + 1) * col_width - 2,
            table_y + row_height - 2
        ]
        draw.rectangle(header_rect, fill=header_color)

        # Texte de l'en-tête
        draw.text((x, y), header, font=font_medium, fill=(255, 255, 255))

    # Dessiner les données
    for row_idx, student in enumerate(students):
        for col_idx, value in enumerate(student):
            x = table_x + col_idx * col_width + 10
            y = table_y + (row_idx + 1) * row_height + 10
            draw.text((x, y), value, font=font_medium, fill=text_color)

    # Ajouter une note en bas
    note = "Document généré automatiquement - École Notre-Dame"
    note_bbox = draw.textbbox((0, 0), note, font=font_medium)
    note_width = note_bbox[2] - note_bbox[0]
    note_x = (width - note_width) // 2
    draw.text((note_x, height - 50), note, font=font_medium, fill=(100, 100, 100))

    return img

def main():
    """Créer les fichiers de test"""
    print("🎨 Création d'un tableau de test...")

    # Créer l'image
    img = create_student_table()

    # Sauvegarder
    img_path = "table_notes.jpg"
    img.save(img_path, "JPEG", quality=95)

    print(f"✅ Image créée: {img_path}")
    print("📊 Tableau contenant:")
    print("  - En-têtes: Nom, Prénom, Math, Français, Moyenne")
    print("  - 8 lignes d'élèves avec leurs notes")
    print("  - Structure tabulaire claire avec bordures")

    # Informations pour l'utilisateur
    print("\n💡 Pour tester:")
    print("1. Placez ce fichier dans le dossier sample_data/")
    print("2. Lancez python main.py")
    print("3. Vérifiez les résultats dans output/")

if __name__ == "__main__":
    main()