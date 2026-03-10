import csv
import os
from pathlib import Path

base_dir = Path('/home/rafael/Projects/python/IFG-Computer_vision/cow-classifier/data/datasets/classifications/test')
classes_csv = Path('/home/rafael/Projects/python/IFG-Computer_vision/cow-classifier/data/datasets/classifications/classes.csv')

def get_first_image(cow_id):
    folder = base_dir / str(cow_id)
    if folder.exists() and folder.is_dir():
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                return str(file.absolute())
    return ''

new_rows = []
try:
    with open(classes_csv, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        if 'image_url' not in header:
            header.append('image_url')
        new_rows.append(header)
        
        for row in reader:
            class_name = row[0]
            first_img = get_first_image(class_name)
            if len(row) >= 3:
                row[2] = first_img
            else:
                row.append(first_img)
            new_rows.append(row[:3])

    with open(classes_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)
    print("CSV updated successfully with local image paths.")
except Exception as e:
    print(f"Error: {e}")
