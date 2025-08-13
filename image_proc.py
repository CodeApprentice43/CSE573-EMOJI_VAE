from PIL import Image
import os

input_dir = "./original_data"
output_dir = "./data/training_data/emojis"
target_size = (128, 128)

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith('.png'):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        img = Image.open(input_path).convert('L')
        img = img.resize(target_size, Image.LANCZOS)
        img.save(output_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("Images converted to greyscale and resized to 128x128 successfully")