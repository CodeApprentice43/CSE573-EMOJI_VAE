from PIL import Image
import os

input_dir = "./original_data"
output_dir = "./training_data/emojis"
target_size = (64, 64)

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith('.png'):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        img = Image.open(input_path).resize(target_size)
        img.save(output_path)
    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("RGB images resized successfully")
