import os
try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Install it with: python -m pip install pillow")
    raise
import numpy as np
import pandas as pd

# Specify the path to your folders (adjust this based on your folder structure)
base_dir = r"C:\Users\USER\Desktop\450data\PlantVillage"

folders = classes = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted"
]


# Initialize list for storing data
data = []

# If base_dir doesn't exist, raise a helpful error
if not os.path.isdir(base_dir):
    raise FileNotFoundError(f"Base directory not found: {base_dir}")

# Use actual subfolders found in base_dir (ignore the hardcoded list)
folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
if not folders:
    print(f"No subfolders found in {base_dir}")

# Loop through each actual folder and track empty ones
empty_folders = []
for folder in folders:
    folder_path = os.path.join(base_dir, folder)

    # List to store dimensions of all images in the folder
    widths = []
    heights = []

    # Only consider common image extensions
    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            continue
        image_path = os.path.join(folder_path, image_name)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)
        except Exception:
            # skip unreadable files
            print(f"Skipped unreadable file: {image_path}")

    # Calculate averages and standard deviations (handle empty case)
    freq = len(widths)
    if freq > 0:
        width_avg = float(np.mean(widths))
        height_avg = float(np.mean(heights))
        width_std = float(np.std(widths))
        height_std = float(np.std(heights))
    else:
        width_avg = height_avg = width_std = height_std = float('nan')

    # Append the result for this folder
    data.append([folder, freq, width_avg, width_std, height_avg, height_std])
    if freq == 0:
        empty_folders.append(folder)

# Create a DataFrame to display the result in a tabular format
df = pd.DataFrame(data, columns=["Category", "Freq", "Width_avg", "Width_std", "Height_avg", "Height_std"])

# Save the result to a CSV file or print it
out_path = os.path.join(base_dir, "image_dimensions.csv")
df.to_csv(out_path, index=False)
print(f"Wrote summary to: {out_path}")
print(df)
if empty_folders:
    print("Note: these folders had no readable images:", empty_folders)