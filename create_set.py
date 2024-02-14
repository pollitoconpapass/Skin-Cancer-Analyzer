import os
import pickle
from PIL import Image
import numpy as np


DATA_DIR = './data'  # Replace with the actual path to your skin condition images folder


# --- FLOW PROCESS ---
data = []
labels = []

for condition_label in os.listdir(DATA_DIR):
    condition_path = os.path.join(DATA_DIR, condition_label)

    if os.path.isdir(condition_path):
        for img_file in os.listdir(condition_path):
            img_path = os.path.join(condition_path, img_file)

            img = Image.open(img_path)
            img = img.resize((224, 224))

            img_array = np.array(img)
            img_flattened = img_array.flatten()

            data.append(img_flattened)
            labels.append(condition_label)


# --- SAVE THE DATA ---
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
