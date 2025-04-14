import os
import numpy as np
from keras.models import load_model
from keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import array_to_img
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import cv2

# 1. Paths to PE folders
malware_folder = "/home/garvitagarwal/Desktop/marauderMap/marauderMapDataset/files/ransomware/"
benign_folder = "/home/garvitagarwal/Desktop/dike/DikeDataset/files/benign/"

# 2. Helper: Convert raw PE binary to image
def pe_to_image(filepath, image_size=(256, 256)):
    with open(filepath, 'rb') as f:
        content = f.read()
    byte_array = np.frombuffer(content, dtype=np.uint8)

    # Reshape to square or padded 1D image
    width = int(np.ceil(np.sqrt(len(byte_array))))
    padded = np.pad(byte_array, (0, width * width - len(byte_array)), 'constant')
    image = padded.reshape((width, width))

    # Resize to fixed shape and convert to 3 channels
    resized = cv2.resize(image, image_size)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb_image

# 3. Load all PE files and convert
def load_pe_images(folder, label):
    data = []
    labels = []
    for filename in tqdm(os.listdir(folder)):
        path = os.path.join(folder, filename)
        try:
            image = pe_to_image(path)
            data.append(image)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return np.array(data), np.array(labels)

# 4. Load malware and benign samples
malware_data, malware_labels = load_pe_images(malware_folder, label=1)
benign_data, benign_labels = load_pe_images(benign_folder, label=0)

# 5. Combine and preprocess
X = np.concatenate((malware_data, benign_data), axis=0)
y = np.concatenate((malware_labels, benign_labels), axis=0)
X = preprocess_input(X)

# 6. Load model and evaluate
model = load_model("drive/MyDrive/model_1_86.h5")
loss, accuracy = model.evaluate(X, y, batch_size=8)
print(f"PE File Dataset Accuracy: {accuracy * 100:.2f}%")

# 7. Predict and show accuracy
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype("int32").flatten()
print("Prediction Accuracy (sklearn):", accuracy_score(y, predicted_classes))
