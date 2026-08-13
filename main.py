import cv2
import numpy as np

try:
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import (
        preprocess_input,
        decode_predictions
    )
except ImportError:
    print("TensorFlow is not installed.")
    exit()

# Load model
print("Loading MobileNetV2...")
model = MobileNetV2(weights="imagenet")

# Read image
image_path = "WIN_20250814_10_28_09_Pro.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Image not found: {image_path}")
    exit()

# Resize image
img = cv2.resize(image, (224, 224))

# Prepare image
x = np.expand_dims(img, axis=0)
x = preprocess_input(x)

# Prediction
predictions = model.predict(x)

print("\nTop 3 Predictions:")
for pred in decode_predictions(predictions, top=3)[0]:
    print(f"{pred[1]} : {pred[2] * 100:.2f}%")

# Brightness detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
brightness = np.mean(gray)

print("\nAmbient Brightness:", brightness)

# Headlight decision
if brightness < 80:
    print("Headlights ON")
else:
    print("Headlights OFF")