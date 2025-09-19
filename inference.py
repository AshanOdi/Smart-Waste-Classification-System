# smart_waste_webcam.py

import torch
from torchvision import transforms
from PIL import Image
import cv2
import csv
from collections import defaultdict
from model import model  # Replace with your actual model class

# ---------------- Config ----------------
MODEL_PATH = "saved_model.pth"  # Path to your saved model
LOG_FILE = "waste_log.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
category_count = defaultdict(int)

# ---------------- Load Model ----------------
model = YourModelClass()  # Replace with your model class
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- Preprocessing ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust size as per your model
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------- Create Log File if Not Exists ----------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame_Number", "Predicted_Class"])

# ---------------- Start Webcam ----------------
cap = cv2.VideoCapture(0)  # 0 for default webcam
frame_number = 0

print("[INFO] Starting live waste classification... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from webcam.")
        break

    frame_number += 1

    # Convert OpenCV image (BGR) to PIL image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_class = CLASS_NAMES[predicted.item()]
    category_count[predicted_class] += 1

    # Log the result
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([frame_number, predicted_class])

    # Overlay prediction on video frame
    cv2.putText(frame, f"Class: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Counts: {dict(category_count)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Live Waste Classification", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- Release Resources ----------------
cap.release()
cv2.destroyAllWindows()

# ---------------- Final Summary ----------------
print("\n--- Final Classification Summary ---")
for category, count in category_count.items():
    print(f"{category}: {count}")
print(f"\nResults logged in {LOG_FILE}")
