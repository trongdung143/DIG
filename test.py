import cv2
import numpy as np
import tensorflow as tf
import os

# Load mô hình đã huấn luyện
model = tf.keras.models.load_model("face_recognition_mobilenetv2.h5")
print("[INFO] Model đã được load thành công!")

# Danh sách các lớp (class). Bạn nên sửa đúng theo số lượng người bạn đã huấn luyện
class_names = ["HuyHoang", "MaiAnh", "TrongDung"]  # Ví dụ: bạn có 3 người
print("[INFO] Class names:", class_names)

# Load bộ nhận diện khuôn mặt (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Hàm dự đoán khuôn mặt
def predict_face(img, threshold=0.7):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_expanded = np.expand_dims(img_resized, axis=0)

    preds = model.predict(img_expanded, verbose=0)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)

    if confidence < threshold:
        return "Unknown", confidence
    return class_names[pred_class], confidence


# Mở webcam
cap = cv2.VideoCapture(0)
print("[INFO] Đang bật webcam... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    # Duyệt qua từng khuôn mặt
    for x, y, w, h in faces:
        face_img = frame[y : y + h, x : x + w]
        label, confidence = predict_face(face_img)

        # Vẽ khung và label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(
            frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

    cv2.imshow("Face Recognition", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
