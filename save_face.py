import cv2
import os
import shutil


class FaceRecognitionPipeline:
    def __init__(self, max_faces=300, base_dir="./data", username="dung"):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.max_faces = max_faces
        self.face_count = 0
        self.username = username

        # Thư mục lưu ảnh: mỗi người 1 folder
        self.save_dir = os.path.join(base_dir, self.username)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"[INFO] Saving faces to: {self.save_dir}")

    def run(self):
        cap = cv2.VideoCapture(0)
        scanning = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            annotated = frame.copy()

            if scanning and len(faces) > 1:
                print("[ERROR] Too many faces! Cancelling scan.")
                # shutil.rmtree(self.save_dir)

            if not scanning and len(faces) == 1:
                scanning = True
                print("[INFO] Scanning started...")

            for x, y, w, h in faces:
                face_crop = frame[y : y + h, x : x + w]
                filename = os.path.join(
                    self.save_dir, f"face_{self.face_count + 1}.jpg"
                )
                cv2.imwrite(filename, face_crop)
                print(f"[INFO] Saved: {filename}")
                self.face_count += 1
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if self.face_count >= self.max_faces:
                print("[INFO] Collected enough faces. Exiting...")
                break

            cv2.imshow("Face Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


username = input("Nhập tên người dùng cần thu thập: ")
app = FaceRecognitionPipeline(max_faces=300, username=username)
app.run()
