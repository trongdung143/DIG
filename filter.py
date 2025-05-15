import cv2
import numpy as np

class PreprocessingPipeline:
    def __init__(self, sharpness_threshold=100, saturation_factor=1.2):
        self.sharpness_threshold = sharpness_threshold
        self.saturation_factor = saturation_factor
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    def adjust_brightness_contrast(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def check_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def sharpen_image(self, image):
        gaussian = cv2.GaussianBlur(image, (0, 0), 3.0)
        return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    def remove_noise(self, image):
        return cv2.bilateralFilter(image, 9, 75, 75)

    def adjust_saturation(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = hsv_image[..., 1] * self.saturation_factor
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def align_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return image
        
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        face_gray = gray[y:y+h, x:x+w]
        
        face_gray = cv2.equalizeHist(face_gray)
        
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(10, 10)
        )
        
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])
            
            left_eye = eyes[0]
            right_eye = eyes[1]
            
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
            
            dx = right_eye_center[0] - left_eye_center[0]
            dy = right_eye_center[1] - left_eye_center[1]
            
            if dx > 0:  
                angle = np.degrees(np.arctan2(dy, dx))
                
                center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                        (left_eye_center[1] + right_eye_center[1]) // 2)
                
                M = cv2.getRotationMatrix2D(center, angle, 1)
                aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                                            flags=cv2.INTER_CUBIC)
                return aligned_image
        
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.3, beta=10)
        return enhanced_image

    def letterbox_image(self, image, target_size, color):
        ih, iw = image.shape[:2]
        h, w = target_size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image_resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        new_image = np.full((h, w, 3), color, dtype=np.uint8)

        top = (h - nh) // 2
        left = (w - nw) // 2

        new_image[top : top + nh, left : left + nw] = image_resized

        return new_image

    def process_frame(self, frame, target_size=(480, 640), color=(0, 0, 0)):
        try:
            aligned_frame = self.align_face(frame)
            
            frame = self.adjust_brightness_contrast(aligned_frame)
            frame = self.remove_noise(frame)
            
            if self.check_sharpness(frame) < self.sharpness_threshold:
                frame = self.sharpen_image(frame)
                
            frame = self.adjust_saturation(frame)
            
            frame = self.letterbox_image(frame, target_size, color=color)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized_frame = rgb_frame / 255.0
            
            input_tensor = np.transpose(normalized_frame, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = input_tensor.astype(np.float32)
            
            return frame, input_tensor
            
        except Exception as e:
            print(f"Lỗi xử lý frame: {str(e)}")
            frame = self.remove_noise(frame)
            frame = self.letterbox_image(frame, target_size, color=color)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            normalized_frame = rgb_frame / 255.0
            
            input_tensor = np.transpose(normalized_frame, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)
            input_tensor = input_tensor.astype(np.float32)
            
            return frame, input_tensor
