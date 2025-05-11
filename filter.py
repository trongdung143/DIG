import cv2
import numpy as np
import cv2
import numpy as np


class PreprocessingPipeline:
    def __init__(self, sharpness_threshold=100, saturation_factor=1.2):
        self.sharpness_threshold = sharpness_threshold
        self.saturation_factor = saturation_factor

    def adjust_brightness_contrast(self, image):
        b, g, r = cv2.split(image)

        b = cv2.equalizeHist(b)
        g = cv2.equalizeHist(g)
        r = cv2.equalizeHist(r)

        return cv2.merge([b, g, r])

    def check_sharpness(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def sharpen_image(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def remove_noise(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def adjust_saturation(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[..., 1] = hsv_image[..., 1] * self.saturation_factor
        hsv_image[..., 1] = np.clip(hsv_image[..., 1], 0, 255)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

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
        # Áp dụng các bước tiền xử lý trên ảnh

        # frame = self.adjust_brightness_contrast(frame)
        frame = self.remove_noise(frame)
        if self.check_sharpness(frame) < self.sharpness_threshold:
            frame = self.sharpen_image(frame)
        frame = self.adjust_saturation(frame)

        frame = self.letterbox_image(frame, target_size, color=color)

        # Chuyển đổi ảnh từ BGR sang RGB và chuẩn hóa giá trị pixel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        normalized_frame = rgb_frame / 255.0

        # Chuyển đổi ảnh thành tensor theo định dạng (batch_size, channels, height, width)
        input_tensor = np.transpose(normalized_frame, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = input_tensor.astype(np.float32)

        return frame, input_tensor
