import cv2
import numpy as np
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array

def detect_and_align_face(img, face_cascade):
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=10)
    
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.03, 
            minNeighbors=2,
            minSize=(20, 20),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None, None
    
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    padding_factor = 0.4
    padding_x = int(w * padding_factor)
    padding_y = int(h * padding_factor)
    
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(img.shape[1], x + w + padding_x)
    y2 = min(img.shape[0], y + h + padding_y)
    
    face_img = img_enhanced[y1:y2, x1:x2]
    if face_img.size == 0:
        return None, None
    
    face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
    
    try:
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        face_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    except:
        pass
    
    aligned_face = cv2.resize(face_img, (224, 224))
    
    return aligned_face, largest_face

def histogram_equalization(img):
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        print(f"Lỗi cân bằng histogram: {e}")
        return img

def predict_face(img, threshold, model, class_names, face_cascade=None):
    if threshold < 0.2:
        threshold = 0.5
    
    variants = []
    
    img_resized = cv2.resize(img, (224, 224))
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_resized
    
    variants.append(img_rgb)
    
    brightened = cv2.convertScaleAbs(img_rgb, alpha=1.3, beta=15)
    variants.append(brightened)
    
    try:
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        variants.append(enhanced)
    except:
        pass
    
    try:
        contrast_enhanced = cv2.convertScaleAbs(img_rgb, alpha=1.5, beta=0)
        variants.append(contrast_enhanced)
    except:
        pass
    
    all_preds = []
    for variant in variants:
        norm_variant = variant / 255.0
        expanded = np.expand_dims(norm_variant, axis=0)
        preds = model.predict(expanded, verbose=0)
        all_preds.append(preds[0])
    
    avg_preds = np.mean(all_preds, axis=0)
    pred_class = np.argmax(avg_preds)
    confidence = float(np.max(avg_preds))
    
    if confidence < threshold:
        return "Unknown", confidence
    
    if pred_class < 0 or pred_class >= len(class_names):
        return "Unknown", confidence
    
    return class_names[pred_class], confidence