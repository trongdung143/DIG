import cv2
import numpy as np


def predict_face(img, threshold, model, class_names):
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
