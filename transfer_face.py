import tensorflow as tf
from tensorflow import keras
from keras import layers, models, regularizers, applications
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import numpy as np

class TransferLearning:
    def __init__(self):
        self.data_dir = "./data/face_data"
        self.input_shape = (224, 224, 3)
        self.img_size = (224, 224)
        self.batch_size = 16
        self.epochs = 20
        self.num_classes = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.load_data()

    def build_model(self):
        base_model = applications.EfficientNetB0(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )
        
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        
        return model

    def load_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.15,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
            shear_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.15
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )
        
        self.test_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=1,
            class_mode="categorical",
            subset="validation",
            shuffle=False
        )

        self.num_classes = self.train_generator.num_classes
        
        class_indices = self.train_generator.class_indices
        with open("class_indices.txt", "w", encoding="utf-8") as f:
            for class_name, idx in class_indices.items():
                f.write(f"{class_name}: {idx}\n")

    def transfer_learning(self):
        self.__init__()
        model = self.build_model()
        model.summary()
        
        callbacks = [
            ModelCheckpoint(
                "./models/face_recognition.keras",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]

        initial_learning_rate = 1e-4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=len(self.train_generator),
            decay_rate=0.9,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        history = model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            class_weight=self.generate_class_weights(),
            workers=4,
            use_multiprocessing=True
        )
        
        num_classes = self.num_classes
        with open("num_classes.txt", "w", encoding="utf-8") as file:
            file.write(str(num_classes))
        
        self.evaluate_model(model)
        
        return model, history
    
    def generate_class_weights(self):
        class_counts = np.bincount(self.train_generator.classes)
        total_samples = float(sum(class_counts))
        class_weights = {i: total_samples / (self.num_classes * count) 
                         for i, count in enumerate(class_counts) if count > 0}
        return class_weights
    
    def evaluate_model(self, model):
        results = model.evaluate(self.test_generator)
        
        with open("model_evaluation.txt", "w", encoding="utf-8") as f:
            metrics = ["Loss", "Accuracy", "Precision", "Recall"]
            for metric, value in zip(metrics, results):
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"Model evaluation: {results}")
        
        y_true = self.test_generator.classes
        steps = len(self.test_generator)
        
        predictions = model.predict(self.test_generator, steps=steps)
        y_pred = np.argmax(predictions, axis=1)
        
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
            
            np.savetxt("confusion_matrix.txt", cm, fmt="%d")
            
            print("Đánh giá hoàn tất và kết quả đã được lưu.")
        except Exception as e:
            print(f"Lỗi khi tạo confusion matrix: {str(e)}")


if __name__ == "__main__":
    run = TransferLearning()
    model, history = run.transfer_learning()
