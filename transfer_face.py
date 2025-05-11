import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2


class TransferLearning:
    def __init__(self):
        self.data_dir = "./data/face_data"
        self.input_shape = (224, 224, 3)
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 5
        self.num_classes = None
        self.train_generator = None
        self.val_generator = None
        self.load_data()

    def build_model(self):
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet",
        )
        base_model.trainable = False

        model = models.Sequential(
            [
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        return model

    def load_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        self.val_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=True,
        )

        self.num_classes = self.train_generator.num_classes

    def transfer_learning(self):
        self.__init__()
        model = self.build_model()

        model.summary()

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "./models/face_recognition.keras",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        history = model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            callbacks=[checkpoint],
        )
