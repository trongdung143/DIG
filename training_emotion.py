from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks
import tensorflow as tf

data_dir = "./data/emotion_data/train"
img_size = (48, 48)
batch_size = 64
epochs = 20

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    color_mode="grayscale",
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    color_mode="grayscale",
)


inputs = layers.Input(shape=(48, 48, 1))

x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(
    512,
    (3, 3),
    padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(0.01),
)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Conv2D(
    512,
    (3, 3),
    padding="same",
    activation="relu",
    kernel_regularizer=regularizers.l2(0.01),
)(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Dropout(0.25)(x)

x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

x = layers.Dense(512, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.25)(x)

outputs = layers.Dense(7, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)


early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    "face_emotion.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[
        checkpoint,
    ],
)
