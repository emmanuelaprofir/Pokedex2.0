import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pathlib
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


project_dir = pathlib.Path(
    "/Users/emmanuelaprofir/Documents/GitHub/pokedex/Pokedex2.0"
)

data_dir = project_dir / "dataset_pokemon"
image_path = project_dir / "img_a_tester" / "test.jpg"

model_dir = project_dir / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "pokemon_model.keras"
classes_path = model_dir / "class_names.json"


print(f"Dataset path: {data_dir.resolve()}")

batch_size = 8
img_height = 200
img_width = 200


train_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


class_names = train_data.class_names

print("Classes détectées :", class_names)

num_classes = len(class_names)


with open(classes_path, "w", encoding="utf-8") as f:
    json.dump(
        class_names,
        f,
        ensure_ascii=False,
        indent=4
    )

print(f"Classes sauvegardées : {classes_path}")


plt.figure(figsize=(10, 10))

for images, labels in train_data.take(1):
    for i in range(min(3, len(images))):
        ax = plt.subplot(1, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

plt.show()


model = tf.keras.Sequential([
    layers.Rescaling(
        1.0 / 255,
        input_shape=(img_height, img_width, 3)
    ),
    layers.Conv2D(
        128,
        4,
        activation="relu"
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        64,
        4,
        activation="relu"
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        32,
        4,
        activation="relu"
    ),
    layers.MaxPooling2D(),
    layers.Conv2D(
        16,
        4,
        activation="relu"
    ),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(
        64,
        activation="relu"
    ),
    layers.Dense(
        num_classes,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),
    metrics=["accuracy"]
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True
)


logdir = project_dir / "logs"

tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=str(logdir),
    histogram_freq=1
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[
        tensorboard_callback,
        early_stopping
    ]
)


model.save(model_path)

print(f"Modèle sauvegardé : {model_path}")
saved_model_dir = project_dir / "models" / "saved_model"

tf.saved_model.save(
    model,
    str(saved_model_dir)
)

print(f"SavedModel sauvegardé : {saved_model_dir}")


from tensorflow.keras.utils import load_img, img_to_array


if not image_path.exists():
    print(f"Image non trouvée : {image_path}")
    sys.exit(1)


image = load_img(
    image_path,
    target_size=(img_height, img_width)
)

image_array = img_to_array(image)

image_array = np.expand_dims(
    image_array,
    axis=0
)


plt.figure(figsize=(5, 5))
plt.imshow(image)
plt.axis("off")
plt.title("Image à prédire")
plt.show()


predictions = model.predict(image_array)

res = np.argmax(
    predictions,
    axis=1
)[0]

predicted_class = class_names[res]

confidence = predictions[0][res] * 100


print(f"Classe prédite : {predicted_class} ({res})")
print(f"Confiance : {confidence:.2f} %")
print("Probabilités :")

for i, probability in enumerate(predictions[0]):
    print(
        f"{class_names[i]} : "
        f"{probability * 100:.2f} %"
    )


def visualiser_filtres(
    name_image,
    model,
    layer_name,
    image
):

    inp = model.inputs

    out1 = model.get_layer(
        layer_name
    ).output

    feature_map_1 = Model(
        inputs=inp,
        outputs=out1
    )

    img_resized = cv2.resize(
        image,
        (img_width, img_height)
    )

    img_normalized = img_resized.astype(
        "float32"
    )

    input_img = np.expand_dims(
        img_normalized,
        axis=0
    )

    f = feature_map_1.predict(
        input_img
    )

    dim = f.shape[3]

    print(
        f"{layer_name} | Features Shape: {f.shape}"
    )

    print(
        f"Dimension: {dim}"
    )

    output_dir = project_dir / (
        f"results_{name_image}"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    columns = 8

    rows = int(
        np.ceil(dim / columns)
    )

    fig = plt.figure(
        figsize=(30, 30)
    )

    for i in range(dim):

        ax = fig.add_subplot(
            rows,
            columns,
            i + 1
        )

        ax.axis("off")

        ax.imshow(
            f[0, :, :, i],
            cmap="viridis"
        )

        output_file = (
            output_dir
            / f"{name_image}_{layer_name}_{i}.jpg"
        )

        plt.imsave(
            output_file,
            f[0, :, :, i],
            cmap="viridis"
        )

    plt.show()


print(f"Classe prédite : {class_names[res]}")
