import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
training_dir = pathlib.Path("training/")
# distorted_test_dir = pathlib.Path("DisCaptcha_v0/_test")

# categories = ["airplane", "car", "cat", "dog", "flower", "fruit", "motorbike", "person"]
# airplanes = list(archive_test_dir.glob('airplane/*'))
classes = ["distorted", "original"]
original = []
distorted = []

image_count = len(list(training_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  training_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  training_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

# for category in categories:
#     original = original + list(archive_test_dir.glob(category + '/*'))

# distorted_categories = ["airplane", "cat"]
# for category in distorted_categories:
#     distorted = distorted + list(distorted_test_dir.glob(category + '/*'))
# PIL.Image.open(str(original[0])).show()
# PIL.Image.open(str(distorted[0])).show()

batch_size = 32
img_height = 180
img_width = 180

# print(str(original[0]))

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)