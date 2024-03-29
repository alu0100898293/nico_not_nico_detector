# Standard import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import matplotlib.pyplot as plt

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# Dataset info
train_dir = os.path.join('datasets', 'dataset_train')
validation_dir = os.path.join('datasets', 'dataset_validation')

BATCH_SIZE = 32
IMG_SIZE = (128, 128)

train_ds = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = train_ds.class_names

# Creating test dataset
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)


print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# Data augmentation

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])


#  Build The Model

# Preprocess to rescale the pixel values from [0,255] to [-1,1]
# MobileNet
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
# ResNet50
#preprocess_input = tf.keras.applications.resnet50.preprocess_input
# ResNet152
#preprocess_input = tf.keras.applications.resnet.preprocess_input
# Xception
#preprocess_input = tf.keras.applications.xception.preprocess_input
# Vgg16
#preprocess_input = tf.keras.applications.vgg16.preprocess_input


# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)

# MobileNet
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
# ResNet50
# base_model = tf.keras.applications.resnet50.ResNet50(...)
# ResNet152
# base_model = tf.keras.applications.resnet.ResNet152(...)
# Xception
# base_model = tf.keras.applications.xception.Xception(...)
# Vgg16  
# base_model = tf.keras.applications.vgg16.VGG16(...)                                             

image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)

# Set layers to non-trainable
base_model.trainable = False

# Add custom layers on top of MobileNet
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(128, 128, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
# Edit name to the right model
face_classifier = tf.keras.Model(inputs, outputs, name='MobileNet')


#  Training

name_to_save = f"models/face_classifier_{face_classifier.name}.h5"

base_learning_rate = 0.0001
face_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                      keras.metrics.BinaryAccuracy()])

face_classifier.summary()

# ModelCheckpoint to save model in case of interrupting the learning process
checkpoint = ModelCheckpoint(name_to_save,
                             monitor="val_binary_crossentropy",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

# EarlyStopping to find best model with a large number of epochs
earlystop = EarlyStopping(monitor='val_binary_crossentropy',
                          restore_best_weights=True,
                          patience=10,  # number of epochs with no improvement after which training will be stopped
                          verbose=1)

# Enabling tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


callbacks = [checkpoint, earlystop, tensorboard]

# Base training

initial_epochs = 300

history = face_classifier.fit(
    train_ds,
    verbose=2,
    epochs=initial_epochs,
    callbacks=callbacks,
    validation_data=val_ds)

# SAVING THE MODEL
face_classifier.save(name_to_save)

# TESTING THE MODEL
loss, crossentropy, accuracy = face_classifier.evaluate(test_ds)
print('Test loss :', loss)
print('Test crossentropy :', crossentropy)
print('Test accuracy :', accuracy)

# Retrieve a batch of images from the test set
image_batch, label_batch = test_ds.as_numpy_iterator().next()
predictions = face_classifier.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(16):
  ax = plt.subplot(4, 4, i + 1)
  plt.imshow(image_batch[i].astype("uint8"))
  plt.title(class_names[predictions[i]])
  plt.axis("off")
plt.show()