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


#  Build The Model (MobileNet)

# Preprocess to rescale the pixel values from [0,255] to [-1,1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

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

initial_epochs_done = len(history.history['binary_crossentropy'])

# Store results for future comparison
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss = history.history['binary_crossentropy']
val_loss = history.history['val_binary_crossentropy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()



# FINE TUNING

base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Compile the model
face_classifier.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=[tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                      keras.metrics.BinaryAccuracy()])

# Continue training the model
fine_tune_epochs = 300
total_epochs =  initial_epochs_done + fine_tune_epochs

history_fine = face_classifier.fit(
                         train_ds,
                         verbose=2,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         callbacks=callbacks,
                         validation_data=val_ds)


# Compare before and after fine tuning

acc += history_fine.history['binary_accuracy']
val_acc += history_fine.history['val_binary_accuracy']

loss += history_fine.history['binary_crossentropy']
val_loss += history_fine.history['val_binary_crossentropy']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.plot([initial_epochs_done-1,initial_epochs_done-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.plot([initial_epochs_done-1,initial_epochs_done-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


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