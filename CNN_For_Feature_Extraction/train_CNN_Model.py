import tensorflow as tf
from create_CNN_Model import create_cnn_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
TRAIN_PATH = "/home/tannishpage/Documents/Sign_Language_Detection/senz3d_dataset/acquisitions"
#VAL_PATH = "/home/tannishpage/Documents/Sign_Language_Detection/archive/Validation"

callbacks = [ModelCheckpoint("/home/tannishpage/Documents/Sign_Language_Detection/feature_cnn_weights.h5", monitor='val_loss', save_best_only=True, verbose=1)]
BATCH_SIZE = 16
EPOCHS = 100
# Get Data
train_data = image_dataset_from_directory(TRAIN_PATH, image_size=(224, 224), seed=42, batch_size=BATCH_SIZE, label_mode='categorical', validation_split=0.2, subset='training')
val_data = image_dataset_from_directory(TRAIN_PATH, image_size=(224, 224), seed=42, batch_size=BATCH_SIZE, label_mode='categorical', validation_split=0.2, subset='validation')

class_names = train_data.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i].index(1)])
    plt.axis("off")
plt.show()
# Create and train model
model = create_cnn_model((224, 224, 3), True, 11)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=EPOCHS, callbacks=callbacks, verbose=1, validation_data = val_data)
