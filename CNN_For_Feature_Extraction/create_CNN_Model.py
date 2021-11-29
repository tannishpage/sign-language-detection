import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout


def create_cnn_model(image_data_shape, include_fc1_layer, num_classes):
    # Creating VGG-16 from scratch
    model = Sequential()
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=image_data_shape))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    return model

"""def create_cnn_model(image_data_shape, include_fc1_layer):
    if include_fc1_layer:
        orig_model = VGG16(include_top=True, input_shape=image_data_shape)

        # remove last layers, keeping only the layers till fc1
        orig_model.layers.pop()
        orig_model.layers.pop()
        cnn_model = Model(orig_model.input, orig_model.layers[-1].output)
    else:
        cnn_model = VGG16(weights='imagenet', include_top=False, input_shape=image_data_shape)

    print('Convolutional base:')
    print(cnn_model.summary())
    #plot_model(cnn_model, to_file='CNN.png', show_shapes=True, show_layer_names=True)
    return cnn_model
"""

if __name__ == "__main__":
    cnn_model = create_cnn_model((224, 224, 3), True, 11)
