import tensorflow as tf
from tensorflow import keras

def my_CNN(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(5, kernel_size=(3, 3), padding='same',
                            name='image_array', input_shape=input_shape, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(10, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Convolution2D(filters=num_classes, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Activation('softmax',name='predictions'))
    return model

def my_newCNN(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                            name='image_array', input_shape=input_shape, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Activation('softmax',name='predictions'))
    return model

def my_smallerCNN(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                            name='image_array', input_shape=input_shape, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='same',
                            strides=(2, 2), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.25))
    model.add(keras.layers.Dense(128,activation='relu'))
    model.add(keras.layers.Dense(num_classes))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Activation('softmax',name='predictions'))
    return model

def my_FeatCNN(input_shape,classes):
    padding = 'valid'
    img_input = keras.layers.Input(shape=input_shape)

    # START MODEL
    conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2,2))(conv_1)
    x = tf.keras.layers.BatchNormalization()(maxpool_1)
    
    # FEAT-EX1
    conv_2a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_2a')(x)
    conv_2b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
    maxpool_2a = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2a')(x)
    conv_2c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1,1), name='conv_2c')(maxpool_2a)
    concat_1 = tf.keras.layers.concatenate(inputs=[conv_2b,conv_2c], axis=3,name='concat2')
    maxpool_2b = keras.layers.MaxPooling2D((3,3), strides=(2,2), padding=padding, name='maxpool_2b')(concat_1)

    # FEAT-EX2
    conv_3a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
    conv_3b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
    maxpool_3a = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3a')(maxpool_2b)
    conv_3c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_3a)
    concat_3 = tf.keras.layers.concatenate(inputs=[conv_3b,conv_3c],axis=3,name='concat3')
    maxpool_3b = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3b')(concat_3)
    
    # FINAL LAYERS
    net = tf.keras.layers.Flatten()(maxpool_3b)
    net = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(net)
    
    # Create model.
    model = tf.keras.Model(img_input, net, name='deXpression')
    return model

def my_newFeatCNN(input_shape,classes):
    padding = 'valid'
    img_input = keras.layers.Input(shape=(57,57,1))

    # START MODEL
    conv_1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
    maxpool_1 = keras.layers.MaxPooling2D((2, 2), strides=(2,2))(conv_1)
    x = tf.keras.layers.BatchNormalization()(maxpool_1)
    
    # FEAT-EX1
    conv_2a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_2a')(x)
    conv_2b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
    dropout_1 = keras.layers.Dropout(0.25, name='dropout_1')(conv_2b)
    maxpool_2a = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2a')(x)
    conv_2c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1,1), name='conv_2c')(maxpool_2a)
    dropout_2 = keras.layers.Dropout(0.25, name='dropout_2')(conv_2c)
    concat_1 = tf.keras.layers.concatenate(inputs=[dropout_1, dropout_2], axis=3,name='concat2')
    maxpool_2b = keras.layers.MaxPooling2D((3,3), strides=(2,2), padding=padding, name='maxpool_2b')(concat_1)

    # FEAT-EX2
    conv_3a = tf.keras.layers.Conv2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
    conv_3b = tf.keras.layers.Conv2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
    dropout_3 = keras.layers.Dropout(0.25, name='dropout_3')(conv_3b)
    maxpool_3a = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3a')(maxpool_2b)
    conv_3c = tf.keras.layers.Conv2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_3a)
    dropout_4 = keras.layers.Dropout(0.25, name='dropout_4')(conv_3c)
    concat_3 = tf.keras.layers.concatenate(inputs=[dropout_3, dropout_4],axis=3,name='concat3')
    maxpool_3b = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3b')(concat_3)
    
    # FINAL LAYERS
    net = tf.keras.layers.Flatten()(maxpool_3b)
    net = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(net)
    
    # Create model.
    model = tf.keras.Model(img_input, net, name='deXpression')
    return model

if __name__ == "__main__":
    input_shape = (64, 64, 1)
    num_classes = 7
    model = my_CNN(input_shape, num_classes)
    model.summary()
