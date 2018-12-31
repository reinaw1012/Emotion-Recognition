import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import TensorBoard
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from utils.datasets import DataManager
from utils.datasets import split_data
from utils.preprocessor import preprocess_input

# parameters
batch_size = 32
num_epochs = 500
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 50
base_path = '../trained_models/emotion_models/'
emotion_model_path = '../trained_models/emotion_models/fer2013_my_newCNN.288-0.62.hdf5'
input_shape = (64, 64, 1)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as session:
    # data generator
    data_generator = ImageDataGenerator(
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=.1,
                            horizontal_flip=True)

    #Load in data
    dataset = 'fer2013'
    data_loader = DataManager(dataset, image_size=input_shape[:2])
    faces, emotions = data_loader.get_data()
    faces = preprocess_input(faces)
    num_samples, num_classes = emotions.shape
    train_data, val_data = split_data(faces, emotions, 0.2)
    train_faces, train_emotions = train_data

    print("Train_faces shape:", train_faces.shape)
    print("Train_emotions shape:", train_emotions.shape)

    #Load model
    emotion_classifier = tf.keras.models.load_model(emotion_model_path, compile=False)

    #Generate hard data
    faces = []
    emotions = []
    #ntrain_faces = train_faces[:50]
    #ntrain_emotions = train_emotions[:50]
    for i in range(len(train_faces)): #For each face
        face = train_faces[i]
        expand_face = np.expand_dims(face,0)
        emotion = train_emotions[i]
        emotionA = np.argmax(emotion)
        emotion_prediction = emotion_classifier.predict(expand_face)
        emotion_label = np.argmax(emotion_prediction)
        print("Correct emotion: ",emotionA)
        print("Prediction: ",emotion_label)
        if emotion_label != emotionA: #Save false predictions
            faces.append(face)
            emotions.append(emotion)
    faces = np.asarray(faces)
    emotions = np.asarray(emotions)
    print("Face length: ",len(faces))
    print("Emotion length: ",len(emotions))
    print("Done generating data.")
    print("Training:")

    # callbacks
    log_file_path = base_path + dataset + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping('val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,patience=int(patience/4), verbose=1)
    tensorboard = TensorBoard(log_dir='./tf_graph', histogram_freq=0,write_graph=True, write_images=True)
    trained_models_path = base_path + dataset + '_my_newCNN'
    model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr,tensorboard]

    #Training hard data
    set_session(session)
    session.run(tf.global_variables_initializer())
    emotion_classifier.summary()
    emotion_classifier.fit_generator(data_generator.flow(faces, emotions,batch_size),
                        steps_per_epoch=len(faces) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=val_data)
    
    