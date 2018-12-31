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
emotion_model_path = '../trained_models/emotion_models/fer2013_my_smallerCNN.91-0.66.hdf5'
input_shape = (64, 64, 1)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
with tf.Session(config=config) as session:

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

    #Generate hard data
    faces = []
    emotions = []
    angry = 0
    disgust = 0
    fear = 0
    happy = 0
    sad = 0
    surprise = 0
    neutral = 0
    #train_faces = train_faces[:50]
    #train_emotions = train_emotions[:50]
    for i in range(len(train_faces)): #For each face
        face = train_faces[i]
        expand_face = np.expand_dims(face,0)
        emotion = train_emotions[i]
        emotionA = np.argmax(emotion)
        emotion_prediction = emotion_classifier.predict(expand_face)
        emotion_label = np.argmax(emotion_prediction)
        if emotion_label != emotionA: #Save false predictions
            print("Correct emotion: ",emotionA)
            print("Prediction: ",emotion_label)
            if emotion_label == 0:
                angry +=1
            if emotion_label == 1:
                disgust +=1
            if emotion_label == 2:
                fear +=1
            if emotion_label == 3:
                happy +=1
            if emotion_label == 4:
                sad +=1
            if emotion_label == 5:
                surprise +=1
            if emotion_label == 6:
                neutral +=1
    print("Miscounts:")
    print("Angry: ",angry)
    print("Disgust: ",disgust)
    print("Fear: ",fear)
    print("Happy: ",happy)
    print("Sad: ",sad)
    print("Surprise: ",surprise)
    print("Neutral: ",neutral)
    print("Done.")
    
    