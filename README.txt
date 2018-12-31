Emotion Recognition Model using the fer2013 dataset, built with Keras and OpenCV, with a 70% accuracy
*Data processing files borrowed from https://github.com/oarriaga/face_classification
*MTCNN package borrowed from https://github.com/ipazc/mtcnn/tree/master/mtcnn

To run the program, open command line and type:
> python3 emotion_color_demo.py

To run the program with the MTCNN face detection neural network instead of OpenCV's Haar Feature Classifer, run: 
> python3 mtcnn_demo.py

To train your own model, run: 
> python3 train_emotion_classifer.py