# Emotion Recognition
Emotion Recognition Model using the fer2013 dataset, built with Keras and OpenCV, with a 70% accuracy
* Data processing files borrowed from [here](https://github.com/oarriaga/face_classification)
* MTCNN package borrowed from [here](https://github.com/ipazc/mtcnn/tree/master/mtcnn)

To run the program, open command line and type:
> python3 emotion_color_demo.py

To run the program with the MTCNN face detection neural network instead of OpenCV's Haar Feature Classifer, run: 
> python3 mtcnn_demo.py

To train your own model, download the fer2013 package from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
Create a "datasets" folder and unzip the file under it: 
>tar -xzf fer2013.tar

Run the training program:
> python3 train_emotion_classifer.py
