Requirements
-	Anaconda3
Setting
-	Open environment.yml
-	Customize name (first row) and prefix(last row) based on your preference and anaconda installation path
-	Run command: conda env create environment.yml
-	Download the model weights from  https://drive.google.com/drive/folders/1ptjh3uOtRFbzLtVJr6FXpJNq3lJlGAkR?usp=sharing
Usage:
-	python demo.py
Limitation:
-	In order to simplify UI, program assumes there is single user is present in camera view

Libraries 
-	face detection from opencv
-	face landmark detection from mediapipe 
-	emotion recognition code and model from https://github.com/serengil/deepface
o	uses tensorflow
-	Hand gesture recognition from https://github.com/MahmudulAlam/Unified-Gesture-and-Fingertip-Detection
-	Other common libs such as numpy and h5py
-	PySimpleGUI 

