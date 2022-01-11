# Dog-Breed-Classifier
Pipeline to process real-world user-provided images to Convolutional Neural Network that classifies image as Dog Breed

## Table of Contents:
1. [Project Introduction](#project-introduction);
2. [File Description](#file-description)
3. [Libraries Used](#libraries-used)
4. [Results](#results)
5. [Acknowledgements](#acknowledgements)

## Project Introduction
In this project, Images containing Dogs from different breeds are provided. Based on the available image data, the task is to classify the dog present in the image with the proper breed label. The data is provided by the Udacity program.

## File Description
- dog_app.ipynb - notebook contains complete python code starting from library imports to Exploratory Data Analysis, Preparing the Image Data to pass through the models, fitting and evaluating the performance of various models.
- saved_model - folder containing the weights of the best performing models trained in the dog_app.ipynb notebook
- images - provided by Udacity, contains images of dogs and people to be used for the classifier with additional images 
- requirements - provided by Udacity, used to install all packages for the notebook
- haarcascades - provided by Udacity, contains .xml file for the OpenCV face detector in the notebook
- dog_app.html - html version of the dog_app notebook that is good to get a view of the logic
- dog_app.zip - compressed file containing this project

## Libraries Used
The following libraries are used:
- Numpy
- Pandas
- Matplotlib
- Keras
- OpenCV

## Detect Humans
Made function that returns the percentage of human faces found in both the dog and human face datasets respectively, both datasets were of size 100.

## Detect Dogs
Using a pre-trained VGG16 model to find the predicted label for an image:
```dog_detector``` function returns ```True``` if a dog is detected ```False``` otherwise. 

## Classify Dog Breeds with custom CNN
The custom CNN architecture of trained model needed to at least achieve 10% accuracy on the test set. This is where the need for Transfer Learning is evident since the custom implementation required alot of time to achieve a score above 10%.

## Acknowledgements
Thank Udacity for the Nanodegree Couse and for providing the Data for the Image Classifier