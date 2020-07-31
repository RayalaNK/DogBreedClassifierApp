import os
import csv
from itertools import chain
import cv2
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import load_model, Sequential
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense

global VGG19_model, ResNet50_model, dog_names

def face_detector(img_path):
    '''
    Function to detect a face in an image
    IN:  img_path - path to image for face detection
    OUT: True if face is detected
         False if no face is detected
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    '''
    Function to detect a dog in an image
    IN:  img_path - path to image
    OUT: "True" if dog is detected in the image, "False" otherwise
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def path_to_tensor(img_path):
    '''
    Function to pre-process image into 4d tensor as input for CNN
    IN:  img_path - path to image
    OUT: 4D-tensor with shape (1,224,224,3)
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    '''
    Function to pre-process images into 4d tensor as input for CNN
    IN:  img_paths - list of image paths
    OUT: list of 4D-tensor with shape (1,224,224,3)
    '''
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    '''
    Function to predict the image based on the pretrained ResNet50 models with weights from imagenet
    IN:  img_path - path to image
    OUT: predicted label with the highest probability
    '''
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def extract_VGG19(tensor):
    '''
    Function to predict an image based on the ResNet50 model with imagenet weights
    IN:  tensor
    OUT: output from the model
    '''
    from keras.applications.vgg19 import VGG19, preprocess_input
    return VGG19(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def VGG19_predict_breed(img_path):
    '''
        Function to detect a dog in an image
        IN:  img_path - path to image
        OUT: "True" if dog is detected in the image, "False" otherwise
        '''
    bottleneck_feature = extract_VGG19(path_to_tensor(img_path))
    predicted_vector = VGG19_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

def predict_dog_breed(img_path):
    '''
    Function to predict whether it's a dog or human is in the image and returns the breed
    of the dog or the resemblence of the dog breed when the image is of a human.
    IN:  img_path - path to image
    OUT: image and predicted dog breed
    '''
    dog_names_pred = VGG19_predict_breed(img_path)
    if dog_detector(img_path):
        prediction = "This looks like a dog and the breed is : {}!".format(dog_names_pred.split('.')[1])
    elif face_detector(img_path):
        prediction = "It's a human and resemblence is of a dog breed {}!".format(dog_names_pred.split('.')[1])
    else:
        prediction = "This is neither a dog nor a human! Try another image!"
    return prediction

def main():
    global VGG19_model, ResNet50_model, dog_names

    # Load the dog breeds
    with open('../data/dog_names.csv', 'r') as f:
        reader = csv.reader(f)
        dog_names = list(reader)

    dog_names = list(chain.from_iterable(dog_names))

    ResNet50_model = ResNet50(weights='imagenet')
    bottleneck_features = np.load('../saved_models/DogVGG19Data.npz')

    # load VGG19_model model and VGG19_model to classify the dog
    # VGG19_model = ResNet50(weights='imagenet')
    # train_VGG19 = bottleneck_features['train']
    # valid_VGG19 = bottleneck_features['valid']
    # test_VGG19 = bottleneck_features['test']

    # Define Architecture for VGG19_model
    VGG19_model = Sequential()
    VGG19_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
    # VGG19_model.add(GlobalAveragePooling2D(input_shape=train_VGG19.shape[1:]))
    VGG19_model.add(Dense(133, activation='softmax'))
    VGG19_model.load_weights('../saved_models/weights.best.VGG19.hdf5')

    # index page of the app
    app = Flask(__name__)
    WEBAPP_ROOT = os.path.dirname(os.path.abspath(__file__))

    # create root
    @app.route("/")
    def index():
        return render_template('index.html')

    @app.route("/upload", methods=['POST'])
    def upload():
        print(VGG19_model.summary())
        filepath = os.path.join(WEBAPP_ROOT, 'static/')
        print("Destination path : {}".format(filepath))
        if not os.path.isdir(filepath):
            os.mkdir(filepath)

        file = request.files["file"]
        print("File : {}".format(file))
        filename = file.filename
        destination = "/".join([filepath, filename])
        print("Destination:", destination)
        file.save(destination)
        desc = predict_dog_breed(destination)
        return render_template("complete.html", description=desc, image_name=filename)

    app.run(port=3001, debug=True)

if __name__ == '__main__':
    main()