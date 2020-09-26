import sys
import os
import string
from flask import Flask,jsonify,request
import random
import operator
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D


featuresdf = pd.read_hdf('./models/dataframes_backup.h5')

classes = {'1': 'gun_shot', '0': 'no_gun_shot'}
################################### Extract features ###########################
def extract_features(file_name):

    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)

    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None

    return mfccsscaled

def extract_features_arr(audio_arr, sample_rate):

    mfccs = librosa.feature.mfcc(y=audio_arr, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)

    return mfccsscaled
################################################################################


tf.enable_eager_execution()
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

num_rows = 4
num_columns = 10
num_channels = 1
num_labels = 2

# Construct model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding="same", input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='sigmoid'))

model.load_weights("./models/best_weights_temp.hdf5")

def predict_class(featuresdf):
    pred_index_tensor=tf.argmax(model.predict(featuresdf.reshape(1, 4, 10, 1)), axis=1) #() #1, x_train.shape[1], x_train.shape[2], x_train.shape[3]
    pred_index_arr = pred_index_tensor.numpy()
    pred_index = str(pred_index_arr[0])
    pred_class = classes[pred_index]
    return (pred_index, pred_class)

def index_to_class(index):
    return classes[str(index)]

app = Flask(__name__)

@app.route('/process', methods=['POST','PUT'])


def post():

    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(r"./audio", filename))
    fileName=os.path.join(r"./audio", filename)
    file_path=fileName

    f_features = extract_features(file_path)
    class_index, class_name = predict_class(f_features)
    print("\tPredicted Class: " + str(class_index))
    print("\tPredicted Class Name: " + str(class_name))

    return jsonify({"Audio" : str(class_name)})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)#
