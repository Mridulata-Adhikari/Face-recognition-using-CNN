import time

import pickle
from sklearn.preprocessing import LabelEncoder
from network import NeuralNetwork
from layers import *
from dataset import GetDataSet
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_cnn_model():

    embeddings_path = "images" + os.sep + "train" + os.sep + "final5c-cnn.pickle"
    classifier_model_path = "models" + os.sep + "face_5c_10iter.pickle"
    label_encoder_path = "models" + os.sep + "label_5c_10iter.pickle"


    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())

    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    print(labels)



    getdataob = GetDataSet()

    X_train, Y_train, X_test, Y_test, X_valid, Y_valid = getdataob.dataset(98)
    end_iter_time = time.time()

    n_classes = len(np.unique(Y_train))


    # Create CNN Feature Extractor
    nn = NeuralNetwork(
        layers=[
            # CNN Feature Extractor
            Conv(
                n_filters=4,
                filter_shape=(3, 3),
                weight_scale=0.1,
                weight_decay=0.01
            ),
            Activation('relu'),
            Pool(
                pool_shape=(2, 2),
                mode='max'
            ),
            Conv(
                n_filters=8,
                filter_shape=(3, 3),
                weight_scale=0.1,
                weight_decay=0.01
            ),
            Activation('relu'),
            Pool(
                pool_shape=(2, 2),
                mode='max'
            ),

            Flatten(),  # Gives Feature Vectors

            # Classifier
            Linear(
                n_out=n_classes,
                weight_scale=0.1,
                weight_decay=0.002
            ),
            LogRegression(),
        ],
    )
    # Initialize(Setup) The Layers of CNN
    nn._setup(X_train, Y_train, X_valid, Y_valid)

    # Fit the Training Set to CNN to learn the task specific filters for Feature Extraction
    nn.fit(X_train, Y_train, X_valid, Y_valid, learning_rate=0.1, max_iter=2, batch_size=64)

    print("\nModel Trained\n\n")

    print("\nTesting Prediction : \n")
    Y_pred = nn.predict(X_test)
    print("Actual  : ", Y_test)
    print("Predicted : ", Y_pred)
    error = nn._error(Y_pred, Y_test)
    print("Testing Error : ", error)



    print("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(nn, write_file)

    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)

    file = pd.read_csv('data.txt')

    lines = file.plot.line(x='epoch', y=['acc', 'val_acc'])
    plt.title('CNN learning curves: Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()
    plt.savefig("Accuracy.png")

    lines2 = file.plot.line(x='epoch', y=['loss', 'valoss'])
    plt.title('CNN learning curves: Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    plt.savefig("Loss.png")



train_cnn_model()