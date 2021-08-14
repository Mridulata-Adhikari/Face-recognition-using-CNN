# import the necessary packages
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from source.dataset import GetDataSet

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB


def train_cnn_model(embeddings_path="", classifier_model_path="", label_encoder_path=""):
    # Trains a MLP classifier using embedding file from "embeddings_path",
    # then saves the trained model as "classifier_model_path" and
    # label encoding as "label_encoder_path".

    # Load the face embeddings
    print("[INFO] loading face embeddings...")
    data = pickle.loads(open(embeddings_path, "rb").read())

    # Encode the labels
    print("[INFO] encoding labels...")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    class_number = len(set(labels))

    # Reshape the data
    embedding_mtx = np.zeros([len(data["embeddings"]), len(data["embeddings"][0])])
    for ind in range(1, len(data["embeddings"])):
        embedding_mtx[ind, :] = data["embeddings"][ind]


    getdataob = GetDataSet()

    X_train, Y_train, X_test, Y_test, X_valid, Y_valid = getdataob.dataset(0.2)


    # Train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] training model...")

    cnn_model = Sequential([
        Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=X_train),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=54, kernel_size=5, activation='relu', input_shape=X_train),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(2024, activation='relu'),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        # 20 is the number of outputs
        Dense(20, activation='softmax')
    ])

    cnn_model.compile(
        loss='sparse_categorical_crossentropy',  # 'categorical_crossentropy',
        optimizer=Adam(lr=0.0001),
        metrics=['accuracy']
    )

    cnn_model.fit(X_train, Y_train)

    print("[INFO] saving model...")
    # Write the actual face recognition model to disk as pickle
    with open(classifier_model_path, "wb") as write_file:
        pickle.dump(cnn_model, write_file)

    # Write the label encoder to disk as pickle
    with open(label_encoder_path, "wb") as write_file:
        pickle.dump(le, write_file)