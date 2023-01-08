import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow.keras as keras


DATA_PATH = "data_10.json"

def load_data(data_path):


    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return  X, y


if __name__ == "__main__":

    X, y = load_data(DATA_PATH)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


    model = keras.Sequential([


        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),


        keras.layers.Dense(512, activation='relu'),


        keras.layers.Dense(256, activation='relu'),


        keras.layers.Dense(64, activation='relu'),


        keras.layers.Dense(10, activation='softmax')
    ])


    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=50)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
