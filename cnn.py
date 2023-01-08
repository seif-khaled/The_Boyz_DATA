import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH = "data_10.json"


def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)


    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y





def prepare_datasets(test_size):
    # load data
    X, y = load_data(DATA_PATH)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)



    X_train = X_train[..., np.newaxis]

    X_test = X_test[..., np.newaxis]

    return X_train, X_test, y_train,  y_test


def build_model(input_shape):


    # build network topology
    model = keras.Sequential()

    # 1st conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 2nd conv layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model


def predict(model, X, y):


    X = X[np.newaxis, ...]

    # perform prediction
    prediction = model.predict(X)


    predicted_index = np.argmax(prediction, axis=1)




if __name__ == "__main__":


    X_train,  X_test, y_train, y_test = prepare_datasets(0.25)


    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_model(input_shape)


    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # train model
    history = model.fit(X_train, y_train,  batch_size=32, epochs=30)




    # evaluate model on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)
