from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input

def get_model():
    # create model
    
    model = Sequential()

    # based on this architecture https://link.springer.com/article/10.1007/s11036-019-01243-5
    model.add(Conv2D(128, 3, activation='relu', input_shape=(32, 32, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(192, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # flat it out to send into dense layers
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))

    # compute probability over classes using softmax activation
    model.add(Dense(26, activation='softmax'))

    return model