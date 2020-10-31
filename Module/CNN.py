from keras.engine.saving import model_from_json, load_model

from Module.Datasets import Datasets
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop


class CNN_model():
    def Cnn(self):
        dataset = Datasets()
        images = dataset.Images()
        labels = dataset.labels()
        # Defining the CNN model using keras
        model = Sequential()
        model.add(Conv2D(64, (5, 5),
                         activation='relu',
                         input_shape=(48, 48, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten(name='flatten'))
        model.add(Dense(7, activation='softmax'))
        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='RMSprop',
                      metrics=['accuracy'])
        model.save('model_Cnn.h5')

        return model

    def train(self, model):
        dataset = Datasets()
        images = dataset.Images()
        labels = dataset.labels()
        X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.20, random_state=42)
        batch_size = 64
        nb_epochs = 30
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=nb_epochs,
                            verbose=True,
                            # verbose controls the infromation to be displayed. 0: no information displayed
                            validation_data=(X_val, Y_val),
                            initial_epoch=0)

        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        return model

    def load_model(self):
        # load json and create model
        model = Sequential()
        model.add(Conv2D(64, (5, 5),
                         activation='relu',
                         input_shape=(48, 48, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten(name='flatten'))
        model.add(Dense(7, activation='softmax'))
        model.load_weights('Module/model.h5')
        return model

#cnn = CNN_model();
#cnn1=cnn.Cnn()
#cnn.train(cnn1)