from keras.engine.saving import model_from_json, load_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,Flatten,Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.layers.core import Activation


# load json and create model
#new_model = load_model('model.h5')
#new_model.summary()


#loaded_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])