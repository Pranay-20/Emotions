import numpy as np
import numpy
import pandas as pd
from keras.utils import to_categorical

class Datasets:

    def Images(self):
        # images
        dataset = pd.read_csv('train.csv', sep=',', header=None)
        dataset.dropna(inplace=True) #removing the null
        newdata = dataset[0].str.split(" ", expand=True) #spliting of the dataset
        data = pd.np.array(newdata, dtype=np.float32(256)) / 255
        data = pd.np.reshape(data, (data.shape[0], 48, 48, 1))
        return data

    def labels(self):
        # labels of emotion category (beween 0 and 6: anger=0, disgust=1, fear=2, happy=3, sad=4, surprise=5, neutral=6)
        Labels = pd.read_csv('Train_label.csv', header=None)
        # label
        Labels = to_categorical(Labels)
        label = np.array(Labels, dtype=np.int)
        return label

    def Test_img(self):
        test_data = pd.read_csv('Module/test.csv', sep=',', header=None)

        test_data.dropna(inplace=True)

        newtest = test_data[0].str.split(" ", expand=True)
        test = np.array(newtest, dtype=np.float32) / 255
        test = np.reshape(test, (test.shape[0], 48, 48, 1))
        return test




