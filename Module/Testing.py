import cv2
import numpy as np
import pandas as pd

from Module.Datasets import Datasets
from Module.CNN import CNN_model

class Testing:
    def testModel(self,image):
        Cnn = CNN_model()
        model = Cnn.load_model()
        pred_classes=model.predict_classes(image)
        return pred_classes
