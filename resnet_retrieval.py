import numpy as np
import tensorflow as tf
import cv2
import sys
from Resnet import ResNet , DenseLayer
sys.path.insert(0,"../code/")
from preprocessing import PreProcess
from Generators import Generators
from Contrastive_loss import Contrastive


## Keras imports
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input , decode_predictions


class Retrieval(object):
    def __init__(self):
        self.resnet_head = ResNet()

    def fit(self,traingen , valgen , contrastive_obj, keras_model):
        X = np.ones((1,400,400,3),dtype=np.float32)

        init = tf.variables_initializer(self.resnet_head.get_params())
        session = keras.backend.get_session()
        self.resnet_head.set_session(session)
        session.run(init)
        self.resnet_head.copyFromKerasLayers(keras_model.layers)
        output= self.resnet_head.predict(X)
        print(output)



def main():
    batch_size = 16
    traingen = Generators(batch_size = batch_size).traindatagen()
    valgen = Generators(batch_size=batch_size).valdatagen()
    contrastive_obj = Contrastive(batch_size) # loss object
    resnet = ResNet50(weights='imagenet')
    Model = Retrieval()
    Model.fit(traingen,valgen,contrastive_obj,resnet)




if __name__ == "__main__":
    main()
