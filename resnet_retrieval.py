import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        self.input = tf.placeholder(tf.float32, shape=(None,400,400,3))
        self.labels = tf.placeholder(tf.float32, shape=(None,))
        self.output = self.forward(self.input)
        self.lr = 0.0001

    def forward(self,X):
        for layer in self.resnet_head.layers:
            X = layer.forward(X)
        X = tf.nn.l2_normalize(X, axis = 1)
        return X


    def fit(self,batch_size,traingen , valgen , contrastive_obj, keras_model):
        N = len(os.listdir("../Data/Augmented_data/Images/"))


        init = tf.variables_initializer(self.resnet_head.get_params())
        session = keras.backend.get_session()
        self.resnet_head.set_session(session)
        session.run(init)
        self.session = session
        self.resnet_head.copyFromKerasLayers(keras_model.layers)

        embedding = self.forward(self.input)
        tf.add_to_collection("embedding", embedding)
        saver = tf.train.Saver()

        margin = 0.4

        anchor_left , anchor_right = contrastive_obj.pair_combos(embedding)
        labels = contrastive_obj.get_binaray_labels(self.labels)
        first_part , second_part , cost = contrastive_obj.contrastive_loss(labels,
                                                                        anchor_left,anchor_right,margin)


        optimizer = tf.train.AdamOptimizer(self.lr)
        trainin_op = optimizer.minimize(cost) # current best 0.0001
        epoch = 50
        n_batches = N // batch_size

        LL_train = []
        LL_val = []
        #self.session.run(tf.variables_initializer(optimizer.variables())) # Gives a wierd cost on batches
        self.session.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in range(n_batches):
                X , Y , _ = next(traingen)
                self.session.run(trainin_op, feed_dict={self.input:X,self.labels:Y})

                if j % 100 ==0:
                    loss_train = self.session.run(cost,feed_dict={self.input:X,self.labels:Y})
                    Xval , Yval , val_name = next(valgen)
                    loss_val  = self.session.run(cost,feed_dict={self.input:Xval,self.labels:Yval})
                    LL_val.append(loss_val)
                    LL_train.append(loss_train)
                    print(" epoch %d of %d iteration %d of %d , val_loss is %.6f" %(i ,epoch-1,j,n_batches,loss_val))
                    print(" epoch %d of %d iteration %d of %d , train_loss is %.6f" %(i ,epoch-1,j,n_batches,loss_train))
        if not os.path.exists("./Models/Resnet50-plain"):
            os.makedirs("./Models/Resnet50-plain")
        saver.save(self.session,"./Models/Resnet50-plain/Resnet50-plain")

        fig = plt.figure("Resnet trained from scratch")
        plt.plot(LL_train, label="train_cost")
        plt.plot(LL_val,label="val_cost")
        plt.legend()
        if not os.path.exists("./Plots"):
            os.makedirs("./Plots")
        fig.savefig("./Plots/resnet_train", transparent=True,bbox_inches = "tight" ,pad_inches=0)


def main():
    batch_size = 16
    traingen = Generators(batch_size = batch_size).traindatagen()
    valgen = Generators(batch_size=batch_size).valdatagen()
    contrastive_obj = Contrastive(batch_size) # loss object
    resnet = ResNet50(weights='imagenet')
    Model = Retrieval()
    Model.fit(batch_size,traingen,valgen,contrastive_obj,resnet)




if __name__ == "__main__":
    main()
