import numpy as np
import tensorflow as tf
import keras

from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.applications.resnet50 import preprocess_input , decode_predictions

from ConvBlock import ConvLayer , BatchNorm , ConvBlock
from IdentityBlock import IdentityBlock
from FirstLayers import ReLu , MaxPool



class AveragePool(object):
    def __init__(self,ksize):
        self.ksize = ksize
    def forward(self,X):
        out = tf.nn.avg_pool(X,
                            ksize=[1,self.ksize,self.ksize,1],
                            strides=[1,1,1,1],
                            padding="VALID")
        return out
    def get_params(self):
        return []

class Flatten(object):
    def forward(self, X):
        return tf.contrib.layers.flatten(X)
    def get_params(self):
        return []

class DenseLayer(object):
    def __init__(self, mi , mo ):
        self.W = tf.Variable((np.random.randn(mi,mo)*np.sqrt(2.0 / mi)).astype(np.float32))
        self.b = tf.Variable(np.zeros(mo,dtype=np.float32))

    def forward(self,X):
        out = tf.matmul(X, self.W) + self.b
        return out

    def copyFromKerasLayers(self, layers):
        W , b = layers.get_weights()
        op1 = self.W.assign(W)
        op2 = self.b.assign(b)
        self.session.run((op1,op2))

    def get_params(self):
        return [self.W , self.b]
class ResNet(object):
    def __init__(self):
        # Lets Build the ResNet architecture
        self.conv = ConvLayer(fd=7,mi=3,mo=64,stride = 2, padding = "SAME")
        self.bn   = BatchNorm(64)
        self.relu = ReLu()
        self.maxpool = MaxPool(dim = 3)
        # ConvBlock 1
        self.convb1 = ConvBlock(mi = 64 , fm_sizes =[64,64,256],stride=1) # shouldnt the stride be 2
        # 2 Identity Blocks 1 & 2
        self.identityb1 = IdentityBlock(mi = 256 , fm_sizes=[64,64,256])
        self.identityb2 = IdentityBlock(mi = 256 , fm_sizes=[64,64,256])
        # ConvBlock 2
        self.convb2 = ConvBlock(mi = 256 , fm_sizes=[128,128,512],stride=2)
        #  Identity Block 3 , 4 & 5
        self.identityb3 = IdentityBlock(mi=512 , fm_sizes=[128,128,512])
        self.identityb4 = IdentityBlock(mi=512 , fm_sizes=[128,128,512])
        self.identityb5 = IdentityBlock(mi=512 , fm_sizes=[128,128,512])
        # ConvBlock 3
        self.convb3 = ConvBlock(mi = 512 , fm_sizes=[256,256,1024],stride = 2)
        # IdentityBlock 6,7,8,9,10
        self.identityb6 = IdentityBlock(mi = 1024,fm_sizes=[256,256,1024])
        self.identityb7 = IdentityBlock(mi = 1024,fm_sizes=[256,256,1024])
        self.identityb8 = IdentityBlock(mi = 1024,fm_sizes=[256,256,1024])
        self.identityb9 = IdentityBlock(mi = 1024,fm_sizes=[256,256,1024])
        self.identityb10 = IdentityBlock(mi = 1024,fm_sizes=[256,256,1024])
        # ConvBlock 4
        self.convb4 = ConvBlock(mi = 1024,fm_sizes=[512,512,2048],stride=2)
        # IdentityBlock 11 , 12
        self.identityb11 = IdentityBlock(mi = 2048, fm_sizes=[512,512,2048])
        self.identityb12 = IdentityBlock(mi = 2048, fm_sizes=[512,512,2048])
        # Add average pool layers
        self.avgpool = AveragePool(ksize=7)
        self.flatten = Flatten()
        # this will change for retrieval pay attention to sizes
        self.dense = DenseLayer(mi=2048,mo = 1000)

        self.layers = [self.conv , self.bn , self.relu, self.maxpool,
                       self.convb1, self.identityb1 , self.identityb2,
                       self.convb2, self.identityb3 , self.identityb4 , self.identityb5,
                       self.convb3, self.identityb6 , self.identityb7 , self.identityb8,
                                    self.identityb9,self.identityb10,
                        self.convb4 , self.identityb11 , self.identityb12,
                        self.avgpool,self.flatten,self.dense]
        # shape will change coz of different sizes
        self.input = tf.placeholder(tf.float32,shape=(None,224,224,3))
        self.output= self.forward(self.input)

    def copyFromKerasLayers(self, layers):
         self.conv.copyFromKerasLayers(layers[1])
         self.bn.copyFromKerasLayers(layers[2])
         self.convb1.copyFromKerasLayers(layers[5:17])
         self.identityb1.copyFromKerasLayers(layers[17:27])
         self.identityb2.copyFromKerasLayers(layers[27:37])
         self.convb2.copyFromKerasLayers(layers[37:49])
         self.identityb3.copyFromKerasLayers(layers[49:59])
         self.identityb4.copyFromKerasLayers(layers[59:69])
         self.identityb5.copyFromKerasLayers(layers[69:79])
         self.convb3.copyFromKerasLayers(layers[79:91])
         self.identityb6.copyFromKerasLayers(layers[91:101])
         self.identityb7.copyFromKerasLayers(layers[101:111])
         self.identityb8.copyFromKerasLayers(layers[111:121])
         self.identityb9.copyFromKerasLayers(layers[121:131])
         self.identityb10.copyFromKerasLayers(layers[131:141])
         self.convb4.copyFromKerasLayers(layers[141:153])
         self.identityb11.copyFromKerasLayers(layers[153:163])
         self.identityb12.copyFromKerasLayers(layers[163:173])
         self.dense.copyFromKerasLayers(layers[175])

    def forward(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def predict(self,X):
        assert(self.session is not None)
        return self.session.run(self.output,feed_dict={self.input:X})

    def set_session(self,session):
        self.session = session
        for layer in self.layers:
            if isinstance(layer,ConvBlock) or isinstance(layer,IdentityBlock):
                layer.set_session(session)
            else:
                layer.session=session
    def get_params(self):
        params = []
        for layer in self.layers:
            params+=layer.get_params()


if __name__ == "__main__":
    # Understand this part thouroughly , write it again if possible

    # you can also set weights to None, it doesn't matter
    resnet_ = ResNet50(weights='imagenet')

    # make a new resnet without the softmax
    x = resnet_.layers[-2].output
    W, b = resnet_.layers[-1].get_weights()
    y = Dense(1000)(x)
    resnet = Model(resnet_.input, y)
    resnet.layers[-1].set_weights([W, b])

    # you can determine the correct layer
    # by looking at resnet.layers in the console
    partial_model = Model(
        inputs=resnet.input,
        outputs=resnet.layers[175].output
        )

    # maybe useful when building your model
    # to look at the layers you're trying to copy
    print(partial_model.summary())

    # create an instance of our own model
    my_partial_resnet = ResNet()

    # make a fake image
    X = np.random.random((1, 224, 224, 3))

    # get keras output
    keras_output = partial_model.predict(X)

    ### get my model output ###

    # init only the variables in our net
    init = tf.variables_initializer(my_partial_resnet.get_params())

    # note: starting a new session messes up the Keras model
    session = keras.backend.get_session()
    my_partial_resnet.set_session(session)
    session.run(init)

    # first, just make sure we can get any output
    first_output = my_partial_resnet.predict(X)
    print("first_output.shape:", first_output.shape)

    # copy params from Keras model
    my_partial_resnet.copyFromKerasLayers(partial_model.layers)

    # compare the 2 models
    output = my_partial_resnet.predict(X)
    diff = np.abs(output - keras_output).sum()
    if diff < 1e-10:
        print("Everything's great!")
    else:
        print("diff = %s" % diff)
