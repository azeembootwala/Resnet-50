import numpy as np
import tensorflow as tf
from ConvBlock import ConvLayer, BatchNorm


class IdentityBlock:
    def __init__(self,mi,fm_sizes,stride=2):
        self.session = None
        assert(len(fm_sizes)==3)
        self.relu = tf.nn.relu
        # Filling object list for Identity Block
        self.conv1 = ConvLayer(1,mi,fm_sizes[0],stride)
        self.bn1   = BatchNorm(fm_sizes[0])
        self.conv2 = ConvLayer(3,fm_sizes[0],fm_sizes[1],1,padding="SAME")
        self.bn2   = BatchNorm(fm_sizes[1])
        self.conv3 = ConvLayer(1,fm_sizes[1],fm_sizes[2],1)
        self.bn3   = BatchNorm(fm_sizes[2])

        self.layers = [self.conv1 , self.bn1,
                       self.conv2 , self.bn2,
                       self.conv3 , self.bn3]

        self.input = tf.placeholder(tf.float32,shape=(None,224,224,mi))
        self.output= self.forward(self.input)


    def forward(self,X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.relu(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.relu(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)
        return self.relu(FX + X)

    def predict(self, X):
        if self.session is not None:
            return self.session.run(self.output,feed_dict={self.input:X})

    def set_session(self, session):
        self.session = session
        for layer in self.layers:
            layer.session = session
    def get_params(self):
        params = []
        for layer in self.layers:
            params+=layer.get_params()
        return params

    def copyFromKerasLayers(self,layers):
        assert(len(layers)==10)
        self.conv1.copyFromKerasLayers(layers[0])
        self.bn1.copyFromKerasLayers(layers[1])
        self.conv2.copyFromKerasLayers(layers[3])
        self.bn2.copyFromKerasLayers(layers[4])
        self.conv3.copyFromKerasLayers(layers[6])
        self.bn3.copyFromKerasLayers(layers[7])


if __name__ == '__main__':
  identity_block = IdentityBlock(256,[64,64,256],1)

  # make a fake image
  X = np.random.random((1, 224, 224, 256))

  init = tf.global_variables_initializer()
  with tf.Session() as session:
    identity_block.set_session(session)
    session.run(init)

    output = identity_block.predict(X)
    print("output.shape:", output.shape)
