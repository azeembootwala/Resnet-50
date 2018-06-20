import numpy as np
import tensorflow as tf


def init_filter(shape , stride):
    W = (np.random.randn(*shape)*np.sqrt(2.0/np.prod(shape[:-1]))).astype(np.float32)
    return W

class ConvLayer(object):
    # fd stands for filter dimentions
    def __init__(self, fd, mi, mo,stride=2,padding="VALID"):
        self.mi = mi
        self.mo = mo
        self.fd = fd
        self.stride = stride
        self.shape = (fd,fd,mi,mo)
        self.W = tf.Variable(init_filter(self.shape,self.stride))
        self.b =tf.Variable(np.zeros(self.mo, dtype = np.float32))
        self.padding = padding


    def forward(self, X):
        conv_out = tf.nn.conv2d(X,self.W,strides =[1,self.stride, self.stride,1],padding = self.padding)
        conv_out = conv_out + self.b
        return conv_out

    def get_params(self):
        return [self.W, self.b]


class BatchNorm(object):
    def __init__(self, D):
        self.running_mean = tf.Variable(np.zeros(D,dtype = np.float32), trainable = False)
        self.running_variance = tf.Variable(np.ones(D, dtype =np.float32), trainable = False)
        self.gamma = tf.Variable(np.ones(D,dtype = np.float32))
        self.beta = tf.Variable(np.zeros(D,dtype = np.float32))
    def forward(self,X):
        normalized = tf.nn.batch_normalization(
                    X,
                    self.running_mean,
                    self.running_variance,
                    self.beta,
                    self.gamma,1e-3)
        return self.session.run(normalized,feed_dict={tfX:X})
    def get_params(self):
        return [self.running_mean, self.running_variance, self.beta , self.gamma]


class ConvBlock(object):
    def __init__(self,mi,fm_sizes,stride):
        pass

    def predict(self):
        pass
    # TODO refactor the class to complete the Conv block archiecture



if __name__ =="__main__":
    conv_block = ConvBlock(mi=3,fm_sizes=[64,64,256],stride = 1)
    bn = BatchNorm(3)
    init = tf.global_variables_initializer()
    X = np.random.randn(1,400, 400 , 3).astype(np.float32)
    tfX = tf.placeholder(tf.float32,shape=(None, 400, 400,3))
    with tf.Session() as session:
        bn.session = session
        session.run(init)
        output = bn.forward(X)
        print(output.shape)
