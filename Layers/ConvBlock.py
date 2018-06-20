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
        return normalized
    def get_params(self):
        return [self.running_mean, self.running_variance, self.beta , self.gamma]


class ConvBlock(object):
    def __init__(self,mi,fm_sizes,stride=2):
        assert(len(fm_sizes)==3)
        self.relu = tf.nn.relu
        self.session = None

        # For resnets kernel size is always 3 in conv2
        # For resnets stride for conv1 is 2 , rest the strides is 1

        # Filling object values for Conv main branch
        # Conv --->BN--->Conv--->BN--->Conv--->BN
        self.conv1 = ConvLayer(1,mi,fm_sizes[0],stride)
        self.bn1   = BatchNorm(fm_sizes[0])
        self.conv2 = ConvLayer(3,fm_sizes[0], fm_sizes[1],1, padding="SAME")
        self.bn2   = BatchNorm(fm_sizes[1])
        self.conv3 = ConvLayer(1,fm_sizes[1],fm_sizes[2],1)
        self.bn3   = BatchNorm(fm_sizes[2])

        #Filling objects for skip connections
        #Input --->Conv--->BN
        self.conv  = ConvLayer(1,mi,fm_sizes[2],stride)
        self.bn    = BatchNorm(fm_sizes[2])


        # Incase needed later
        self.layers = [self.conv1 , self.bn1,
                       self.conv2 , self.bn2,
                       self.conv3 , self.bn3]

        # will only be used when file is called directly
        self.input = tf.placeholder(tf.float32,shape=(None, 400, 400,mi))
        self.output = self.forward(self.input)


    def forward(self,X):
        FX = self.conv1.forward(X)
        FX = self.bn1.forward(FX)
        FX = self.relu(FX)
        FX = self.conv2.forward(FX)
        FX = self.bn2.forward(FX)
        FX = self.relu(FX)
        FX = self.conv3.forward(FX)
        FX = self.bn3.forward(FX)


        Sx = self.conv.forward(X)
        Sx = self.bn.forward(Sx)

        return self.relu(FX+Sx)

    def predict(self,X):
        assert(self.session is not None)
        return self.session.run(self.output,feed_dict={self.input:X})

    def set_session(self, session):
        for layer in self.layers:
            layer.session = session
        self.conv.session = session
        self.bn.session   = session
        self.session      = session
    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params


if __name__ =="__main__":
    conv_block = ConvBlock(mi=3,fm_sizes=[64,64,256],stride = 1)
    init = tf.global_variables_initializer()
    X = np.random.randn(1,400, 400 , 3).astype(np.float32)

    with tf.Session() as session:
        conv_block.set_session(session)
        session.run(init)
        output = conv_block.predict(X)
        print(output.shape)
