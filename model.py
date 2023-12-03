import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """
    Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    num_nonzero_elems: The number of non-zero elements in the sparse matrix
    keep_prob:
    x: input
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GraphConvolutionSparse(Layer):
    """
    Graph convolution layer for sparse inputs.
    多了一个features_nonzero
    """

    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""

    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        """
        这个decoder部分实际上就只是input的转置再乘input
        """
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class GCN(Model):
    def __init__(self, placeholders, num_features, features_nonzero, settings, **kwargs):
        super(GCN, self).__init__(**kwargs)
        """
        inputs: Input features
        input_dim: dimensionality
        feature_nonzero：Non-zero feature number
        adj: adjacency matrix
        dropout：dropout
        """
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.settings = settings

    def construct(self, inputs=None, hidden=None, reuse=False):
        if inputs == None:
            inputs = self.inputs

        with tf.variable_scope('Encoder', reuse=reuse):
            self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                                  output_dim=self.settings.hidden1,
                                                  adj=self.adj,
                                                  features_nonzero=self.features_nonzero,
                                                  act=tf.nn.relu,
                                                  dropout=self.dropout,
                                                  logging=self.logging,
                                                  name='e_dense_1')(inputs)

            self.noise = gaussian_noise_layer(self.hidden1, 0.1)
            if hidden == None:
                hidden = self.hidden1
            self.embeddings = GraphConvolution(input_dim=self.settings.hidden1,
                                               output_dim=self.settings.hidden2,
                                               adj=self.adj,
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               name='e_dense_2')(hidden)
            self.z_mean = self.embeddings
            self.reconstructions = InnerProductDecoder(input_dim=self.settings.hidden2,
                                                       act=lambda x: x,
                                                       logging=self.logging)(self.embeddings)
            return self.z_mean, self.reconstructions


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        # np.random.seed(1)
        tf.set_random_seed(1)
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class Discriminator(Model):
    def __init__(self, settings, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.act = tf.nn.relu
        self.settings = settings

    def construct(self, inputs, reuse=False):
        with tf.variable_scope('Discriminator'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, self.settings.hidden2, self.settings.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, self.settings.hidden3, self.settings.hidden1, name='dc_den2'))
            output = dense(dc_den2, self.settings.hidden1, 1, name='dc_output')
            return output


class D_graph(Model):
    def __init__(self, num_features, **kwargs):
        super(D_graph, self).__init__(**kwargs)
        self.act = tf.nn.relu
        self.num_features = num_features

    def construct(self, inputs, reuse=False):
        # input是一张Graph的adj，把每一列当成一个通道，所以input的通道数是num_nodes
        with tf.variable_scope('D_Graph'):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            # np.random.seed(1)
            # tf.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, self.num_features, 512, name='GD_den1'))  # (bs,num_nodes,512)
            dc_den2 = tf.nn.relu(dense(dc_den1, 512, 128, name='GD_den2'))  # (bs, num_nodes, 128)
            output = dense(dc_den2, 128, 1, name='GD_output')  # (bs,num_nodes,1)
            return output


class Generator_z2g(Model):
    def __init__(self, placeholders, num_features, features_nonzero, settings, **kwargs):
        super(Generator_z2g, self).__init__(**kwargs)
        """
        inputs:输入
        input_dim:feature的数量，即input的维度？
        feature_nonzero：非0的特征
        adj:邻接矩阵
        dropout：dropout
        """

        self.inputs = placeholders['real_distribution']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.settings = settings

    def construct(self, inputs=None, reuse=False):
        if inputs == None:
            inputs = self.inputs
        with tf.variable_scope('Decoder', reuse=reuse):
            self.hidden1 = GraphConvolution(input_dim=self.settings.hidden2,
                                            output_dim=self.settings.hidden1,
                                            adj=self.adj,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='GG_dense_1')(inputs)

            self.embeddings = GraphConvolution(input_dim=self.settings.hidden1,
                                               output_dim=self.input_dim,
                                               adj=self.adj,
                                               act=lambda x: x,
                                               dropout=self.dropout,
                                               logging=self.logging,
                                               name='GG_dense_2')(self.hidden1)

        self.z_mean = self.embeddings
        return self.z_mean, self.hidden1


class BGAN(object):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, settings):
        self.discriminator = Discriminator(settings)
        self.D_Graph = D_graph(num_features)
        self.d_real = self.discriminator.construct(placeholders['real_distribution'])
        self.GD_real = self.D_Graph.construct(placeholders['features_dense'])
        self.ae_model = GCN(placeholders, num_features, features_nonzero, settings)
        self.model_z2g = Generator_z2g(placeholders, num_features, features_nonzero, settings)
