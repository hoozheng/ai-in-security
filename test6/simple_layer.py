import tensorflow as tf
from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dropout


class MLPLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 is_batch_norm=False,
                 is_dropout=0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        self.units = [units] if not isinstance(units, list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.is_batch_norm = is_batch_norm
        self.is_dropout = is_dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims = [last_dim] + self.units

        self.kernels = []
        self.biases = []
        self.bns = []
        for i in range(len(dims) - 1):
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i], dims[i + 1]],
                                initializer=self.kernel_initializer,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[
                                        dims[i + 1],
                                    ],
                                    initializer=self.bias_initializer,
                                    trainable=True))
            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built = True

    def call(self, inputs, is_train=False):
        _input = inputs
        for i in range(len(self.units)):
            _input = MatMul(a=_input, b=self.kernels[i])
            if self.use_bias:
                _input = BiasAdd(value=_input, bias=self.biases[i])
            # BN
            if self.is_batch_norm:
                _input = self.bns[i](_input)
            # ACT
            if self.activation is not None:
                _input = self.activation(_input)
            # DROP
            if is_train and self.is_dropout > 0:
                _input = Dropout(self.is_dropout)(_input)

        return _input


def buildMMOEModel(config: dict):
    input_dic = {}
    for f in config['feature_name_list']:
        input_dic[f] = tf.keras.Input(shape=(1,), name=f, dtype=tf.int64)

    embed = tf.keras.layers.Embedding(config['feature_size'],
                                      config['embedding_size'],
                                      embeddings_regularizer="l2")

    embed_list = []
    for k, v in input_dic.items():
        embed_list.append(embed(v))

    embed_list = tf.concat(embed_list, axis=1)
    embed_list = tf.keras.layers.Flatten()(embed_list)
    # expert list[batch, 8]
    experts = []
    for i in range(config['expert_num']):
        tmp = MLPLayer(units=[64, 8], activation='relu')(embed_list)
        experts.append(tmp)
    experts = tf.stack(experts, axis=1)  # [batch, self.expert_num, 8]

    # gate [batch, self.expert_num]
    ctr_gate = MLPLayer(units=[64, config['expert_num']],
                        activation='relu')(embed_list)
    cvr_gate = MLPLayer(units=[64, config['expert_num']],
                        activation='relu')(embed_list)
    ctr_gate = tf.expand_dims(ctr_gate, axis=2)
    cvr_gate = tf.expand_dims(cvr_gate, axis=2)

    ctr_input = tf.keras.layers.Flatten()(experts * ctr_gate)
    cvr_input = tf.keras.layers.Flatten()(experts * cvr_gate)

    # tower
    _ctr_out = MLPLayer(units=[64, 8], activation='relu')(ctr_input)
    _ctr_out = MLPLayer(units=[1], activation='sigmoid')(_ctr_out)
    _cvr_out = MLPLayer(units=[64, 8], activation='relu')(cvr_input)
    _cvr_out = MLPLayer(units=[1], activation='sigmoid')(_cvr_out)
    model = tf.keras.Model(input_dic, [_ctr_out, _cvr_out])
    model.summary()
    return model
