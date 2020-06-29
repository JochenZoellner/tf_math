import tensorflow as tf

from model_fn.model_fn_2d.util_2d.graphs_2d import Graph2D
import model_fn.util_model_fn.keras_compatible_layers as layers
from util.flags import update_params


class GraphConv2MultiFF(Graph2D):
    def __init__(self, params):
        super(GraphConv2MultiFF, self).__init__(params)
        # v0.1
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["conv_layer_activation"] = "leaky_relu"
        self.graph_params["input_dropout"] = 0.0
        self.graph_params["batch_norm"] = False
        self.graph_params["dense_layers"] = [256, 128]
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        if self.graph_params["conv_layer_activation"] == "None":
            conv_layer_activation_fn = None
        else:
            conv_layer_activation_fn = getattr(layers, self.graph_params["conv_layer_activation"])
        if self.graph_params["mid_layer_activation"] == "None":
            mid_layer_activation_fn = None
        else:
            mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        fc = tf.cast(inputs['fc'], dtype=tf.float32)
        fc = tf.reshape(fc, [-1, 3, self._flags.data_len, 1])
        # Conv1
        with tf.compat.v1.variable_scope("conv1"):
            kernel_dims = [3, 6, 1, 8]
            conv_strides = [1, 1, 3, 1]
            conv1 =  layers.conv2d_bn_lrn_drop(inputs=fc, kernel_shape=kernel_dims, is_training=is_training,
                                                                strides=conv_strides, activation=conv_layer_activation_fn,
                                                                use_bn=False, use_lrn=False, padding='SAME')
            conv1_len = int((self._flags.data_len + conv_strides[2] - 1) / conv_strides[2])
        # Conv2
        with tf.compat.v1.variable_scope("conv2"):
            kernel_dims = [1, 8, 8, 16]
            conv_strides = [1, 1, 6, 1]
            conv2 =  layers.conv2d_bn_lrn_drop(inputs=conv1, kernel_shape=kernel_dims, is_training=is_training,
                                                                strides=conv_strides, activation=conv_layer_activation_fn,
                                                                use_bn=False, use_lrn=False, padding='SAME')
            conv2_len = int((conv1_len + conv_strides[2] - 1) / conv_strides[2])
        ff_in = tf.reshape(conv2, [-1, conv2_len * 16 * 3])
        ff_in = fc
        for index, nhidden in enumerate(self.graph_params["dense_ keraslayers"]):
            ff_in =  layers.ff_layer(inputs=ff_in, outD=nhidden,
                                                      is_training=is_training, activation=mid_layer_activation_fn,
                                                      use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

        ff_final =  layers.ff_layer(inputs=ff_in, outD=self._flags.max_edges * 2,
                                                     is_training=is_training, activation=None, name="ff_final")

        radius_final =  layers.ff_layer(inputs=ff_final,
                                                         outD=1,
                                                         is_training=is_training,
                                                         activation=None,
                                                         name="radius_final")

        rotation_final =  layers.ff_layer(inputs=ff_final,
                                                           outD=1,
                                                           is_training=is_training,
                                                           activation=None,
                                                           name="rotation_final")

        translation_final =  layers.ff_layer(inputs=ff_final,
                                                              outD=2,  # 2 dimension problem
                                                              is_training=is_training,
                                                              activation=None,
                                                              name="translation_final")

        edge_final =  layers.ff_layer(inputs=ff_in,
                                                       outD=self._flags.max_edges - 3,  # at least a triangle!
                                                       is_training=is_training,
                                                       activation=layers.softmax,
                                                       name="edge_final")

        return {"radius_pred": radius_final,
                "rotation_pred": rotation_final,
                "translation_pred": translation_final,
                "edges_pred": edge_final}


class GraphMultiFF(Graph2D):
    def __init__(self, params):
        super(GraphMultiFF, self).__init__(params)
        # v0.2
        self.graph_params["mid_layer_activation"] = "leaky_relu"
        self.graph_params["batch_norm"] = False
        self.graph_params["dense_layers"] = [512, 256, 128, 64]
        self.graph_params["dense_dropout"] = []  # [0.0, 0.0] dropout after each dense layer
        self.graph_params["input_dropout"] = 0.01
        self.graph_params["abs_as_input"] = False
        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

    def infer(self, inputs, is_training):
        if self.graph_params["mid_layer_activation"] == "None":
            mid_layer_activation_fn = None
        else:
            mid_layer_activation_fn = getattr(layers, self.graph_params["mid_layer_activation"])

        if self.graph_params["abs_as_input"]:
            z_values = tf.slice(inputs['fc'], [0, 1, 0], [-1, 2, -1])
            z_squared = tf.square(z_values)
            abs_in = tf.sqrt(tf.reduce_sum(z_squared, axis=1, keep_dims=True))
            ff_in = tf.stack([abs_in, tf.slice(inputs['fc'], [0, 0, 0], [-1, 1, -1])])
            ff_in = tf.reshape(ff_in, (-1, 2 * self._flags.data_len))
        else:
            ff_in = tf.reshape(inputs['fc'], (-1, 3 * self._flags.data_len))

        if is_training and self.graph_params["input_dropout"] > 0:
            ff_in = tf.nn.dropout(ff_in, keep_prob=1.0 - self.graph_params["input_dropout"])

        for index, nhidden in enumerate(self.graph_params["dense_layers"]):
            ff_in = layers.ff_layer(inputs=ff_in, outD=nhidden,
                                    is_training=is_training, activation=mid_layer_activation_fn,
                                    use_bn=self.graph_params["batch_norm"], name="ff_{}".format(index + 1))

            if is_training and self.graph_params["dense_dropout"] and float(
                    self.graph_params["dense_dropout"][index]) > 0.0:
                ff_in = tf.nn.dropout(ff_in, keep_prob=1.0 - self.graph_params["input_dropout"])

        ff_final = ff_in
        radius_final = layers.ff_layer(inputs=ff_final,
                                       outD=1,
                                       is_training=is_training,
                                       activation=None,
                                       name="radius_final")

        rotation_final = layers.ff_layer(inputs=ff_final,
                                         outD=1,
                                         is_training=is_training,
                                         activation=None,
                                         name="rotation_final")

        translation_final = layers.ff_layer(inputs=ff_final,
                                            outD=2,  # 2 dimension problem
                                            is_training=is_training,
                                            activation=None,
                                            name="translation_final")

        edge_final = layers.ff_layer(inputs=ff_in,
                                     outD=self._flags.max_edges - 3,  # at least a triangle!
                                     is_training=is_training,
                                     activation=layers.softmax,
                                     name="edge_final")

        return {"radius_pred": radius_final,
                "rotation_pred": rotation_final,
                "translation_pred": translation_final,
                "edges_pred": edge_final}


class GraphConv1MultiFF(Graph2D):
    def __init__(self, params):
        super(GraphConv1MultiFF, self).__init__(params)
        # v0.4
        if not self._flags.complex_phi:
            self.fc_size_0 = 3
        else:
            self.fc_size_0 = 4
        self.graph_params["dense_layers"] = [512,1024,1024,256,128,64,32]
        self.graph_params["input_dropout"] = 0.0
        self.graph_params["ff_dropout"] = 0.0
        self.graph_params["uniform_noise"] = 0.0
        self.graph_params["normal_noise"] = 0.0
        self.graph_params["nhidden_dense_final"] = 6
        self.graph_params["edge_classifier"] = False
        self.graph_params["batch_norm"] = False
        self.graph_params["nhidden_max_edges"] = 6
        self.graph_params["pre_activation"] = None


        self.graph_params = update_params(self.graph_params, self._flags.graph_params, "graph")

        # initilize keras layer
        if self.graph_params["batch_norm"]:
            self._tracked_layers["batch_norm"] = tf.keras.layers.BatchNormalization(axis=2)
            if self.global_epoch >= 2:
                self._tracked_layers["batch_norm"].trainable = True
        self._tracked_layers["conv_1"] = tf.keras.layers.Conv1D(filters=4, kernel_size=8,
                                                                padding="same",
                                                                strides=2, activation=tf.nn.leaky_relu)
        self._tracked_layers["flatten_1"] = tf.keras.layers.Flatten()
        # loop over all number in self.graph_params["dense_layers"]
        for layer_index, n_hidden in enumerate(self.graph_params["dense_layers"]):
            name = "ff_{}".format(layer_index + 1)
            self._tracked_layers[name] = tf.keras.layers.Dense(n_hidden, activation=tf.nn.leaky_relu, name=name)

        self._tracked_layers["radius"] = tf.keras.layers.Dense(1, activation=None, name="radius")
        self._tracked_layers["rotation"] = tf.keras.layers.Dense(1, activation=None, name="rotation")
        self._tracked_layers["translation"] = tf.keras.layers.Dense(1, activation=None, name="translation")
        self._tracked_layers["edges"] = tf.keras.layers.Dense(self._flags.max_edges, activation=None, name="edges")

    @tf.function
    def call(self, inputs, training=False, build=None):

        ff_in = inputs["fc"][:, 1:]
        if self.graph_params["pre_activation"]:
            ff_in = getattr(layers, self.graph_params["pre_activation"])(ff_in)
        if self.graph_params["batch_norm"]:
            ff_in = self._tracked_layers["batch_norm"](ff_in, training)
        prepare_conv = tf.transpose(ff_in, [0, 2, 1])
        prepare_conv  = tf.expand_dims(self._tracked_layers["flatten_1"](prepare_conv),axis=-1)
        conv_1_res = self._tracked_layers["conv_1"](prepare_conv)
        conv_1_reshaped = tf.transpose(conv_1_res, [0, 2, 1])

        ff_in = tf.concat((ff_in, conv_1_reshaped), axis=1)

        ff_in = self._tracked_layers["flatten_1"](ff_in)

        if training and self.graph_params["input_dropout"] > 0:
            ff_in = tf.nn.dropout(ff_in, rate=self.graph_params["input_dropout"])
        # loop over all number in self.graph_params["dense_layers"]
        for layer_index, n_hidden in enumerate(self.graph_params["dense_layers"]):
            name = "ff_{}".format(layer_index + 1)
            ff_in = self._tracked_layers[name](ff_in)
            if training and self.graph_params["uniform_noise"] > 0:
                ff_in += tf.random.uniform(tf.shape(ff_in), minval=-self.graph_params["uniform_noise"],
                                           maxval=self.graph_params["uniform_noise"])
            if training and self.graph_params["normal_noise"] > 0:
                ff_in += tf.random.normal(tf.shape(ff_in), stddev=self.graph_params["normal_noise"])

        self._graph_out = {"fc": inputs["fc"]}

        pre_radius = self._tracked_layers["radius"](ff_in)
        self._graph_out["pre_radius"] = pre_radius

        pre_rotation = self._tracked_layers["rotation"](ff_in)
        self._graph_out["pre_rotation"] = pre_rotation

        pre_translation = self._tracked_layers["translation"](ff_in)
        self._graph_out["pre_translation"] = pre_translation

        pre_edges = self._tracked_layers["edges"](ff_in)
        self._graph_out["pre_edges"] = pre_edges

        return self._graph_out
