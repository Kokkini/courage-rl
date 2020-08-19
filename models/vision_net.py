from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()


def conv_layer(depth, name):
    return tf.keras.layers.Conv2D(
        filters=depth, kernel_size=3, strides=1, padding="same", name=name
    )

def residual_block(x, depth, prefix):
    inputs = x
    assert inputs.get_shape()[-1].value == depth
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv0")(x)
    x = tf.keras.layers.ReLU()(x)
    x = conv_layer(depth, name=prefix + "_conv1")(x)
    return x + inputs

def conv_sequence(x, depth, strides, prefix):
    x = conv_layer(depth, prefix + "_conv")(x)
    if strides > 1:
        x = tf.keras.layers.MaxPool2D(pool_size=3, strides=strides, padding="same")(x)
    x = residual_block(x, depth, prefix=prefix + "_block0")
    x = residual_block(x, depth, prefix=prefix + "_block1")
    return x

def make_base_model(x, depths, strides, prefix):
    for i, depth in enumerate(depths):
        x = conv_sequence(x, depth, strides[i], prefix=f"{prefix}_seq{i}")
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(units=256, activation="relu", name=f"{prefix}_hidden")(x)
    return x

class VisionNet(TFModelV2):
    """
    Network from IMPALA paper implemented in ModelV2 API.

    Based on https://github.com/ray-project/ray/blob/master/rllib/models/tf/visionnet_v2.py
    and https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/models.py#L28
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        print("LOADED CUSTOM MODEL")
        self.state_danger = model_config.get("custom_model_config", {}).get("state_danger", False)
        use_curiosity = model_config.get("custom_model_config", {}).get("use_curiosity", False)
        print(f"model is using state danger: {self.state_danger}")
        print(f"model_config: {model_config}")
        print(f"observation shape: {obs_space.shape}")
        print(f"use curiosity: {use_curiosity}")
        depths = [16, 32, 32]
        strides = [2,2,2]


        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs

        x_danger = make_base_model(x, depths, strides, "danger")
        encoding = 0
        encoding_random = 0
        if use_curiosity:
            x_encode = make_base_model(x, depths, strides, "encode")
            x_random = make_base_model(x, depths, strides, "random")
            encoding_size = model_config["custom_model_config"]["curiosity_encoding_size"]
            encoding = tf.keras.layers.Dense(units=encoding_size, name="encode_out", use_bias=False)(x_encode)
            encoding_random = tf.keras.layers.Dense(units=encoding_size, name="encode_random_out", use_bias=False)(x_random)

        x = make_base_model(x, depths, strides, "main")

        logits = tf.keras.layers.Dense(units=num_outputs, name="pi", use_bias=False)(x)
        value = tf.keras.layers.Dense(units=1, name="vf")(x)
        if not self.state_danger:
            danger_score = tf.keras.layers.Dense(units=num_outputs,
                                                 name="danger_score", kernel_initializer="zeros",
                                                 use_bias=False)(x_danger)
        else:
            danger_score = tf.keras.layers.Dense(units=1,
                                                 name="danger_score", kernel_initializer="zeros",
                                                 use_bias=False)(x_danger)

        self.base_model = tf.keras.Model(inputs, [logits, value, danger_score, encoding, encoding_random])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value, self._danger_score, self._encoding, self._encoding_random = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])

    def get_encoding(self):
        return self._encoding

    def get_encoding_random(self):
        return self._encoding_random

    def danger_score_function(self):
        if not self.state_danger:
            return self._danger_score
        else:
            return tf.reshape(self._danger_score, [-1])

# Register model in ModelCatalog
ModelCatalog.register_custom_model("vision_net", VisionNet)
