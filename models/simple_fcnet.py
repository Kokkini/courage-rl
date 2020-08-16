from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.models import ModelCatalog

tf = try_import_tf()

def make_base_model(x, layers, prefix):
    for i, size in enumerate(layers):
        x = tf.keras.layers.Dense(
                size,
                activation="relu",
                use_bias=True
        )(x)
    return x


class SimpleFCNet(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        print("LOADED CUSTOM MODEL")
        self.state_danger = model_config.get("custom_model_config", {}).get("state_danger", False)
        print(f"model is using state danger: {self.state_danger}")
        print(f"model_config: {model_config}")
        print(f"observation shape: {obs_space.shape}")
        layers = [64, 128, 64]

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        scaled_inputs = tf.cast(inputs, tf.float32) / 255.0

        x = scaled_inputs

        x_danger = make_base_model(x, layers, "danger")
        x = make_base_model(x, layers, "main")

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

        self.base_model = tf.keras.Model(inputs, [logits, value, danger_score])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        obs = tf.cast(input_dict["obs"], tf.float32)
        logits, self._value, self._danger_score = self.base_model(obs)
        return logits, state

    def value_function(self):
        return tf.reshape(self._value, [-1])

    def danger_score_function(self):
        if not self.state_danger:
            return self._danger_score
        else:
            return tf.reshape(self._danger_score, [-1])

# Register model in ModelCatalog
ModelCatalog.register_custom_model("simple_fcnet", SimpleFCNet)
