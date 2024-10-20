import tensorflow as tf

# a PReLU layer with a fixed alpha parameter, for imitating LeakyReLU on Google Coral
# https://coral.ai/docs/edgetpu/models-intro/#supported-operations
@tf.keras.utils.register_keras_serializable()
class FixedPReLU(tf.keras.layers.Layer):
    def __init__(self, alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, **kwargs):
        super(FixedPReLU, self).__init__(**kwargs)
        self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
        self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
        self.alpha_constraint = tf.keras.constraints.get(alpha_constraint)

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(input_shape[-1],),
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint,
                                     trainable=False)  # Set trainable to False to freeze the alpha parameter

    def call(self, inputs):
        pos = tf.nn.relu(inputs)
        neg = -self.alpha * tf.nn.relu(-inputs)
        return pos + neg

    def get_config(self):
        config = super(FixedPReLU, self).get_config()
        config.update({
            'alpha_initializer': tf.keras.initializers.serialize(self.alpha_initializer),
            'alpha_regularizer': tf.keras.regularizers.serialize(self.alpha_regularizer),
            'alpha_constraint': tf.keras.constraints.serialize(self.alpha_constraint),
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['alpha_initializer'] = tf.keras.initializers.deserialize(config['alpha_initializer'])
        config['alpha_regularizer'] = tf.keras.regularizers.deserialize(config['alpha_regularizer'])
        config['alpha_constraint'] = tf.keras.constraints.deserialize(config['alpha_constraint'])
        return cls(**config)