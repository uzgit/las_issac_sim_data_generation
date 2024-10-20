import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Resizing, LayerNormalization, PReLU, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, BatchNormalization, LeakyReLU, Concatenate, Input
from tensorflow.keras.activations import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

import configuration
from FixedPReLU import *

def down_block(x, filters, use_maxpool=True):
    x = Conv2D(filters, 3, padding='same')(x)
    x = FixedPReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = FixedPReLU()(x)
    if use_maxpool == True:
        return MaxPooling2D(strides=(2, 2))(x), x
    else:
        return x


def up_block(x, y, filters):
    # x = UpSampling2D()(x)
    x = Conv2DTranspose(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, y])
    x = Conv2D(filters, 3, padding='same')(x)
    x = FixedPReLU()(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = FixedPReLU()(x)
    return x

def Unet(input_size=(512, 512, 1), *, classes, dropout):

    filter = [8, 16, 32, 64, 128, 256, 512]
    _input = Input(shape=input_size, batch_size=configuration.BATCH_SIZE, dtype=configuration.IMAGE_DATA_TYPE)
    # _input2 = tf.keras.layers.Resizing(height=256, width=256)(_input)
    # starting placeholder
    # x = input_layer

    # encode
    x, temp1 = down_block(_input, filter[0])
    x, temp2 = down_block(x, filter[1])
    x, temp3 = down_block(x, filter[2])
    x, temp4 = down_block(x, filter[3])

    # bridge
    x = down_block(x, filter[1], use_maxpool=False)

    # decode
    x = up_block(x, temp4, filter[3])
    x = up_block(x, temp3, filter[2])
    x = up_block(x, temp2, filter[1])
    x = up_block(x, temp1, filter[0])
    x = Dropout(dropout)(x)

    _output = Conv2D(classes, 1, activation='softmax')(x)
    # _output2 = Resizing(512, 512, interpolation="nearest")(_output)

    model = models.Model(_input, _output, name='unet')
    return model

train_dataset, test_dataset = configuration.create_datasets()
train_batches, test_batches = configuration.create_batches()

configuration.display_sample(train_dataset, samples_to_show=min(3, len(train_dataset)))

model = Unet(input_size=(configuration.IMAGE_HEIGHT, configuration.IMAGE_WIDTH, configuration.NUM_INPUT_CHANNELS),
             classes=configuration.NUM_CLASSES, dropout=0.2)
model.summary()
plot_model(model, show_shapes=True)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000025),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        configuration.display_sample(train_dataset, model=model, samples_to_show=3)

callbacks = [tf.keras.callbacks.ModelCheckpoint("unet.keras", save_best_only=True),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15),
             DisplayCallback()
            ]

model_history = model.fit(x=train_batches,
                         validation_data=test_batches,
                         epochs=100,
                         callbacks=callbacks
                        )

configuration.plot_history(model_history)