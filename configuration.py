import glob
import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf
import matplotlib.gridspec as gridspec
from playsound import playsound

# IMAGE_RESOLUTION = 512
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_DATA_TYPE = tf.float32
BATCH_SIZE = 1
NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CHANNELS = 2
NUM_CLASSES = 2
OUTPUT_HEIGHT = IMAGE_HEIGHT
OUTPUT_WIDTH = IMAGE_WIDTH
OUTPUT_SHAPE = (OUTPUT_HEIGHT, OUTPUT_WIDTH)

MASK_MIN = 0
MASK_MAX = 255
CLASS_INTERVAL_SIZE = (MASK_MAX - MASK_MIN) / NUM_CLASSES
# print(CLASS_INTERVAL_SIZE)
CLASS_SEGMENTS = [MASK_MIN + CLASS_INTERVAL_SIZE*i for i in range(0, NUM_CLASSES + 1)]
CLASS_SEGMENTS = numpy.array(CLASS_SEGMENTS).astype(numpy.uint8)
# CLASS_IDS = numpy.arange(-(NUM_CLASSES // 2), NUM_CLASSES // 2 + 1, 1)
# print(CLASS_IDS)

EPOCHS = 150

DATASET_PATH = "dataset"
TRAIN_PATH = "train"
TEST_PATH = "test"
REPRESENTATIVE_PATH = "representative"
IMAGE_GLOB = "*image*"
LABEL_GLOB = "*mask*"
SAMPLE_OUTPUT_DIRECTORY = "samples"
PREDICTION_OUTPUT_DIRECTORY = "predictions"

CMAP = "magma"
# CMAP = "gnuplot2"
# CMAP = "gist_stern"
# CMAP = "plasma"
# CMAP = "gist_grey"

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

def load_image(image_path):

    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, IMAGE_DATA_TYPE)
    if( IMAGE_DATA_TYPE == tf.float32 ):
        image = image / 255.0
    image = tf.image.resize(image, IMAGE_SHAPE)

    if( NUM_INPUT_CHANNELS == 1 ):
        image = tf.image.rgb_to_grayscale(image)
        # image = tf.image.convert_image_dtype(image, tf.uint8)

    return image


def load_mask(mask_path):

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.resize(mask, OUTPUT_SHAPE, method='nearest')

    for i in range(NUM_CLASSES):
        mask = tf.where(tf.math.logical_and(mask > CLASS_SEGMENTS[i], mask <= CLASS_SEGMENTS[i + 1]),
                        tf.ones_like(mask) * i, mask)

    return mask


def load_image_and_mask(image_path, mask_path):

    image = load_image(image_path)
    mask = load_mask(mask_path)

    return image, mask

def get_image_and_mask_paths(directory):
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    image_paths = []
    mask_paths = []

    for subdirectory in subdirectories:
        image_paths += sorted(glob.glob(os.path.join(subdirectory, IMAGE_GLOB)))
        mask_paths += sorted(glob.glob(os.path.join(subdirectory, LABEL_GLOB)))

    return image_paths, mask_paths

def create_datasets():

    train_images, train_masks = get_image_and_mask_paths(os.path.join(DATASET_PATH, TRAIN_PATH))
    test_images, test_masks = get_image_and_mask_paths(os.path.join(DATASET_PATH, TEST_PATH))

    print(f"Number of training images: {len(train_images)}")
    print(f"Number of training masks: {len(train_masks)}")
    print(f"Number of testing images: {len(test_images)}")
    print(f"Number of testing masks: {len(test_masks)}")

    train_dataset = (tf.data.Dataset.from_tensor_slices((train_images, train_masks))
                        .map(load_image_and_mask))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_masks))
                        .map(load_image_and_mask))

    return train_dataset, test_dataset


def create_representative_dataset():

    representative_images = sorted(glob.glob(os.path.join(DATASET_PATH, REPRESENTATIVE_PATH, IMAGE_GLOB)))
    print(f"Number of representative images: {len(representative_images)}")

    representative_dataset = (tf.data.Dataset.from_tensor_slices((representative_images))
                        .map(load_image))

    return representative_dataset

def create_batches():

    train_dataset, test_dataset = create_datasets()

    train_batches = train_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    print(train_batches.element_spec)
    print(train_batches.cardinality())

    test_batches = test_dataset.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    print(test_batches.element_spec)
    print(test_batches.cardinality())

    return train_batches, test_batches


def display_sample(dataset, model=None, samples_to_show=1, sample_index=1, epoch=None):

    samples_to_show = min(samples_to_show, len(dataset))

    # Set figure size and GridSpec for layout
    fig = plt.figure(figsize=(10, 6 * samples_to_show))

    if( epoch is not None ):
        plt.suptitle(f"Epoch: {epoch}")
    gs = gridspec.GridSpec(samples_to_show, 3 if model else 2, height_ratios=[1] * samples_to_show)

    for i in range(samples_to_show):
        index = sample_index + i

        for image, mask in dataset.take(index):

            print(f"{image.shape}")

            # Image subplot
            ax_image = fig.add_subplot(gs[i, 0])
            if( NUM_INPUT_CHANNELS == 3 ):
                ax_image.imshow(image)
            elif( NUM_INPUT_CHANNELS == 1 ):
                ax_image.imshow(image, cmap=CMAP)
            ax_image.axis('off')
            print(f"{type(image)=}")
            ax_image.set_title(f"Image {image.shape}, [{tf.reduce_min(image):0.2f}, {tf.reduce_max(image):0.2f}]", fontsize=12)
            print(f"Image {image.shape}, {image.dtype.name}, [{numpy.min(image):0.2f}, {numpy.min(image):0.2f}]")

            # Mask subplot
            ax_mask = fig.add_subplot(gs[i, 1])
            ax_mask.imshow(mask, cmap=CMAP)
            ax_mask.axis('off')
            # ax_mask.set_title("Mask", fontsize=12)
            ax_mask.set_title(f"Mask {mask.shape}, {mask.dtype.name}, [{int(numpy.min(mask)):d}, {int(numpy.max(mask)):d}]", fontsize=12)
            print(f"Mask {mask.shape}, {mask.dtype.name}, [{int(numpy.min(mask)):d}, {int(numpy.max(mask)):d}]")

            if model is not None:
                # Prediction subplot
                ax_pred = fig.add_subplot(gs[i, 2])
                prediction = model.predict(image[tf.newaxis, ...])
                prediction = tf.math.argmax(prediction, axis=-1).numpy().squeeze()
                # prediction = prediction > 0.5
                ax_pred.imshow(prediction, cmap=CMAP)
                ax_pred.axis('off')
                # print(type(prediction))
                ax_pred.set_title(f"Prediction {prediction.shape}, [{prediction.min()}, {prediction.max()}]", fontsize=12)
                print(f"Prediction {prediction.shape}, [{prediction.min()}, {prediction.max()}]")

        print()
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_history(history):
    # Get accuracy and loss data
    accuracy = history.history['accuracy']
    loss = history.history['loss']

    # Find the index of the highest accuracy
    max_accuracy_index = accuracy.index(max(accuracy))
    max_accuracy_value = accuracy[max_accuracy_index]

    # Plot accuracy and loss on the same graph
    plt.figure(figsize=(14, 6))
    plt.plot(accuracy, label='Accuracy', color='blue', marker='o')
    plt.plot(loss, label='Loss', color='red', marker='x')

    # Highlight the maximum accuracy point
    plt.scatter(max_accuracy_index, max_accuracy_value, color='green', zorder=5, label='Max Accuracy')

    # Title and labels
    plt.title('Model Accuracy and Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')

    # Set x-axis ticks to be sparse
    epochs = len(accuracy)
    plt.xticks(ticks=range(0, epochs, max(1, epochs // 20)))  # Adjust the divisor to control tick density

    # Add legend and grid
    plt.legend()
    plt.grid()

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Play alert sound
    from playsound import playsound
    playsound("alert.mp3")

    plt.savefig("history.png")

if __name__ == "__main__":

    print("Running configuration file")
    print("Attempting to create datasets:")

    train_dataset, test_dataset = create_datasets()
    # train_batches, test_batches = create_batches()

    train_dataset = train_dataset.shuffle(1000)
    test_dataset  = test_dataset.shuffle(1000)

    display_sample(train_dataset, samples_to_show=min(3, len(train_dataset)))
    display_sample(test_dataset, samples_to_show=min(3, len(test_dataset)))

    representative_dataset = create_representative_dataset()
