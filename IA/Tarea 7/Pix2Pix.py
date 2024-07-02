import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import tensorflow.keras as keras
import numpy as np
from skimage.transform import resize
import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";


conv_base=None
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
conv_base.summary()

# Define paths for input and output directories
base_dir = '/data_depth_selection/depth_selection'
train_dir = os.path.join(base_dir, 'test_depth_completion_anonymous')
val_dir = os.path.join(base_dir, 'val_selection_cropped')

# Create ImageDataGenerator for input images with preprocessing function
input_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rescale=1./255.0)
# Create ImageDataGenerator for output images with resizing (if needed)
output_datagen = ImageDataGenerator(rescale=1./255.0)
# Define batch size
batch_size = 8

image_train_generator = input_datagen.flow_from_directory(
    os.path.join(train_dir, 'image'),
    target_size=(224, 224),  # Assuming VGG16 input size
    batch_size=batch_size,
    shuffle    = False,
    class_mode=None  # Since this is an autoencoder, we don't need labels
)

depth_train_generator = output_datagen.flow_from_directory(
    os.path.join(train_dir, 'velodyne_raw'),
    target_size=(224, 224),  # Assuming VGG16 input size
    batch_size=batch_size,
    shuffle    = False,
    class_mode=None  # Since this is an autoencoder, we don't need labels
)

image_val_generator = input_datagen.flow_from_directory(
    os.path.join(val_dir, 'image'),
    target_size=(224, 224),  # Assuming VGG16 input size
    batch_size=batch_size,
    shuffle    = False,
    class_mode=None  # Since this is an autoencoder, we don't need labels
)

depth_val_generator = output_datagen.flow_from_directory(
    os.path.join(val_dir, 'velodyne_raw'),
    target_size=(224, 224),  # Assuming VGG16 input size
    batch_size=batch_size,
    shuffle    = False,
    class_mode=None  # Since this is an autoencoder, we don't need labels
)

# Custom generator to combine two generators into one

class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        assert len(input_gen1) == len(input_gen2), "Input generators must have the same length."
    def __len__(self):
        return min(len(self.gen1), len(self.gen2))
    def __getitem__(self, i):
        x = self.gen1.__getitem__(i)
        y = self.gen2.__getitem__(i)
        return x, y
    def on_epoch_end(self):
        if hasattr(self.gen1, 'on_epoch_end'):
            self.gen1.on_epoch_end()
        if hasattr(self.gen2, 'on_epoch_end'):
            self.gen2.on_epoch_end()


train_generator = JoinedGen(image_train_generator, depth_train_generator)
val_generator = JoinedGen(image_val_generator, depth_val_generator)

conv_base.trainable = False

# Define input shape
input_shape = (224, 224, 3)  # Assuming input images have this shape

# Define input tensor
inputs = layers.Input(shape=input_shape)

# Encoder (using conv_base)
block1_conv1 = conv_base.get_layer('block1_conv1')(inputs)
block1_conv2 = conv_base.get_layer('block1_conv2')(block1_conv1)
block1_pool = conv_base.get_layer('block1_pool')(block1_conv2)
block2_conv1 = conv_base.get_layer('block2_conv1')(block1_pool)
block2_conv2 = conv_base.get_layer('block2_conv2')(block2_conv1)
block2_pool = conv_base.get_layer('block2_pool')(block2_conv2)
block3_conv1 = conv_base.get_layer('block3_conv1')(block2_pool)
block3_conv2 = conv_base.get_layer('block3_conv2')(block3_conv1)
block3_conv3 = conv_base.get_layer('block3_conv3')(block3_conv2)
block3_pool = conv_base.get_layer('block3_pool')(block3_conv3)
block4_conv1 = conv_base.get_layer('block4_conv1')(block3_pool)
block4_conv2 = conv_base.get_layer('block4_conv2')(block4_conv1)
block4_conv3 = conv_base.get_layer('block4_conv3')(block4_conv2)
block4_pool = conv_base.get_layer('block4_pool')(block4_conv3)
block5_conv1 = conv_base.get_layer('block5_conv1')(block4_pool)
block5_conv2 = conv_base.get_layer('block5_conv2')(block5_conv1)
block5_conv3 = conv_base.get_layer('block5_conv3')(block5_conv2)
block5_pool = conv_base.get_layer('block5_pool')(block5_conv3)

# Decoder
block6_conv1 = layers.Conv2DTranspose(512, (3, 3), activation='sigmoid', padding='same', name='block6_conv1')(block5_pool)
block6_upsamp = layers.UpSampling2D((4, 4),name='block6_upsamp')(block6_conv1)
block7_concat = layers.Concatenate(name='block7_concat')([block6_upsamp, block4_conv3])
block7_conv1 = layers.Conv2DTranspose(128, (3, 3), activation='sigmoid', padding='same',name='block7_conv1')(block7_concat)
block7_conv2 = layers.Conv2DTranspose(128, (3, 3), activation='sigmoid', padding='same',name='block7_conv2')(block7_conv1)
block7_upsamp = layers.UpSampling2D((4, 4),name='block7_upsamp')(block7_conv2)
block8_concat = layers.Concatenate(name='block8_concat')([block7_upsamp, block2_conv2])
block8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation='sigmoid', padding='same',name='block8_conv1')(block8_concat)
block8_upsamp = layers.UpSampling2D((2, 2),name='block8_upsamp')(block8_conv1)
decoded_output = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same',name='decoded_output')(block8_upsamp)  # Output image has 3 channels (RGB)

# Create autoencoder model
autoencoder = models.Model(inputs, decoded_output)

# Define SSIM loss
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255.0))

# Compile the model with SSIM loss
autoencoder.compile(optimizer='adam', loss = [ssim_loss])

# Print encoder summary
autoencoder.summary()

discriminator = models.Sequential()
discriminator.add(layers.Conv2D(64, kernel_size=4, strides=2, padding='same'))
discriminator.add(layers.LeakyReLU(negative_slope=0.2))
discriminator.add(layers.Conv2D(128, kernel_size=4, strides=2, padding='same'))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU(negative_slope=0.2))
discriminator.add(layers.Conv2D(256, kernel_size=4, strides=2, padding='same'))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU(negative_slope=0.2))
discriminator.add(layers.Conv2D(512, kernel_size=4, strides=2, padding='same'))
discriminator.add(layers.BatchNormalization())
discriminator.add(layers.LeakyReLU(negative_slope=0.2))
discriminator.add(layers.Flatten())
discriminator.add(layers.Dense(1, activation='sigmoid'))

discriminator.build((None, 224, 224, 6))
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.summary()

# Discriminator summary
discriminator.summary()