import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import tensorflow.keras as keras

import numpy as np
from skimage.transform import resize
import os


from tensorflow.keras import models
from tensorflow.keras import layers
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
base_dir = '/app/data_depth_selection/depth_selection'
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


# Define the decoder part of the autoencoder
autoencoder = models.Sequential()
autoencoder.add(conv_base)
autoencoder.add(layers.UpSampling2D((8, 8)))
autoencoder.add(layers.Conv2DTranspose(256, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Conv2DTranspose(256, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Dense(128, activation='sigmoid'))
autoencoder.add(layers.UpSampling2D((2, 2)))
autoencoder.add(layers.Conv2DTranspose(128, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Conv2DTranspose(128, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Dense(64, activation='sigmoid'))
autoencoder.add(layers.UpSampling2D((2, 2)))
autoencoder.add(layers.Conv2DTranspose(64, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Conv2DTranspose(64, (3, 3), activation='sigmoid', padding='same'))
autoencoder.add(layers.Dense(3, activation='sigmoid'))

# Define SSIM loss
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255.0))

# Compile the model with SSIM loss
autoencoder.compile(optimizer='adam', loss = [ssim_loss])

# Compile the model with Cosine Similarity loss
#autoencoder.compile(optimizer='adam', loss='cosine_similarity')

# Compile the model with Mean Squared Error loss
#autoencoder.compile(optimizer='sgd', loss='mse')  # Mean Squared Error loss for image reconstruction

# Compile the model with Mean Absolute Error loss
#autoencoder.compile(optimizer='adam', loss='mae')  # Mean Squared Error loss for image reconstruction

autoencoder.build((None, 224, 224, 3))

# Print summary
autoencoder.summary()

num_epochs = 20

# Train the autoencoder
steps_per_epoch = len(train_generator) # Number of batches per epoch
validation_steps = len(val_generator)  # Number of batches for validation

autoencoder.fit(train_generator, 
                steps_per_epoch=steps_per_epoch, 
                epochs=num_epochs, 
                validation_data=val_generator, 
                validation_steps=validation_steps)

# Save the trained model
autoencoder.save('/app/convautoencoder.keras')


from PIL import Image

# Load a sample image
sample_image_path = '/app/data_depth_selection/depth_selection/val_selection_cropped/image/1/2011_09_26_drive_0005_sync_image_0000000092_image_02.png'
sample_image = Image.open(sample_image_path)
sample_image = sample_image.resize((224, 224))  # Resize to match model input size
sample_image_array = np.array(sample_image)

# Preprocess the sample image using input_datagen
preprocessed_sample_image = preprocess_input(sample_image_array)

# Expand dimensions to match model input shape
preprocessed_sample_image = np.expand_dims(preprocessed_sample_image, axis=0)

# Predict using the autoencoder model
predicted_output = autoencoder.predict(preprocessed_sample_image)

# Denormalize the output (if needed)
predicted_output = np.clip(predicted_output * 255.0, 0, 255).astype(np.uint8)

# Convert the predicted output to an image
predicted_output_image = Image.fromarray(predicted_output[0]).resize((1216,352))

# Save the predicted output image
predicted_output_image.save("predicted_output.png")

print("Predicted output image saved successfully.")
