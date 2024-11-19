import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
# Define a data generator for loading a small batch of images from your validation set

model = load_model('skin_new_model.h5')
datagen = ImageDataGenerator(rescale=1./255)  # Adjust preprocessing if necessary

# Replace with your validation directory path
validation_dir = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2'

# Load a batch of images
batch_size = 100  # Adjust batch size as needed
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),  # Replace with your input image size
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Get a batch of images and labels
images, labels = next(validation_generator)

# Run predictions on the batch
predictions = model.predict(images)

# Print raw predictions and class probabilities
for i, prediction in enumerate(predictions):
    print(f"Image {i+1}:")
    print("Raw Output (Logits or Softmax):", prediction)
    print("Predicted Class:", np.argmax(prediction))
    print("True Class:", np.argmax(labels[i]))
    print()
