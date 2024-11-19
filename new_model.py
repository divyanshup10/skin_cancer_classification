import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import shutil
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
# Add this import at the top of your script
from sklearn.utils import class_weight

from tensorflow.keras.losses import SparseCategoricalCrossentropy   

import os
metadata_path = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_metadata.csv'  # Path to metadata CSV
image_dir_1 = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_1'  # Training images directory
image_dir_2 = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2'  

# Read metadata CSV
metadata = pd.read_csv(metadata_path)

# Ensure the directory for each class exists
for label in metadata['dx'].unique():
    class_dir_1 = os.path.join(image_dir_1, label)
    class_dir_2 = os.path.join(image_dir_2, label)
    os.makedirs(class_dir_1, exist_ok=True)
    os.makedirs(class_dir_2, exist_ok=True)


for _, row in metadata.iterrows():
    file_name = row['image_id'] + '.jpg'  # Ensure your images have the correct extension
    class_label = row['dx']

    src_path_1 = os.path.join(image_dir_1, file_name)
    src_path_2 = os.path.join(image_dir_2, file_name)

   
    if os.path.exists(src_path_1):
        dst_path_1 = os.path.join(image_dir_1, class_label, file_name)
        shutil.move(src_path_1, dst_path_1)
    
   
    elif os.path.exists(src_path_2):
        dst_path_2 = os.path.join(image_dir_2, class_label, file_name)
        shutil.move(src_path_2, dst_path_2)

print("Images successfully organized into class-specific folders.")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
   
     r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2',
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

validation_generator = test_datagen.flow_from_directory(
   r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_1',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Compute class weights for the training data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

y_train = train_generator.classes  
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
y_train_classes = np.argmax(y_train, axis=1)  
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_classes), y=y_train_classes)

class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers


# custom layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  
output = Dense(7, activation='softmax')(x)

model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # Adjust to the number of classes in your dataset
])

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=156, 
    epochs=15,  
    validation_data=validation_generator,
    validation_steps=50,  
    class_weight=class_weights_dict  
)

# Save the trained model
model.save("skin_new_model.h5")
print("Model saved successfully!")

# Load the model when needed
model = load_model("skin_new_model.h5")
print("Model loaded successfully!")

    
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(validation_generator, steps=50)
print(f"Test accuracy: {test_acc}")

# Predict on validation data
predictions = model.predict(validation_generator, steps=50)
predicted_classes = np.argmax(predictions, axis=1)

# Print classification report
print(classification_report(validation_generator.classes, predicted_classes))

# Print confusion matrix
conf_matrix = confusion_matrix(validation_generator.classes, predicted_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

