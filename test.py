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
from tensorflow.keras.losses import SparseCategoricalCrossentropy   

# from tensorflow.keras.applications import MobileNetV2
# data_dir = 'C:\Users\om\Desktop\ml project\skin_cancer\HAM10000_images_part_2'
# data_dir = 'C:\\Users\\om\\Desktop\\ml project\\skin_cancer\\HAM10000_images_part_2'
import os

metadata_path = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_metadata.csv'  # Path to metadata CSV
image_dir_1 = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_1'  # Training images directory
image_dir_2 = r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2'  # Validation images directory

# Read metadata CSV
metadata = pd.read_csv(metadata_path)

# Ensure the directory for each class exists
for label in metadata['dx'].unique():
    class_dir_1 = os.path.join(image_dir_1, label)
    class_dir_2 = os.path.join(image_dir_2, label)
    os.makedirs(class_dir_1, exist_ok=True)
    os.makedirs(class_dir_2, exist_ok=True)

# Move images to class-specific folders in both directories
for _, row in metadata.iterrows():
    file_name = row['image_id'] + '.jpg'  # Ensure your images have the correct extension
    class_label = row['dx']

    # Check and move the file if it exists in either directory
    src_path_1 = os.path.join(image_dir_1, file_name)
    src_path_2 = os.path.join(image_dir_2, file_name)

    # Move image in the training directory if it exists there
    if os.path.exists(src_path_1):
        dst_path_1 = os.path.join(image_dir_1, class_label, file_name)
        shutil.move(src_path_1, dst_path_1)
    
    # Move image in the validation directory if it exists there
    elif os.path.exists(src_path_2):
        dst_path_2 = os.path.join(image_dir_2, class_label, file_name)
        shutil.move(src_path_2, dst_path_2)

print("Images successfully organized into class-specific folders.")

# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
# Apply data augmentation to training images
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load training and testing data (make sure your paths are correct)
train_generator = train_datagen.flow_from_directory(
    # r'C:\Users\om\Desktop\ml project\skin_cancer\HAM10000_images_part_1',
     r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2',
    target_size=(150, 150),
    batch_size=32,
    class_mode='sparse'
)

validation_generator = test_datagen.flow_from_directory(

    # r'C:\Users\om\Desktop\ml project\skin_cancer\HAM10000_images_part_2',
      r'C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_1',
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)

# Compute class weights for the training data
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Build the model
# For multi-class classification, adjust the model and loss function
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(7, activation='softmax')  # Adjust output units for multi-class
# ])
base_model = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
output = Dense(7, activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=output)

# Compile the model with SparseCategoricalCrossentropy loss
# model.compile(optimizer=Adam(learning_rate=0.0001),
#               loss=SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

# # Use EarlyStopping to avoid overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # Fit the model with class_weight
# history = model.fit(
#     train_generator,
#     epochs=30,
#     validation_data=validation_generator,
#     class_weight=class_weights_dict,
#     callbacks=[early_stopping]
# )

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile model with categorical_crossentropy for multi-class classification
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Get predictions for validation set
y_pred = np.argmax(model.predict(validation_generator), axis=1)
y_true =validation_generator.classes
# Compile the model
model.add(Dense(7, activation='softmax'))  # where num_classes is the total number of classes
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=10,  # Adjust based on your dataset size
    epochs=30,  # You can change the number of epochs as needed
    validation_data=validation_generator,
    validation_steps=50  # Adjust based on your validation dataset size
)

# Save the trained model
model.save("skin_cancer_model.h5")
print("Model saved successfully!")

# Load the model when needed
model = load_model("skin_cancer_model.h5")
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

