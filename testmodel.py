# import os
# import numpy as np
# import cv2
# import pandas as pd
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt

# # Load the trained model
# model = load_model("skin_new_model.h5")
# # directory_path=r'C:\Users\om\Desktop\skin_cancer\data\testimages'
# # Dictionary to map class indices to actual labels
# class_labels = {0: 'Type1', 1: 'Type2', 2: 'Type3', 3: 'Type4', 4: 'Type5', 5: 'Type6', 6: 'Type7'}

# def predict_bulk_images(directory_path, model, class_labels, output_csv="predictions.csv"):
#     """
#     Predict on all images in a directory and save results to a CSV file.
#     """
#     results = []

#     # Iterate through all image files in the directory
#     for file_name in os.listdir(directory_path):
#         file_path = os.path.join(directory_path, file_name)

#         # Check if the file is an image
#         if file_name.endswith(('.png', '.jpg', '.jpeg')):
#             try:
#                 # Load and preprocess the image
#                 image = cv2.imread(file_path)
#                 image_resized = cv2.resize(image, (150, 150))
#                 image_normalized = image_resized / 255.0
#                 image_input = np.expand_dims(image_normalized, axis=0)
                
#                 # Predict using the model
#                 predictions = model.predict(image_input)
#                 predicted_class = np.argmax(predictions, axis=1)[0]
#                 confidence = predictions[0][predicted_class]
                
#                 # Append the results
#                 results.append({
#                     "File Name": file_name,
#                     "Predicted Label": class_labels[predicted_class],
#                     "Confidence": f"{confidence * 100:.2f}%"
#                 })
#             except Exception as e:
#                 print(f"Error processing file {file_name}: {e}")
#                 continue

#     # Save results to a CSV file
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_csv, index=False)
#     print(f"Predictions saved to {output_csv}")
  
#  # Directory containing test images
# test_images_directory = r"C:\Users\om\Desktop\skin_cancer\data\testimages"  # Replace with the path to your test images directory

# # Run bulk testing
# predict_bulk_images(test_images_directory, model, class_labels, output_csv="bulk_predictions.csv")
import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model("skin_new_model.h5")

# Dictionary to map class indices to actual labels
class_labels = {0: 'Type1', 1: 'Type2', 2: 'Type3', 3: 'Type4', 4: 'Type5', 5: 'Type6', 6: 'Type7'}

def predict_bulk_images(directory_path, model, class_labels, output_csv="predictions.csv"):
    """
    Predict on all images in a directory and save results to a CSV file.
    """
    results = []

    # Iterate through all image files in the directory
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        # Check if the file is an image
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Load and preprocess the image
                image = cv2.imread(file_path)
                image_resized = cv2.resize(image, (150, 150))
                image_normalized = image_resized / 255.0
                image_input = np.expand_dims(image_normalized, axis=0)
                
                # Predict using the model
                predictions = model.predict(image_input)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = predictions[0][predicted_class]
                
                # Append the results
                results.append({
                    "File Name": file_name,
                    "Predicted Label": class_labels[predicted_class],
                    "Confidence": f"{confidence * 100:.2f}%"
                })
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")
                continue

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Directory containing test images
test_images_directory = r"C:\Users\om\Desktop\skin_cancer\data\HAM10000_images_part_2\AKIEC"  # Replace with the path to your test images directory

# Run bulk testing
predict_bulk_images(test_images_directory, model, class_labels, output_csv="bulk_predictions.csv")
