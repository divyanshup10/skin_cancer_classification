# skin_cancer_classification

1. Problem Statement:
Skin cancer is a serious and potentially life-threatening condition affecting millions of people worldwide. Early detection and diagnosis play a critical role in effective treatment and improving patient survival rates. However, diagnosing skin cancer can be challenging due to the wide variety of lesion types and their similarities to benign conditions. This project aims to create a robust and user-friendly automated system that leverages deep learning to classify skin lesions, helping healthcare professionals and patients make more informed decisions.
2. Introduction:
Skin cancer encompasses a variety of conditions, from relatively harmless lesions to highly aggressive melanomas. Manual examination and diagnosis of skin lesions often require expertise and are subject to human error and limitations. With the rapid advancements in deep learning, there is an opportunity to create accurate automated systems capable of identifying different types of skin lesions. This project focuses on developing a skin cancer classification system using convolutional neural networks (CNNs) to predict the type of lesion from images and integrating this capability into a user-friendly web application.
3. Objectives:
- To develop a deep learning-based model capable of classifying skin lesions using image data.
- To create a web-based interface for uploading images and receiving real-time predictions.
- To preprocess and augment data for improved model performance and generalization.
4. Software and Tools Used:
The project leveraged several software tools and libraries to develop the deep learning model and web application:
- Programming Language: Python
- Deep Learning Framework: TensorFlow and Keras for building, training, and deploying the model.
- Data Handling: Pandas and NumPy for reading and processing data.
- Image Processing: OpenCV and PIL for loading and transforming images.
- Web Framework: Flask for building the web application.
5. Algorithm: 
The project used a Convolutional Neural Network (CNN) architecture to classify skin lesions. Two different approaches were explored:
- Transfer Learning with MobileNetV2: The MobileNetV2 model was used as a base model. This pre-trained model was fine-tuned for the specific skin lesion classification task by adding a custom classifier on top.
  - Benefits: This approach leverages pre-trained weights from a large dataset, speeding up the training process and improving accuracy by using feature representations learned from general images.
- Custom CNN Model: A sequential model was designed with multiple convolutional layers, pooling layers, dropout layers for regularization, and dense layers.
  - Benefits: Allows for flexibility in architecture design tailored specifically to the dataset.
-Model Training Steps:
- Images were resized to 150x150 pixels, normalized, and passed through a series of layers for feature extraction and classification.
- The model was compiled with the Adam optimizer and categorical cross-entropy loss function.
- Data augmentation was used during training to increase the dataset's variability, making the model more robust to real-world data.
6. Data Collection and Preprocessing:
Data Source: The HAM10000 dataset, containing over 10,000 dermatoscopic images of different types of skin lesions, was used. The dataset included metadata with diagnosis labels.
Steps Taken:
- Reading Metadata: Extracted relevant features such as diagnosis (`dx`) and associated each image with its class label.
- Data Organization: Images were grouped into folders based on their class labels.
- Data Augmentation: Applied transformations like rotations, shifts, zooms, and flips using `ImageDataGenerator` to artificially expand the training dataset.
7. Model Architecture and Training:
Transfer Learning with MobileNetV2:
- Base Model: MobileNetV2 (pre-trained on ImageNet) with the top layers removed.
- Custom Layers: Added a global average pooling layer, dropout layer, and dense layer for classification.
- Freezing Layers: Initial layers of the base model were frozen to retain general feature representations, while custom layers were trained for skin lesion classification.
Custom CNN Model:
- Conv2D and Pooling Layers: Extracted features from images using convolution and pooling.
- Dropout Layers: Added to reduce overfitting.
- Dense Layers: Used for final classification.
8. Web Application Development:
The classification system was integrated into a web application using **Flask** to provide a user-friendly interface for uploading images and displaying predictions. Users can upload a skin lesion image, and the application preprocesses the image, passes it through the trained model, and displays the predicted lesion type.

9. Results:
- Achieved high accuracy on the training data.
- Validation accuracy highlighted the model's ability to generalize to unseen data.

10. Conclusion:
The skin cancer classification system demonstrates the potential of deep learning to assist in early skin cancer diagnosis. While challenges remain, the integration of machine learning into healthcare tools holds promise for improving patient outcomes and reducing diagnostic burdens.

