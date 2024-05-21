# -Skin-Disease-Classification
This project focuses on developing a machine learning model to classify various types of skin diseases using image data. The project leverages the power of Spark for handling large-scale data and Keras for building the neural network.

Table of Contents
Introduction
Project Goals
Tools and Technologies Used
Dataset
Methodology
Results
Repository Structure
Usage
Contributing
License
Introduction
Skin disease classification is a critical task in dermatology, facilitating early detection and treatment planning. This project aims to develop a machine learning model capable of accurately identifying various types of skin diseases from image data.

Project Goals
Build a classification model using deep learning techniques.
Utilize Apache Spark for handling large-scale image datasets.
Implement data augmentation to improve model generalization.
Evaluate model performance on a separate test dataset.
Tools and Technologies Used
Spark: Distributed computing framework for big data processing.
Keras and TensorFlow: Deep learning libraries for building and training neural networks.
PySpark: Python API for Apache Spark, used for data preprocessing and model training.
PIL (Pillow) and NumPy: Libraries for image processing and manipulation.
Scikit-learn: Machine learning library for evaluation metrics and data preprocessing.
Dataset
The dataset consists of images of various skin diseases, sourced from medical databases and repositories. It includes classes such as actinic keratosis, squamous cell carcinoma, basal cell carcinoma, pigmented benign keratosis, nevus, dermatofibroma, vascular lesion, melanoma, and seborrheic keratosis.

Methodology
Data Preparation
Images were preprocessed and resized using the PIL library.
Spark DataFrame was used to manage and process the image data.
Data augmentation techniques such as rotation, shift, shear, zoom, and flip were applied using Keras' ImageDataGenerator.
Model Building
A Multilayer Perceptron (MLP) neural network was chosen for its effectiveness in image classification tasks.
The model architecture included several fully connected layers with dropout for regularization.
The output layer used softmax activation to predict probabilities for each class.
Training and Evaluation
The model was trained on the augmented data using Spark's MLlib.
Training progress was monitored, and learning rate was adjusted using ReduceLROnPlateau callback.
Model performance was evaluated using accuracy metrics on training, validation, and test datasets.
Results
The model achieved an accuracy of X% on the test dataset, demonstrating its effectiveness in classifying skin diseases.
Class-wise accuracies and confusion matrix were analyzed to understand the model's performance across different disease categories.
Repository Structure
css
Copy code
skin-disease-classification/
│
├── data/
│   ├── Train/
│   └── Test/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── train_spark_keras.py
│   ├── evaluate_model.py
│   └── inference.py
│
├── README.md
└── requirements.txt
Usage
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/skin-disease-classification.git
cd skin-disease-classification
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run training script:

bash
Copy code
python src/train_spark_keras.py
Evaluate the model:

bash
Copy code
python src/evaluate_model.py
Run inference on new images:

bash
Copy code
python src/inference.py /path/to/your/image.jpg
Contributing
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

License
This project is licensed under the MIT License - see the LICENSE file for details.
