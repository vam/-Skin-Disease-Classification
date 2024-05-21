# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, monotonically_increasing_id, explode
from pyspark import SparkConf
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType, StringType, BinaryType
from io import BytesIO
import tensorflow as tf
from keras.models import save_model

# Set Spark configurations
conf = SparkConf().setAppName("YourAppName").set("spark.executor.memory", "8g")

# Create a Spark session
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Define the directories
train_dir = '/home/vamshi/Desktop/Big data/experiment/skin/finalproject/Train'
test_dir = '/home/vamshi/Desktop/Big data/experiment/skin/finalproject/Test'

# Load train data into a PySpark DataFrame
train_data = []
for label, directory in enumerate(os.listdir(train_dir)):
    if not directory.startswith('.'):
        for filename in os.listdir(os.path.join(train_dir, directory)):
            image_path = os.path.join(train_dir, directory, filename)
            train_data.append((image_path, label))

train_schema = ["image_path", "label"]
train_df = spark.createDataFrame(train_data, train_schema)

# Load test data into a PySpark DataFrame
test_data = []
for label, directory in enumerate(os.listdir(test_dir)):
    if not directory.startswith('.'):
        for filename in os.listdir(os.path.join(test_dir, directory)):
            image_path = os.path.join(test_dir, directory, filename)
            test_data.append((image_path, label))

test_schema = ["image_path", "label"]
test_df = spark.createDataFrame(test_data, test_schema)

# Display the DataFrames
train_df.show()
test_df.show()

# Combine train and test DataFrames
df = train_df.union(test_df)
df.show()

# Create a label map for reference
labels = [directory for directory in os.listdir(train_dir) if not directory.startswith('.')]
label_data = [(label_id, label) for label_id, label in enumerate(labels)]
label_schema = ["label_id", "label"]
label_df = spark.createDataFrame(label_data, label_schema)
label_df = label_df.withColumn("key", monotonically_increasing_id())
label_map = label_df.select("label_id", "label").rdd.collectAsMap()
print(label_map)


# Check for available GPU and set memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as ex:
    print(ex)

# Get the number of available CPU cores
max_workers = spark._jsc.sc().getExecutorMemoryStatus().size()
print("Number of available CPU cores:", max_workers)

# Define a UDF to resize image arrays
def resize_image_array_udf(image_path):
    image = Image.open(image_path).resize((100, 75))
    byte_io = BytesIO()
    image.save(byte_io, format='JPEG')
    return byte_io.getvalue()

# Register the UDF with PySpark
resize_udf = spark.udf.register("resize_image_array", resize_image_array_udf, BinaryType())

# Use the UDF to resize images in the DataFrame
resized_df = df.withColumn("resized_image", resize_udf(col("image_path")))
resized_df.show()

# Get the number of classes
num_classes = resized_df.select('label').distinct().count()

# Group by 'label' and count the number of occurrences
class_counts = df.groupBy("label").count().orderBy("label")

# Display the dataset summary
print("Dataset Summary")
print("-" * 60)
print(f"{'Class Label':<15} {'Class Name':<30} {'Count':<10}")
print("-" * 60)

for class_label, class_name in label_map.items():
    count = class_counts.filter(col("label") == class_label).select("count").first()[0]
    print(f"{class_label:<15} {class_name:<30} {count:<10}")

print("-" * 60)
total_count = class_counts.selectExpr("sum(count) as total").first()["total"]
print(f"{'Total':<45} {total_count:<10}")

# Define a UDF to apply data augmentation
def augment_image_udf(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image_array = np.array(image)
    image_array = image_array.reshape((1,) + image_array.shape)
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    augmented_images = []
    for _ in range(15):
        for batch in datagen.flow(image_array, batch_size=1):
            augmented_images.append(batch[0])
            break
    augmented_bytes = [BytesIO(image.astype(np.uint8).tobytes()).getvalue() for image in augmented_images]
    return augmented_bytes

# Register the UDF with PySpark
augment_udf = spark.udf.register("augment_image", augment_image_udf, ArrayType(BinaryType()))

# Use the UDF to apply data augmentation on images in the DataFrame
augmented_df = resized_df.withColumn("augmented_images", augment_udf(col("resized_image")))
augmented_df = augmented_df.withColumn("augmented_image", explode(col("augmented_images")))
augmented_df = augmented_df.drop("resized_image", "augmented_images")
augmented_df.show()

# Group by 'label' and count the number of occurrences after data augmentation
class_counts_augmented = augmented_df.groupBy("label").count().orderBy("label")

# Display the dataset summary after data augmentation
print("Dataset Summary after Data Augmentation")
print("-" * 60)
print(f"{'Class Label':<15} {'Class Name':<30} {'Count':<10}")
print("-" * 60)

for class_label, class_name in label_map.items():
    count = class_counts_augmented.filter(col("label") == class_label).select("count").first()[0]
    print(f"{class_label:<15} {class_name:<30} {count:<10}")

print("-" * 60)
total_count_augmented = class_counts_augmented.selectExpr("sum(count) as total").first()["total"]
print(f"{'Total':<45} {total_count_augmented:<10}")

# Define a UDF to convert binary data to a vector
binary_to_vector_udf = udf(lambda binary_data: Vectors.dense(np.frombuffer(binary_data, dtype=np.uint8)), VectorUDT())

# Apply the UDF to the augmented_df DataFrame
augmented_df = augmented_df.withColumn("augmented_vector", binary_to_vector_udf("augmented_image"))

# Split the data into training, testing, and validation sets
train_data, test_data, validation_data = augmented_df.randomSplit([0.7, 0.15, 0.15], seed=42)

# Define the VectorAssembler to combine features into a single "features" column
vector_assembler = VectorAssembler(inputCols=["augmented_vector"], outputCol="features")

# Apply the VectorAssembler to the training data
train_data = vector_assembler.transform(train_data)

# Apply the VectorAssembler to the test data
test_data = vector_assembler.transform(test_data)

# Apply the VectorAssembler to the validation data
validation_data = vector_assembler.transform(validation_data)

# Select only the necessary columns for training
train_data = train_data.select("features", "label")

# Select only the necessary columns for testing
test_data = test_data.select("features", "label")

# Select only the necessary columns for validation
validation_data = validation_data.select("features", "label")

# Define the neural network model
layers = [len(augmented_df.first()["augmented_vector"]), 512, 216, 128, 64, num_classes]
mlp = MultilayerPerceptronClassifier(layers=layers, blockSize=128, seed=42, featuresCol="features", labelCol="label")

# Set up the pipeline
pipeline = Pipeline(stages=[mlp])

# Train the model for multiple epochs
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Training Epoch {epoch + 1}/{num_epochs}")
    model = pipeline.fit(train_data)
    train_predictions = model.transform(train_data)
    val_predictions = model.transform(validation_data)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    train_accuracy = evaluator.evaluate(train_predictions)
    print(f"Training Accuracy: {train_accuracy}")
    val_accuracy = evaluator.evaluate(val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

# Evaluate the model on the test set
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(model.transform(test_data))
print(f"Test Accuracy: {accuracy}")





# Save the trained model in .h5 format
model_output_path = "/home/vamshi/Desktop/experiment/skin/finalproject/output"
model.save(model_output_path)

# Stop the Spark session
spark.stop()




