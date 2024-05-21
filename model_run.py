from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import BinaryType
from PIL import Image
import numpy as np
from io import BytesIO

# Step 2: Create a Spark session
spark = SparkSession.builder.appName("MLPModelInference").getOrCreate()

# Step 3: Load the pre-trained Multilayer Perceptron classification model
model_path = "/home/vamshi/Desktop/experiment/skin/finalproject/output/stages/model"  # Replace with the actual path to your model
model = MultilayerPerceptronClassificationModel.load(model_path)

# Step 4: Prepare input data (replace this with your actual image loading and pre-processing code)

# Sample image path
image_path = "/home/vamshi/Desktop/experiment/skin/finalproject/Test/actinic keratosis/ISIC_0025368.jpg"

# Read the image and convert it to byte data
with open(image_path, "rb") as f:
    image_bytes = f.read()

# Define a UDF (User Defined Function) to process and convert the image
def process_image_udf(image_bytes):
    # Convert the byte data to a PIL Image
    image = Image.open(BytesIO(image_bytes))
    
    # Resize the image to your desired dimensions
    resized_image = image.resize((100, 75))
    
    # Convert the resized image to a NumPy array
    resized_array = np.array(resized_image)
    
    # Flatten the array and convert it to bytes
    flattened_bytes = resized_array.flatten().tobytes()
    
    return flattened_bytes

# Register the UDF with PySpark
process_image = udf(process_image_udf, BinaryType())

# Apply the UDF to process the image
df_input = spark.createDataFrame([(image_bytes,)], ["input_image"])
df_processed = df_input.withColumn("processed_image_data", process_image("input_image"))

# Convert processed image bytes to a vector
binary_to_vector_udf = udf(lambda binary_data: Vectors.dense(np.frombuffer(binary_data, dtype=np.uint8)), VectorUDT())
df_processed_vector = df_processed.withColumn("features", binary_to_vector_udf("processed_image_data"))

# Step 5: Apply the model to the input data
predictions = model.transform(df_processed_vector)

# Step 6: Extract and display the predictions (replace this with your actual result processing code)
selected_columns = ["prediction", "rawPrediction","features"]  # Adjust as needed
selected_predictions = predictions.select(selected_columns)
selected_predictions.show()

# Get the class name corresponding to the predicted label
label_mapping = {0: "actinic keratosis",1:"squamous cell carcinoma",2:"basal cell carcinoma",3:"pigmented benign keratosis",4:"nevus",5:"dermatofibroma",6:"vascular lesion",7:"melanoma",8:"seborrheic keratosis"}  # Replace with your label mapping
get_class_name_udf = udf(lambda prediction: label_mapping[prediction], "string")
result_df = selected_predictions.withColumn("class_name", get_class_name_udf("prediction"))

# Display the resulting DataFrame
result_df.show()

# Step 7: Stop the Spark session
spark.stop()

