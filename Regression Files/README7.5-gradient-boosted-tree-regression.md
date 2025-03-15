
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Loretta Gray 7.5 Gradient Boosted Tree Regression Commented Hw6
'''
Gradient Boosted Trees (GBT):

Modeling Approach: GBT builds trees sequentially, where each tree tries to correct
the errors of the previous one. It focuses on the residuals (errors) left by prior
trees.
Ensemble Method: Boosting is used, which means trees are added one at a time, and
each subsequent tree minimizes the errors of the previous ones.
Strength: Often leads to higher accuracy but is more prone to overfitting,
especially with noisy data.
'''
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code initializes a Spark session for a PySpark application. It starts by setting
up the environment for Spark with the findspark.init() method, which makes the Spark
library accessible. Then, it imports the necessary module to create a Spark session,
which is the entry point for working with PySpark. The session is configured with a
custom application name ("Python Spark GBTRegressor example") and an optional
configuration setting. Finally, it creates or retrieves an existing Spark session. '''

import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark GBTRegressor example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code reads a CSV file ("Advertising.csv") into a DataFrame with inferred
schema and header options enabled. It then displays the first 5 rows of the DataFrame
and prints its schema. '''

df = spark.read.format('csv') \
    .options(header='true', inferschema='true') \
    .load("file:///Users/ellegreyllc/Desktop/Advertising.csv")

df.show(5, True)
df.printSchema()


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df.describe().show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Convert the data to dense vector (features and label)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code transforms the DataFrame by converting each row into a feature
vector (excluding the last column) and assigns the last column as the label.
It then creates a new DataFrame with two columns: 'features' (containing the
feature vector) and 'label' (containing the target variable). Finally, it
displays the first 5 rows of the transformed DataFrame. '''

from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

transformed=df.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])
transformed.show(5)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Deal with the Categorical variables
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code applies a VectorIndexer to the 'features' column, which is used to
index categorical features (if any) in the dataset. It creates a new column,
'indexedFeatures', to store the indexed features. The maxCategories parameter
is set to 4, meaning any feature with more than 4 distinct categories will be
treated as continuous. The transformed data is then displayed with the
first 5 rows. '''

from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

featureIndexer = VectorIndexer(inputCol="features", \
                               outputCol="indexedFeatures",\
                               maxCategories=4).fit(transformed)

data = featureIndexer.transform(transformed)
data.show(5,True)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Split the data into training and test sets (40% held out for testing)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code splits the data into two subsets: 60% for training (trainingData) and 40%
for testing (testData). It then displays the first 5 rows of both the training and test datasets. '''

# Split the data into training and test sets (40% held out for testing)
(trainingData, testData) = data.randomSplit([0.6, 0.4])

trainingData.show(5)
testData.show(5)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#Fit RandomForest Regression Model with GBTRegressor
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            ''' This code imports the GBTRegressor class and initializes a Gradient Boosted Tree (GBT)
regression model. The model can be customized with parameters like numTrees, maxDepth, and seed,
though they are not specified here. '''

# Import LinearRegression class
from pyspark.ml.regression import GBTRegressor

# Define LinearRegression algorithm
gbt = GBTRegressor() #numTrees=2, maxDepth=2, seed=42
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #If you decide to use the indexedFeatures features, you need to add the parameter
#featuresCol="indexedFeatures".

#Pipeline Architecture
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # This code creates a machine learning pipeline by chaining the featureIndexer and the GBTRegressor model as stages.
# The pipeline is then fitted on the trainingData to create a trained model.
pipeline = Pipeline(stages=[featureIndexer, gbt])
model = pipeline.fit(trainingData)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Make predictions
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # This code applies the trained model to the testData to make predictions.
# It then selects and displays the 'features', 'label', and 'prediction' columns for the first 5 rows.
predictions = model.transform(testData)
predictions.select("features","label", "prediction").show(5)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Evaluation
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # This code uses a RegressionEvaluator to compute the R-squared (R2) metric,
# which measures the model's goodness of fit on the test data by comparing predictions with actual labels.
# It then prints the R-squared value.
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="r2")
print("R Squared (R2) on test data = %g" % evaluator.evaluate(predictions))

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # This code retrieves and displays the feature importances from the trained Gradient Boosted Trees model.
# Feature importances represent the relative importance of each feature in making predictions.
importances = model.stages[-1].featureImportances
print("Feature Importances: ", importances)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # This code retrieves the individual decision trees from the trained Gradient Boosted Trees model.
# The `model.stages[-1]` accesses the GBT model, and `.trees` returns a list of all the decision trees that were trained.
trees = model.stages[-1].trees
for tree in trees:
    print("Trained Decision Tree: \n", tree)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
'''I wanted to visualize the data for a Gradient Boosted Trees (GBT) regression model, you can use the
following three different types of plots:

1. Prediction vs. Actual Plot
This plot compares the predicted values versus the actual values from the test set, which shows how
well the model's predictions match the real data.
'''
# Extracting predictions and actual values
predicted_values = predictions.select("prediction").rdd.flatMap(lambda x: x).collect()
actual_values = predictions.select("label").rdd.flatMap(lambda x: x).collect()

# Plotting prediction vs actual
plt.figure(figsize=(8,8))
plt.scatter(actual_values, predicted_values, alpha=0.5)
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs Actual for GBT Model')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
2. Feature Importance Plot
This plot shows the relative importance of each feature in the GBT model, helping to identify which
features are contributing most to the model's predictions.
'''
import matplotlib.pyplot as plt
import numpy as np

# Feature importances
importances = model.stages[-1].featureImportances
features = transformed.columns[:-1]

# Plotting the feature importances
plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in GBT Model')
plt.show()


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    