
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            Loretta Gray 7.3 Decision Tree Regression Commmeted Hw6
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell initializes Spark and creates a SparkSession.
1. Import findspark to configure PySpark for local environment.
2. Call findspark.init() to initialize Spark settings.
3. Import SparkSession from pyspark.sql.
4. Create a SparkSession using a builder pattern to configure the application name and any necessary settings.
'''

import findspark
findspark.init()

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell reads a CSV file into a DataFrame using Spark.
1. Use spark.read.format('csv') to specify the file format as CSV.
2. Set options for the CSV file:
   - header='true' to treat the first row as the header.
   - inferschema='true' to automatically infer column data types.
3. Load the CSV file from the specified path and store it in a DataFrame 'df'.
'''

df = spark.read.format('csv').\
                       options(header='true', \
                       inferschema='true').\
            load("file:///Users/ellegreyllc/Desktop/Advertising.csv",header=True)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell displays the first 5 rows of the DataFrame and prints its schema.
1. df.show(5, True) shows the first 5 rows of the DataFrame, including column names and data types.
2. df.printSchema() prints the schema of the DataFrame, showing column names and inferred data types.
'''

df.show(5,True)
df.printSchema()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell provides summary statistics for the DataFrame.
1. df.describe() generates summary statistics for all numeric columns (e.g., count, mean, standard deviation, min, max).
2. .show() displays the summary statistics in the output.
'''

df.describe().show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Convert the data to dense vector (features and label)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell transforms the DataFrame into a format suitable for machine learning.
1. Import Row from pyspark.sql and Vectors from pyspark.ml.linalg.
2. Convert the DataFrame to an RDD and use map() to transform each row:
   - r[-1] extracts the last column as the label.
   - Vectors.dense(r[:-1]) converts the remaining columns into a dense feature vector.
3. Convert the transformed RDD back to a DataFrame with columns 'label' and 'features'.
4. Display the first 5 rows of the transformed DataFrame.
'''

from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

transformed = df.rdd.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).toDF(['label','features'])
transformed.show(5)


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #You will find out that all of the machine learning algorithms in Spark are based on
#the features and label. That is to say, you can play with all of the machine learning
#algorithms in Spark when you get ready the features and label.

#Deal with categorical variables
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell sets up a feature indexer for machine learning and applies it to the DataFrame.
1. Import necessary modules for pipeline, regression, and evaluation.
2. VectorIndexer automatically identifies categorical features and indexes them:
   - inputCol specifies the input column with features.
   - outputCol specifies the output column where the indexed features will be stored.
   - maxCategories=4 ensures that features with more than 4 distinct values are treated as continuous.
3. Fit the feature indexer to the transformed data and transform it to create indexed features.
'''

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4
# distinct values are treated as continuous.

featureIndexer = VectorIndexer(inputCol="features", \
                               outputCol="indexedFeatures",\
                               maxCategories=4).fit(transformed)

data = featureIndexer.transform(transformed)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell displays the first 5 rows of the transformed DataFrame with indexed features.
1. data.show(5) shows the first 5 rows of the DataFrame with the indexed features, allowing you to check how the transformation has been applied.
'''

data.show(5)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Fit Decision Tree Regression Model
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell sets up a DecisionTreeRegressor for training the model.
1. Import DecisionTreeRegressor from pyspark.ml.regression.
2. Instantiate a DecisionTreeRegressor model, specifying the input column for features (indexedFeatures) that will be used for training.
'''

from pyspark.ml.regression import DecisionTreeRegressor

# Train a DecisionTree model.
dt = DecisionTreeRegressor(featuresCol="indexedFeatures")

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell splits the data into training and test datasets.
1. Use data.randomSplit() to randomly split the data into training (80%) and test (20%) datasets.
2. The seed ensures the split is reproducible.
3. Show the first 5 rows of the training dataset to verify the split.
'''

# split data into training and test datasets
trainingData, testData = data.randomSplit([0.8, 0.2], seed=1234)
trainingData.show(5)


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Pipeline Architecture
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell chains the feature indexer and decision tree into a Pipeline and trains the model.
1. Create a Pipeline by specifying the stages:
   - featureIndexer: the feature indexing step.
   - dt: the decision tree regression model.
2. Fit the pipeline to the training data to create the model.
'''

# Chain indexer and decision tree in a Pipeline
pipeline = Pipeline(stages=[featureIndexer, dt])

model = pipeline.fit(trainingData)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Make predictions
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell makes predictions using the trained model on the test data.
1. Use model.transform(testData) to apply the trained pipeline model to the test dataset.
2. This will generate predictions, which include the predicted label and features.
'''

# Make predictions.
predictions = model.transform(testData)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell selects and displays specific columns from the predictions.
1. Use predictions.select() to select the "features", "label", and "prediction" columns.
2. .show(5) displays the first 5 rows of the selected columns to review the predictions.
'''

# Select example rows to display.
predictions.select("features","label","prediction").show(5)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Evaluation
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell evaluates the model's performance using the Root Mean Squared Error (RMSE).
1. Import RegressionEvaluator from pyspark.ml.evaluation.
2. Instantiate a RegressionEvaluator to compute the RMSE:
   - labelCol specifies the column containing the true labels.
   - predictionCol specifies the column containing the predicted labels.
   - metricName="rmse" tells the evaluator to compute RMSE.
3. Use evaluator.evaluate(predictions) to calculate the RMSE based on the predictions.
'''

# from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")

rmse = evaluator.evaluate(predictions)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell calculates the R-squared (R²) score to evaluate the model's performance.
1. Convert the "label" and "prediction" columns of the predictions DataFrame into pandas DataFrames (y_true and y_pred).
2. Import sklearn.metrics to use the r2_score function.
3. Use sklearn.metrics.r2_score() to compute the R² score based on true labels (y_true) and predicted values (y_pred).
4. Print the R² score.
'''

y_true = predictions.select("label").toPandas()
y_pred = predictions.select("prediction").toPandas()

import sklearn.metrics
r2_score = sklearn.metrics.r2_score(y_true, y_pred)
print('r2_score: {0}'.format(r2_score))

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This cell extracts the feature importances from the trained model.
1. Access the last stage of the pipeline (the decision tree model) using model.stages[-1].
2. Use the .featureImportances attribute to retrieve the importance of each feature used by the model.
'''

model.stages[-1].featureImportances

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    