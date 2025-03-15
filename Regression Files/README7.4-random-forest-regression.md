
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Set up spark context and SparkSession</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            Loretta Gray 7.4 Random Forrest Regression Commented with visuals Hw6
'''
Random Forest:

Modeling Approach: Random Forest builds many decision trees in parallel (independently).
It then averages the predictions of all trees to get the final output.
Ensemble Method: Bagging (Bootstrap Aggregating) is used, where each tree is trained on
a random subset of data with replacement, and the final prediction is an average of the
individual trees' predictions.
Strength: Good for reducing variance and avoiding overfitting. It tends to be more robust
to noisy data.
'''
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code initializes a Spark session in PySpark, which is essential
for working with Spark's DataFrame and SQL functionalities. It begins
by importing the findspark module and initializing it to locate the
Spark installation. Then, it imports SparkSession from pyspark.sql
and creates a new Spark session with the application name "Python
Spark RandomForest Regression example" and a placeholder configuration
option. The getOrCreate() method ensures that an existing Spark session is used if available; otherwise, it creates a
new one
'''

import findspark
findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark RandomForest Regression example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code reads a CSV file named "Advertising.csv" into a PySpark
DataFrame, specifying that the first row contains headers and that
Spark should infer the data types of each column. The df.show(5, True)
command displays the first five rows of the DataFrame along with the full
content of each column. The df.printSchema() command outputs the schema
of the DataFrame, showing the column names and their respective data
types.
'''
df = spark.read.format('csv') \
    .options(header='true', inferschema='true') \
    .load("file:///Users/ellegreyllc/Desktop/Advertising.csv")

df.show(5, True)
df.printSchema()


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
The code df.describe().show() generates summary statistics for all numerical columns in the DataFrame.
It computes metrics like count, mean, standard deviation, minimum, and maximum values for each column.
The .describe() function returns a new DataFrame containing these statistics, and .show() displays the
results in a readable format. This helps in understanding the distribution and characteristics of the
dataset.
'''
df.describe().show()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Convert the data to dense vector (features and label)</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code transforms the DataFrame into a format suitable for machine learning in PySpark. It uses the
Resilient Distributed Dataset (RDD) API to map each row, converting all but the last column into a
feature vector (Vectors.dense(r[:-1])) and treating the last column as the label (r[-1]). The
transformed data is then converted back into a DataFrame with two columns: 'features' (containing
the vectorized inputs) and 'label' (the target variable). Finally, transformed.show(5) displays
the first five rows.
'''
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
transformed=df.rdd.map(lambda r: [Vectors.dense(r[:-1]),r[-1]]).toDF(['features','label'])
transformed.show(5)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Deal with the Categorical variables, even they are numeric, if a feature columns has no more than 4 distinct values, it will be considered categorical and will be indexed to improve training model.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code uses VectorIndexer to automatically identify and index categorical features in the
'features' column, creating a new 'indexedFeatures' column. It then transforms the dataset to
prepare it for regression modeling. Finally, it displays the first five rows of the updated
DataFrame.
'''
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
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
            # Import LinearRegression class
'''
This code initializes a RandomForestRegressor model in PySpark, using the 'indexedFeatures' column
as input. It prepares the model for training, with optional parameters like the number of trees,
max depth, and random seed.
'''
from pyspark.ml.regression import RandomForestRegressor

# Define LinearRegression algorithm
rf = RandomForestRegressor(featuresCol="indexedFeatures") # featuresCol="indexedFeatures",numTrees=2, maxDepth=2, seed=42

            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Split the data into training and test sets (40% held out for testing)</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code splits the dataset into training (60%) and testing (40%) subsets using randomSplit().
It then displays the first five rows of both trainingData and testData to verify the split.
'''
(trainingData, testData) = data.randomSplit([0.6, 0.4])

trainingData.show(5)
testData.show(5)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Fit RandomForest Regression Model

If you decide to use the indexedFeatures features, you need to add the parameter featuresCol="indexedFeatures".

Pipeline Architecture</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code creates a Pipeline with the RandomForestRegressor as the only stage and fits it to the
trainingData. The resulting model is trained on the training dataset.
'''
pipeline = Pipeline(stages=[rf])
model = pipeline.fit(trainingData)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Make test predictions with testdata</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code uses the trained model to make predictions on the testData.
It then selects and displays the 'indexedFeatures', 'label', and 'prediction' columns for
the first five rows of the predicted data.
'''
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("indexedFeatures","label", "prediction").show(5)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Evaluation</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Select (prediction, true label) and compute test error
'''
This code initializes a RegressionEvaluator to calculate the Root Mean Squared Error (RMSE) between
the actual 'label' and predicted 'prediction' values. It then evaluates the model's performance on
the test data and prints the RMSE.
'''

evaluator = RegressionEvaluator(
    labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code initializes a RegressionEvaluator to calculate the R-squared (R2) value, which measures
how well the model's predictions match the actual labels. It then evaluates the model on the test
data and prints the R2 value.
'''
evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
print("R Squared (R2) on test data = %g" % evaluator.evaluate(predictions))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code returns the type of the last stage in the Pipeline (which is the trained model,
RandomForestRegressor). It shows the class type of the model, confirming that it's a
RandomForestRegressor.
'

type(model.stages[-1])
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''This code retrieves the feature importances from the trained RandomForestRegressor model,
which indicates the relative importance of each feature in making predictions. The output is
typically a vector of values corresponding to each feature, with higher values indicating greater
importance.
'''
model.stages[-1].featureImportances
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Feature importances, there are 3 features, index 0 is the root had highest feature importance value, 0.4736, .... This means, advertising on TV is most important feature</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Show all decision trees in the random forest</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code retrieves the individual decision trees from the trained RandomForestRegressor model.
It returns a list of trees used in the random forest, allowing you to inspect each tree's structure
and splits.
'''

model.stages[-1].trees
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
This code creates a RegressionEvaluator to calculate the R-squared (R2) value for the predictions,
comparing the actual 'label' with the predicted 'prediction' values. It then evaluates and prints
the R2 value, indicating the model's goodness of fit.
'''

rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(predictions))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''

rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="rmse")
print("(RMSE) on test data = %g" % rf_evaluator.evaluate(predictions))
'''

rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="rmse")
print("(RMSE) on test data = %g" % rf_evaluator.evaluate(predictions))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
I wanted to see a visual representation of what the data looks like:

Residual Plot: This code creates a scatter plot showing the residuals (errors) of the model.
It compares the predicted values with the actual values, helping to identify any patterns in
the model's errors, where a random spread of points around zero suggests a good fit.
'''
import matplotlib.pyplot as plt

# Get the residuals (actual - predicted)
residuals = predictions.select("label", "prediction") \
    .rdd.map(lambda row: row['label'] - row['prediction']).collect()

# Get predicted values
predicted_values = predictions.select("prediction").rdd.map(lambda row: row['prediction']).collect()

# Create the residual plot
plt.scatter(predicted_values, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')  # line at 0 to show residuals distribution
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Get actual values
'''
Actual vs. Predicted Scatter Plot: This code plots the actual values against the predicted values
to visualize how well the model's predictions match the true values. A perfect
fit would show the points along
the diagonal line (y = x).
'''
actual_values = predictions.select("label").rdd.map(lambda row: row['label']).collect()

# Create the scatter plot
plt.scatter(actual_values, predicted_values, color='green')
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.show()

'''
In the Actual vs. Predicted scatter plot, the points that fall on the line (y = x) indicate perfect
predictions, where the model's predicted values exactly match the actual values.

Here’s what it means for points on the line:

- The model's predictions are accurate for those data points, meaning the prediction error
(residual) is zero.
- These points demonstrate that the model has correctly captured the underlying pattern for
these specific instances.
- If a large number of points are close to or on the line, it suggests the model is performing well.

In contrast, points away from the line indicate prediction errors, where the model's predictions
differ from the actual values, showing where the model could improve.
'''


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # RMSE value from previous evaluation
#3. RMSE Bar Plot:
'''
A bar plot can work for visualizing a single evaluation metric like RMSE. It would display
the RMSE value as a purple bar, helping you understand how the model's error behaves. However,
it’s more useful for comparing multiple metrics or models. For just one RMSE value, it provides
a simple, clear visual representation.
'''

rmse = evaluator.evaluate(predictions)

# Plot RMSE as a bar chart
plt.bar(['RMSE'], [rmse], color='purple')
plt.ylabel('RMSE')
plt.title('RMSE for Model')
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    