
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #bigdata_homework2.  RDD 1.2 - Loretta Gray
#Create a Spark Core RDD from Different Ways:
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # create entry points to spark
!pip install pyspark
try:
    sc.stop()
except:
    pass
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
sc=SparkContext()
spark = SparkSession(sparkContext=sc)
#The spark context has stopped and the driver is restarting. Your notebook will be automatically reattached.
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mtcars = spark.read.csv(path='/content/sample_data/mtcars.csv',
                        sep=',',
                        encoding='UTF-8',
                        comment=None,
                        header=True,
                        inferSchema=True)
mtcars.show(n=5, truncate=False)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Create DataFrame from rdd with createDataFrame function
from pyspark.sql import Row
rdd = sc.parallelize([
    Row(x=[1,2,3], y=['a','b','c']),
    Row(x=[4,5,6], y=['e','f','g'])
])
rdd.collect()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df = spark.createDataFrame(rdd)
df.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #from pandas dataframe
import pandas as pd
pdf = pd.DataFrame({
    'x': [[1,2,3], [4,5,6]],
    'y': [['a','b','c'], ['e','f','g']]
})
pdf
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Create spark dataframe from pandas dataframe
df = spark.createDataFrame(pdf)
df.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df.dtypes
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            my_list = [['a', 1], ['b', 2]]
df = spark.createDataFrame(my_list, ['my_column1', 'my_column2'])
df.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df.dtypes
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            my_list = [(['a', 1], ['b', 2])]
df = spark.createDataFrame(my_list, ['x', 'y'])
df.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df.select("x").show()
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    