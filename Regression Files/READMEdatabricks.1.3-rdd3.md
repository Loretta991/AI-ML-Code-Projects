
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #bigdata_homework2.  RDD 1.3-conversion-between-rdd-and-dataframe - Loretta Gray
#Create a Spark Core RDD from Different Ways:
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
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mtcars.rdd.take(2)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mtcars_map = mtcars.rdd.map(lambda x: (x['_c0'], x['mpg']))
mtcars_map.take(5)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            mtcars_mapvalues = mtcars_map.mapValues(lambda x: [x, x * 10])
mtcars_mapvalues.take(5)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            rdd_raw = sc.textFile('/content/sample_data/mtcars.csv')
rdd_raw.take(5)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            header = rdd_raw.map(lambda x: x.split(',')).filter(lambda x: x[1] == 'mpg').collect()[0]
header[0] = 'model'
header
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            rdd = rdd_raw.map(lambda x: x.split(',')).filter(lambda x: x[1] != 'mpg')
rdd.take(2)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Convert RDD elements to RDD Row objects

'''
First we define a function which takes a list of column names and a list of values and create a Row of key-value pairs. Since keys in an Row object are variable names, we can’t simply pass a dictionary to the Row() function. We can think of a dictionary as an argument list and use the ** to unpack the argument list.
See an example.
'''

from pyspark.sql import Row
my_dict = dict(zip(['a', 'b', 'c'], range(1, 4)))
Row(**my_dict)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Let's define a funciton
def list_to_row(keys, values):
    row_dict = dict(zip(keys, values))
    return Row(**row_dict)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            rdd_rows = rdd.map(lambda x: list_to_row(header, x))
rdd_rows.take(3)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #Now we can convert the RDD to a DataFrame.

df = spark.createDataFrame(rdd_rows)
df.show(5)
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    