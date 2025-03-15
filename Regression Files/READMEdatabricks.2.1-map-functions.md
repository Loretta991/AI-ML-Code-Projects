
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #bigdata_homework2.  RDD 2.1-map-functions - Loretta Gray
#Create a Spark Core RDD from Different Ways:
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # create entry points to spark
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
            '''
Map functions

These functions are probably the most commonly used functions when dealing with an RDD object.

map()
mapValues()
flatMap()
flatMapValues()
map¶

The map() method applies a function to each elements of the RDD. Each element has to be a valid input to the function. The returned RDD has the function outputs as its new elements.

Elements in the RDD object map_exp_rdd below are rows of the mtcars in string format. We are going to apply the map() function multiple times to convert each string elements as a list elements. Each list element has two values: the first value will be the auto model in string format; the second value will be a list of numeric values.


'''

# create an example RDD
map_exp_rdd = sc.textFile('/content/sample_data/mtcars.csv')
map_exp_rdd.take(4)

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # split auto model from other feature values
map_exp_rdd_1 = map_exp_rdd.map(lambda x: x.split(',')).map(lambda x: (x[0], x[1:]))
map_exp_rdd_1.take(4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # remove the header row
header = map_exp_rdd_1.first()
# the filter method apply a function to each elemnts. The function output is a boolean value (TRUE or FALSE)
# elements that have output TRUE will be kept.
map_exp_rdd_2 = map_exp_rdd_1.filter(lambda x: x != header)
map_exp_rdd_2.take(4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # convert string values to numeric values
map_exp_rdd_3 = map_exp_rdd_2.map(lambda x: (x[0], list(map(float, x[1]))))
map_exp_rdd_3.take(4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
mapValues
The mapValues function requires that each element in the RDD has a key/value pair structure, for example, a tuple of 2 items, or a list of 2 items. The mapValues function applies a function to each of the element values. The element key will remain unchanged.
We can apply the mapValues function to the RDD object mapValues_exp_rdd below.


'''

mapValues_exp_rdd = map_exp_rdd_3
mapValues_exp_rdd.take(4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
mapValues_exp_rdd_1 = mapValues_exp_rdd.mapValues(lambda x: np.mean(x))
mapValues_exp_rdd_1.take(4)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
When using mapValues(), the x in the above lambda function refers to the element value, not including the element key.

flatMap
This function first applies a function to each elements of an RDD and then flatten the results. We can simply use this function to flatten elements of an RDD without extra operation on each elements.

'''

x = [('a', 'b', 'c'), ('a', 'a'), ('c', 'c', 'c', 'd')]
flatMap_exp_rdd = sc.parallelize(x)
flatMap_exp_rdd.collect()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            '''
flatMapValues
The flatMapValues function requires that each element in the RDD has a key/value pair structure. It applies a function to each element value of the RDD object and then flatten the results.
For example, my raw data looks like below. But I would like to transform the data so that it has three columns: the first column is the sample id; the second the column is the three types (A,B or C); the third column is the values.

sample id    A    B    C
1    23    18    32
2    18    29    31
3    34    21    18



'''


# example data
my_data = [
    [1, (23, 28, 32)],
    [2, (18, 29, 31)],
    [3, (34, 21, 18)]
]
flatMapValues_exp_rdd = sc.parallelize(my_data)
flatMapValues_exp_rdd.collect()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # merge A,B,and C columns into on column and add the type column
flatMapValues_exp_rdd_1 = flatMapValues_exp_rdd.flatMapValues(lambda x: list(zip(list('ABC'), x)))
flatMapValues_exp_rdd_1.collect()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # unpack the element values
flatMapValues_exp_rdd_2 = flatMapValues_exp_rdd_1.map(lambda x: [x[0]] + list(x[1]) )
flatMapValues_exp_rdd_2.collect()
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    