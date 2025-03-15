
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#The Covid 19 Dataset

#The COVID19 Dataset is a derived from information collected by the U.S.:

#STATE- total covid cases by state
#TOT_CASES - total number of covid cases
#CONF_CASES - total number of confirmed covid cases
#PROB_CASES - probable covid cases
#NEW_CASES- new covid cases
#PNEW_CASES - possible new covid cases
#TOT_DEATH- proportion of owner-occupied units built prior to 1940
#CONF-death- weighted distances to five Boston employment centres
#PROB_DEATH- index of accessibility to radial highways
#NEW_DEATH- new covid deaths
#PNEW_DEATH - possible new covid deaths
#CREATED_AT- datetime covid record recorded
#CONSENT_CASES - consent to use covid cases data case 'agree' or 'not agree
#CONSENT_DEATH- consent to use covid death data case 'agree' or 'not agree

#This analysis will focus on the TOT_CASES and TOT_DEATH, By STATE


#Let's see what in the raw data looks like first, and set up a fuction
#to read_csv file of 600,061 reported Covid 19 cases, afterwards this
#project wilthen focus on data linear regression subplots


from numpy.core.fromnumeric import reshape
from pandas.core.internals.construction import dataclasses_to_dicts
from numpy.ma.core import frombuffer
from pandas.core.groupby.groupby import DataError


#COVID-19 Data

# Import pandas
import pandas as pd
import numpy as np
import datetime as dt


# reading csv file
data= pd.read_csv('Covid19Proj.csv')
# dropping the rows having NaN values
   
# Initialising numpy array
data=data.fillna(data.mean())

print(data)

#Dimension of the dataset
print(np.shape(60061.14))
# Let's summarize the data to see the distribution of data
#print(data.describe()) 
# dataframe.shape

# Python code to demonstrate
# to replace nan values
# with average of columns
  
import numpy as np
  
# Initialising numpy array
data=data.fillna(data.mean())


#Logistics from Scratch Problem

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(10)
num_observations = 60061

data.isnull().sum()

# Visualizing the differences between actual prices and predicted values
plt.scatter(data.tot_cases, data.tot_death)
plt.xlabel("Total Cases")
plt.ylabel("Total Deaths")
plt.title("Cases vs Deaths")
plt.show()


# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Create a Random Forest Regressor
reg = RandomForestRegressor()
# Initialising numpy array
data=data.fillna(data.mean())
# Train the model using the training sets 

 #reg.fit(data.tot_cases, data.tot_death)


data=data.fillna(data.mean())



from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

N = 1000
X0 = np.random.normal(np.repeat(np.random.uniform(0, 20, 4), N), 1)
X1 = np.random.normal(np.repeat(np.random.uniform(0, 10, 4), N), 1)
X = np.vstack([X0, X1]).T
y = np.repeat(range(4), N)
colors = ['red', 'blue', 'purple', 'green']
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=y, palette=colors, alpha=0.5, s=7)

means = np.vstack([X[y == i].mean(axis=0) for i in range(4)])
ax = sns.scatterplot(means[:, 0], means[:, 1], hue=range(4), palette=colors, s=20, ec='black', legend=False, ax=ax)
plt.show()


#Multivariate logistic regression is like simple logistic regression but with multiple predictors.\
#Logistic regression is similar to linear regression but you can use it when your response variable is binary. \
#This is common in medical research because with multiple logistic regression you can adjust for confounders.\
#For example you may be interested in predicting whether or not someone may develop a disease based on the\
#exposure to some substance. You can also use multiple logistic regression to increase \
#your prediction power by adding more predictors instead of just using one.You could calculate the mean 
#of each group, and draw a scatter dot at that position.

#Alternatively, Scikit Learns's KMeans could be used to calculate both the KMeans labels and the means:


# to ignore the warnings
#--- LOGOSTOC REGRESSION SCATTER PLOT
from warnings import filterwarnings
N= 60061
x1 = np.random.multivariate_normal([61, 378], [[60061, 10.75],[10.75, 60061]], N)
x2 = np.random.multivariate_normal([395, 115], [[60061, 10.75],[10.75, 60061]], N)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
#Let’s see how it looks.

plt.figure(figsize=(20,20))
plt.scatter(simulated_separableish_features[:, 0], 
            simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)

#Picking the link function
def sigmoid(tot_cases):
    return 1 / (1 + np.exp(-tot_cases))

#Run the link function to take care of the mixed algebra
def log_likelihood(features, target, weights):
    tot_cases = np.dot(features, weights)
    ll = np.sum( target*tot_cases - np.log(1 + np.exp(tot_cases)) )
    return ll

#time to run
    weights = logistic_regression(simulated_separableish_features, simulated_labels,
    num_steps = 60061, learning_rate = 5e-5, add_intercept=True)
    #print(weights)
plt.show ()


#---Another method
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

#---Kmeans CLUSTER
N = 60061
X0 = np.random.normal(np.repeat(np.random.uniform(0, 20, 20), N), 10)
X1 = np.random.normal(np.repeat(np.random.uniform(0, 10, 20), N), 11)
X = np.vstack([X0, X1]).T
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters).fit(X)

colors = ['orange', 'blue', 'purple', 'grey']
ax = sns.scatterplot(X[:, 0], X[:, 1], hue=kmeans.labels_, palette=colors, alpha=0.5, s=7)
ax = sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                     hue=range(num_clusters), palette=colors, s=60, ec='black', legend=False, ax=ax)
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import array 
import pandas as pd
# to ignore the warnings
from warnings import filterwarnings
# Create a set of colors
colors = ['#EED6AF', '#B7C3F3', '#DD7596', '#8EB897', '#C67171']
# slice properties

# Wedge properties
slices= { 'linewidth' : 1, 'edgecolor' : "black" }
slices = [19, 40, 25, 30, 15]

# Creating explode data
explode = (0.1, 0.0, 0.2, 0.3, 0.0, 0.0)
labels = ['tot_death','tot_cases','conf_cases', 'new_case','state']


fig = plt.figure()
#ax = plt.subplots(figsize =(20,20))
ax = fig.add_axes([1,1,1,1])
# Equal aspect ratio ensures the pie chart is circular.

# Creating autocpt arguments
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

# Equal aspect ratio ensures that pie is drawn as a circle  
ax.axis('equal')
ax.set_title('Covid 19 Statistics', fontsize=16)
plt.pie(slices, colors=colors, labels=labels, startangle=90,\
        shadow=True, autopct ='%1.2f%%', wedgeprops={'linewidth': 5.0, 'edgecolor': 'white'})
plt.legend(loc = 'upper right', title = 'cases')

plt.show()  


import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib.pyplot as plt
import scipy 

# to ignore the warnings
from warnings import filterwarnings    
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(50,20))
sns.lineplot(data=data, x="state", y="tot_cases")
data = data[~(data['tot_death'] >= 50.0)]
print(np.shape(data))

#Plots All Working Above, now generate the statistical data below
"""
perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
print("Column %s outliers = %.2f%%" % (k, perc))

data = data[~(data['tot_death'] >= 50.0)]
print(np.shape(data))

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(7,8))
index = 0
axs = axs.flatten()
for k,v in data.items():
    sns.scatterplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
"""

plt.figure(figsize=(15, 10))
sns.heatmap(data.corr().abs(),  annot=True)


# Let's scale the columns before plotting them against tot_cases
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
columns_sels= ['tot_death','tot_cases','conf_cases', 'prob_cases', 'new_case']

x = data.loc[:,columns_sels]
y = data['tot_cases']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=columns_sels)

fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20,5))
                                                   
index = 0
axs = axs.flatten()
for i, k in enumerate(columns_sels):
    sns.scatterplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Initialising numpy array
data=data.fillna(data.mean())

# Importing libraries
import seaborn as sns
import matplotlib.pyplot as plt
 
# Setting the data
x = ["tot_cases", "tot_death"]
y = [60061, 22033]
 
# setting the dimensions of the plot
#fig, ax = plt.subplots(figsize=(15, 5))
fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(5,5))
# drawing the plot
sns.boxplot(x = y)
plt.show()

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    