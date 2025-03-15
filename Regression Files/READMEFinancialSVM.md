
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # import packages
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC, SVC  # Linear Support Vector Classification


RANDOM_STATE = 1234
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # read the data
payments = pd.read_csv("payment_data.csv")
payments = payments.set_index("id")

customers = pd.read_csv("customer_data.csv")
customers = customers.set_index("id")

# merge two frames into one
customer_data = customers.join(payments)
customer_data
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # get the label distribution
(customer_data["label"]).value_counts()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # get the count of missing values for each column
customer_data.isnull().sum(axis=0)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # check if there is any customer with too many missing values
print("max missing = {}\n min missing = {}\n mean missing = {}".format(
        customer_data.isnull().sum(axis=1).max(),
        customer_data.isnull().sum(axis=1).min(),
        customer_data.isnull().sum(axis=1).mean()))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # make the prod_code to categorical variable
for prod_code in customer_data["prod_code"].unique():
    customer_data["prod_code_{}".format(prod_code)] = customer_data["prod_code"] == prod_code
# Category features are fea1, fea3, fea5, fea6, fea7, fea9
for feature_id in [1, 3, 5, 6, 7, 9]:
    for value in customer_data["fea_{}".format(feature_id)].unique():
        customer_data["feature_{}_{}".format(feature_id, value)] = customer_data["fea_{}".format(feature_id)] == value
customer_data
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # remove useless columns
# prod_limit: too many missing values
# dates: not expressive
customer_data = customer_data.drop(["prod_limit", "report_date", "update_date", "prod_code", "fea_1", "fea_3", "fea_5", "fea_6", "fea_7", "fea_9"], axis=1)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # convert to np.array for training and fill in the missing value
data = customer_data.to_numpy(na_value=np.nan).astype(float)
# fill in missing vales with mean value
imputer = SimpleImputer(verbose=1)
data = imputer.fit_transform(data)
data
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # check if there is missing value
(data == np.nan).any()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # scale the data within the range of [0, 1]
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # # balance the dataset
# mask = data[:, 0] == 1
# risk_customers = data[mask]
# print("number of risk customers = {}".format(risk_customers.shape[0]))
#
# safe_customers = data[~mask]
# indices = np.random.choice(safe_customers.shape[0], risk_customers.shape[0]*2, replace=False)
# safe_customers = safe_customers[indices]
# print("number of safe customers = {}".format(safe_customers.shape[0]))
#
# data = np.concatenate([risk_customers, safe_customers], axis=0)
# print("Shape of selected data = ", data.shape)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # split the data into X, y
X = data[:, 1:]
y = data[:, 0]
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            linear_model = LinearSVC(random_state=RANDOM_STATE)
cross_val_score(linear_model, X, y, cv=3, n_jobs=-1, scoring="accuracy")
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # get the prediction during cross validation
pred_tags = cross_val_predict(linear_model, X, y, cv=3, n_jobs=-1, method="predict")
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # get confusion matrix
confusion_matrix(y, pred_tags)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # polynomial kernel
poly_svc = SVC(kernel="poly", random_state=RANDOM_STATE)
pred_tags = cross_val_predict(poly_svc, X, y, cv=3, n_jobs=-1, method="predict")
confusion_matrix(y, pred_tags)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Gaussian RBF kernel
gaussian_svc = SVC(kernel="rbf", random_state=RANDOM_STATE)
pred_tags = cross_val_predict(gaussian_svc, X, y, cv=3, n_jobs=-1, method="predict")
confusion_matrix(y, pred_tags)
            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    