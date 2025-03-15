
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### "Amazon-Alexa" text classification using "Bag of Words and "TF-IDF" ( w/ spacy and sklearn pipelines )</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # install the following if they are not installed yet !

#!pip install spacy
#!python -m spacy download en_core_web_sm
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### import the required libraries</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Load the input data ( "amazon alexa reviews data")</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Loading TSV file
df_amazon = pd.read_csv ("amazon_alexa.tsv", sep="\t")
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Top 5 records
df_amazon.head()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Data Exploration</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.columns.tolist()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # shape of dataframe
df_amazon.shape
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.feedback.unique()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['feedback'].value_counts()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.rating.unique()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['rating'].value_counts()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # View data information
df_amazon.info()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Checking the missing values</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.isnull().sum()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Basic summary statistics</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.describe()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['length'] = df_amazon['verified_reviews'].apply(len)
df_amazon.shape
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Visual exploration of data</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df_amazon['rating'].value_counts().plot.bar(color = 'red',figsize = (11, 7))
plt.title('Visualizing the ratings dist.')
plt.xlabel('ratings')
plt.ylabel('count')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['rating'].value_counts()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            labels = '5', '4', '3', '2', '1'
sizes = [2286, 455, 161, 152, 96]
colors = ['cyan', 'magenta', 'pink', 'yellow', 'red']
explode = [0.001, 0.001, 0.001, 0.001, 0.001]
plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True)
plt.title(' pie chart representing ratings occuposition')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['variation'].value_counts().plot.bar(color = 'green', figsize = (11, 7))
plt.title('Visualizing the variations dist.')
plt.xlabel('variations')
plt.ylabel('count')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['feedback'].value_counts().plot.bar(color = 'orange', figsize = (7, 9))
plt.title('Visualizing the feedbacks dist.')
plt.xlabel('feedbacks')
plt.ylabel('count')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon['length'].value_counts().plot.hist(color = 'red', figsize = (14, 5), bins = 50)
plt.title('Visualizing the length dist.')
plt.xlabel('lengths')
plt.ylabel('count')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon[df_amazon['length'] == 50]['verified_reviews'].iloc[0]
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.date.describe()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.groupby('variation').mean()[['rating']].plot.bar(color = 'green', figsize=(14, 6))
plt.title("Variation wise Mean Ratings")
plt.xlabel('variatiions')
plt.ylabel('ratings')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.groupby('feedback').mean()[['rating']].plot.bar(color = 'crimson', figsize=(5, 4))
plt.title("feedback wise Mean Ratings")
plt.xlabel('feedbacks')
plt.ylabel('ratings')
plt.show()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(df_amazon.verified_reviews)
sum_words = words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'yellow')
plt.title("Most Frequently Occuring Words - Top 20")
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            df_amazon.feedback.value_counts()
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            import string
import spacy

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            print(stop_words)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Creating our tokenizer function
def spacy_tokenizer(sentence):
    
    """This function will accepts a sentence as input and processes the sentence into tokens, performing lemmatization, 
    lowercasing, removing stop words and punctuations."""
    
    # Creating our token object, which is used to create documents with linguistic annotations.
    #mytokens = parser(sentence)
    mytokens = nlp(sentence)
    #print(mytokens)
    
    # Lemmatizing each token and converting each token into lowercase
    # Note that spaCy uses '-PRON-' as lemma for all personal pronouns lkike me, I etc
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Data Cleaning - Custom transformer class
#### 1. We will create custom transformer to clean the tokenized data
#### 2. We will create a custom predictors class which inherits the TransformerMixin class. This class overrides the transform, fit and get_parrams methods. We’ll also create a clean_text() function that removes spaces and converts text into lowercase.
#### 3. In object-oriented programming languages, a mixin (or mix-in) is a class that contains methods for use by other classes without having to be the parent class of those other classes.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from sklearn.base import TransformerMixin

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # It checks the parameters in the dataset using X_train and y_train. It then converts them into a 
        #         structure that the model can understand
        # Cleaning Text - Override the transform method to clean text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        # it fits the model into the dataset. This enables the model to learn by understanding patterns in the 
        #.   dataset
        return self

    def get_params(self, deep=True):
        # This method retrieves all the converted and optimized parameters to produce an optimized model
        return {}

# Basic function to clean the text
def clean_text(text):
    # This function cleans our dataset and converts all the texts into lower case
    # Removing spaces and converting text into lowercase
    return text.strip().lower()
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Feature Engineering
#### Remember that, the objective of this exercise is to predict whether an Amazon Alexa product review is positive or negative based on review text. And as of now we are manager to clean our text but in order our model to understand it we must convert it into numeric format.</div>

<div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Vectorization

#### 1. Note that labels (feedback 0 and 1 in this case) are already in numeric format
#### 2. We are going to use Bag of Words(BoW) to convert text into numeric format.
#### 3. BoW converts text into the matrix of occurrence of words within a given document. It focuses on whether given word occurred or not in given document and generate the matrix called as BoW matrix/Document Term Matrix
#### 4. We are going to use sklearn's CountVectorizer to generate BoW matrix.
#### 5. In CountVectorizer we will use custom tokenizer 'spacy_tokenizer' and ngram range to define the combination of adjacent words. So unigram means sequence of single word and bigrams means sequence of 2 continuous words like wise n means sequence of n continuous words.
#### 6. In this example we are going to use unigram, so our lower and upper bound of ngram range will be (1,1)</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### TF-IDF

#### 1. We are also going to look at TF-IDF(Term Frequency-Inverse Document Frequency). This is helpful to normalize our BoW by looking at each word's frequency in comparison to document frequency.
#### 2. In other words, it's a way of representing how important a particular term is, in the context of given document based on how many times that term appears and in how many documents it appears.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Create Train and Test Datasets
#### 1. We are going to create train and test datasets
#### 2. Train dataset will be used to train the model and test dataset to test the model performance
#### 3. For more detail about splitting the dataset, please refer Train Test Split</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from sklearn.model_selection import train_test_split

X = df_amazon['verified_reviews'] # the features we want to analyze
ylabels = df_amazon['feedback'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)
print(f'X_train dimension: {X_train.shape}')
print(f'y_train dimension: {y_train.shape}')
print(f'X_test dimension: {X_test.shape}')
print(f'y_train dimension: {y_test.shape}')
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Creating a Pipeline and Generating the Model
#### 1. We are going to use LogisticRegression classifier for review classification. For more details about logistic regression please refer Logistic Regression From Scratch With Python, Binary Logistic Regression Using Sklearn, Multiclass Logistic Regression Using Sklearn
#### 2. Our pipeline contains three components a cleaner, a vectorizer and a classifier
##### Cleaner: Cleaner uses our 'predictors' class object to clean and preprocess the text.
##### Vectorizer: Vectorizer uses 'CountVectorizer' object to create the bag of words matrix for our text.
##### Classifier: It just performs the logistic regression to classify the sentiments.
#### 3. Once our pipeline is built, we will fit the pipeline components using fit()</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">### Model Score
#### 1. Since our model is trained using training data, we can now use test data for model evaluation
#### 2. Below are the few metrics that we will use to evaluate our model
##### Accuracy refers to the percentage of the total predictions our model makes that are completely correct.
##### Precision describes the ratio of true positives to true positives plus false positives in our predictions.
##### Recall describes the ratio of true positives to true positives plus false negatives in our predictions.
#### 3. All three metrics are measured from 0 to 1, where 1 is predicting everything completely correctly.</div>


            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)
print(pipe.score(X_test, y_test))

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            print(len(X_test), len(predicted))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            print(predicted)
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            # Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe2 = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', classifier)])

# model generation
pipe2.fit(X_train,y_train)
print(pipe2.score(X_test, y_test))
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            pred = pipe.predict(['Love this just wish it had a built in hub','I will be returning this piece of garbage','item returned for repair ,receivded item back'])

pred2 = pipe2.predict(['Love this just wish it had a built in hub','I will be returning this piece of garbage','item returned for repair ,receivded item back'])
print(pred)
print(pred2)
            </pre>


            <div style="background-color:#f4f8fb; padding:10px; border-radius:8px; margin:10px 0;">#### So overall, our model correctly identified a comment’s sentiment 91.95% of the time. When it predicted a review was positive, that review was actually positive 91.94% of the time. When handed a positive review, our model identified it as positive 100.0% of the time</div>


    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    