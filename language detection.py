# basic libraries
import os
import re
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# preprocessing tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# model building tools
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score,recall_score,accuracy_score,classification_report,confusion_matrix #metrics
from sklearn import tree

data = pd.read_csv(r"C:\Neha\kaggle Projects\Git hub\NLP-Language-Detection\dataset.csv")

data.columns = ('text','language')
data


# train test split:
x_train, x_test, y_train, y_test = train_test_split(data.text.values, data.language.values,
                                                    test_size=0.1, random_state=42)

x_train.shape, y_train.shape, x_test.shape, y_test.shape


# Unique Language

print("No of Unique Languages : ",len(np.unique(data['language'])))

print("examples in each language : ",data.language.value_counts())


# function to clean text
def clean_txt(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]',' ',text)
    text=re.sub(r'[_0-9]',' ',text)
    text=re.sub(r'\s\s+',' ',text)
    return text

# example
txt = 'I am  (&*(()))finding %$#a Job '
print(clean_txt(txt))

x_train = [clean_txt(text) for text in tqdm(x_train)]
x_test = [clean_txt(text) for text in tqdm(x_test)]

# using Tfidf Vectorizer:
tfidf = TfidfVectorizer()
tfidf.fit(x_train)
x_train_ready = tfidf.transform(x_train)
x_test_ready = tfidf.transform(x_test)

x_train_ready,x_test_ready


enc = LabelEncoder()
enc.fit(y_train)
y_train_ready = enc.transform(y_train)
y_test_ready = enc.transform(y_test)

# storing encoded label hast list as 'labels'
labels = enc.classes_

# display first 5 labels:
print(labels[:5])


preds = enc.inverse_transform([0,3,5])
print(preds)


#Naive Classifier 0.941
em_model = MultinomialNB().fit(x_train_ready, y_train_ready)
pred_test_MNB = em_model.predict(x_test_ready)
precision = precision_score(y_test_ready, pred_test_MNB,average='weighted',zero_division=1)
recall = recall_score(y_test_ready, pred_test_MNB,average='weighted')
accuracy = accuracy_score(y_test_ready, pred_test_MNB)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))

#Tree 0.9
em_model = tree.DecisionTreeClassifier().fit(x_train_ready, y_train_ready)
pred_test_MNB = em_model.predict (x_test_ready)
precision = precision_score(y_test_ready, pred_test_MNB,average='weighted')
recall = recall_score(y_test_ready, pred_test_MNB,average='weighted')
accuracy = accuracy_score(y_test_ready, pred_test_MNB)#806
print('Precision: {} / Recall: {} / Accuracy: {}'.format(np.round(precision, 3), np.round(recall, 3), np.round(accuracy, 3)))
