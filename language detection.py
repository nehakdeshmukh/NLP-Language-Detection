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