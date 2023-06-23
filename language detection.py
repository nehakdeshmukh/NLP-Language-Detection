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


data = pd.read_csv(r"C:\Neha\kaggle Projects\Language detection\dataset.csv")

