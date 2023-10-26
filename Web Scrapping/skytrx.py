import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer

base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
pages = 37
page_size = 100

reviews = []

for i in range(1, pages + 1):

    print(f"Scraping page {i}")

    # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    # Collect HTML data from this page
    response = requests.get(url)

    # Parse content
    content = response.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class": "text_content"}):
       reviews.append(para.get_text())
    
    print(f"   ---> {len(reviews)} total reviews")
    df = pd.DataFrame()
df["reviews"] = reviews

df.to_csv("BA_reviews.csv")
reviews = pd.read_csv("BA_reviews.csv")
reviews = reviews.pop('reviews')
reviews
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

reviews = reviews.str.replace('Trip Verified |', '')
reviews = reviews.str.replace('✅', '')
reviews = reviews.str.replace('|', '')
reviews = reviews.str.replace(r'\b(\w{1,3})\b', '')
reviews = reviews.apply(remove_punctuations)
reviews
freq_words = pd.Series(' '.join(reviews).lower().split()).value_counts()[:50]
freq_words
categories = ['negative', 'positive']
num_cat = len(categories)
num_cat
# TF-IDF Feature Generation
# # Initialize regex tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# # Vectorize document using TF-IDF
tf_idf_vect = TfidfVectorizer(lowercase=True,
                        stop_words='english',
                        ngram_range = (1,1),
                        tokenizer = tokenizer.tokenize)

# Fit and Transfrom Text Data
reviews_counts = tf_idf_vect.fit_transform(reviews)

# Check Shape of Count Vector
reviews_counts.shape

# Import KMeans Model
from sklearn.cluster import KMeans

# Create Kmeans object and fit it to the training data 
kmeans = KMeans(n_clusters=num_cat).fit(reviews_counts)

# Get the labels using KMeans
pred_labels = kmeans.labels_
pred_labels
unique, counts = np.unique(pred_labels, return_counts=True)
dict(zip(unique, counts))
df_reviews = pd.DataFrame({'review': reviews, 'label': pred_labels})
df_reviews