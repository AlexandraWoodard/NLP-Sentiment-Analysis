# Task 21 Capstone Project - NLP Applications
# Alexandra Woodard - March 2024

# PROJECT OVERVIEW
# 1. Implement a sentiment analysis model using spaCy
# 2. Preprocess the text data
# 3. Create a function for sentiment analysis
# 4. Test your model on sample product reviews
# 5. Write a brief summary of the project in a separate PDF file.

# Import spaCy and load the small English language model

import spacy

nlp = spacy.load('en_core_web_sm')

# Import pandas to make a dataframe from the CSV file

import pandas as pd

# Load the datafile with Amazon reviews

amz_data = pd.read_csv("1429_1.csv", low_memory=False)

# Preprocessing

# Create a dataframe with only review data
amz_review_data = amz_data['reviews.text']

# Exclude rows with missing values for reviews
amz_review_data_clean = amz_review_data.dropna() 

# Check out the basic information about the clean dataframe using head and describe

print(amz_review_data_clean.head())
print(" ")
print(amz_review_data_clean.describe())

# Create a variable to pick a review number at random in the file and then pull in the content of that review

import random 
random_review_number = int(random.randint(1, 34659))
random_review = str((amz_review_data_clean[random_review_number])) 

# Print the text of the randomly chosen review

print("Review text for Sentiment Analysis:", end=' ')
print(random_review)

# Create a function to perform sentiment analysis on a random review

from textblob import TextBlob

def amz_review_sentiment_analysis(random_review):
    # Make the review lower case and remove whitespace and characters at the beginning and end of the text.
    random_review = random_review.lower().strip() 
    # Tokenize the review
    doc = nlp(random_review)
    # Remove stop words from the review text
    filtered_tokens = [token.text for token in doc if not token.is_stop]
    filtered_review = " ".join(filtered_tokens)
    # Perform sentiment analysis and print the results 
    blob = TextBlob(filtered_review)
    polarity = blob.sentiment.polarity
    print("Sentiment Analysis : the polarity of the review (-1 for very negative, +1 for very positive) is", polarity)
    return polarity

# Call the function
    
amz_review_sentiment_analysis(random_review)

# Compare the similarity of two reviews

# Generate two random numbers in the range of the data in the file. 

random_number_one = int(random.randint(1, 34659))
random_number_two = int(random.randint(1, 34659))

# Pull the reviews into two new variables and print them. 

review_1 = str((amz_review_data_clean[random_number_one]))
print("Review 1 for Similarity Analysis:", end=' ')
print(review_1)

review_2 = str((amz_review_data_clean[random_number_two]))
print("Review 2 for Similarity Analysis:", end=' ')
print(review_2)

# Make text all text lowercase and remove spaces and characters at the beginning and end of the string (e.g. punctuation)

review_1 = review_1.lower().strip() 
review_2 = review_2.lower().strip() 

# Tokenize the text

review_1_nlp = nlp(review_1)
review_2_nlp = nlp(review_2)

# Create a function to compare the two random reviews

def review_comparison(review_1_nlp, review_2_nlp):
    print("The similarity of Review 1 to Review 2 text: ", end=' ')
    print(review_1_nlp.similarity(review_2_nlp))
    print("Similarity scores range from 0 (dissimilar) to 1 (similar)")

# Call the function     

review_comparison(review_1_nlp, review_2_nlp)

