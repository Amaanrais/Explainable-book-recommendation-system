1# Imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pathlib
import textwrap
import google.generativeai as genai
from recos_model import clean_data, build_model, recommend, generate_explanation

df = clean_data()
edf = build_model(df)

input_book = input("Enter a book:")

recommendations = recommend(input_book, edf)

explanations = generate_explanation(input_book, df, recommendations)

explanation_index = recommendations[0].index
print(explanation_index)

print(explanations)
