import numpy as np
import pandas as pd
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords', 'production_companies', 'cast', 'crew']]

movies.dropna(inplace=True)

def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['production_companies'] = movies['production_companies'].apply(convert)

def convertcast(obj):
    l = []
    c = 0
    for i in ast.literal_eval(obj):
        c += 1
        l.append(i['name'])
        if c == 3:
            break
    return l

movies['cast'] = movies['cast'].apply(convertcast)

def fetchdirector(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies['crew'] = movies['crew'].apply(fetchdirector)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['production_companies'] = movies['production_companies'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['production_companies'] + movies['cast'] + movies['crew']

new_df = movies[['id', 'title', 'tags']]
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Using .loc[] to avoid SettingWithCopyWarning
new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)

def generate_similarity():
    pickle.dump(similarity, open('similarity.pkl', 'wb'))