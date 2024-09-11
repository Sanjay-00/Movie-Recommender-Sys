# 🎬 Movie Recommender System

This is a simple, content-based movie recommender system built using Python and Streamlit. It recommends movies based on the similarity between them using a pre-trained model and a similarity matrix.

## 📌 Features

- Recommends 5 similar movies based on a selected movie.
- Fetches movie posters using The Movie Database (TMDb) API.
- Clean and simple user interface built with **Streamlit**.
- Fully interactive app with a dropdown selection for choosing movies.

## 🚀 Demo

Check out the live demo of the app here: [Movie Recommender System]([https://your-app-url.streamlit.app](https://movie-recommender-sys-j5otxdqzv3buk7zdwfqvjz.streamlit.app/)) (replace with your deployed app link).

## 🛠️ Tech Stack

- **Python**
- **Streamlit**: For building the web app.
- **Pandas**: For handling movie data.
- **Requests**: To interact with the TMDb API.
- **TMDb API**: To fetch movie posters.
- **Pickle**: For loading pre-trained similarity matrices.

## ⚙️ How It Works

1. The app loads a pre-trained movie similarity matrix (using cosine similarity).
2. A user selects a movie from the dropdown list.
3. The app fetches the top 5 movies that are most similar to the selected one.
4. The TMDb API is used to fetch and display the posters for the recommended movies.

