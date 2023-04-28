import streamlit as st
import pandas as pd
import time
import ast
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="My App", page_icon=":books:", layout="wide")


def page1():
    
    st.title("Welcome to My Recommender System")
    st.subheader("In this recommender system you can get suggestions for Movies, Anime, Tv shows!")
    st.write("select recommendations page to see the results ")
    st.write("In the second page you can select which dataset you want to work with and also select a title. YOu can also select number of recommendations you want.")

    # Movies

    movie_images = ["Movies_end_of_year_2017.jpg"]
    st.image(movie_images, width=1000)

    # TV Shows

    tv_images = ["shows-to-watch-on-netflix-2020-circuit-breaker-quarantine-things-to-do-1280x705-1.jpg"]
    st.image(tv_images, width=1000)

    # Anime

    anime_images = ["anime-les-plus-connus-1024x576.png"]
    st.image(anime_images, width=1000)

def page2():

    # Load the input datasets
    d1 = pd.read_csv("movies_metadata.csv", low_memory=False)
    d2 = pd.read_csv("anime.csv", low_memory=False)
    d3 = pd.read_csv("TV Series.csv", low_memory=False)

    d1 = d1.drop_duplicates()
    d2 = d2.drop_duplicates()
    d3 = d3.drop_duplicates()

    # Convert the 'genre' column to a list of strings
    d1['genre'] = d1['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)])
    d1 = d1.drop(columns= 'genres')

    # Rename a column in d2
    d2 = d2.rename(columns={'Genres': 'genre'})
    d2 = d2.rename(columns={'Name': 'title'})

    # Rename a column in d3
    d3 = d3.rename(columns={'Genre': 'genre'})
    d3 = d3.rename(columns={'Series Title': 'title'})

    # Data preprocessing
    d1 = d1.drop(['adult','belongs_to_collection','homepage','id','imdb_id','original_language','original_title','overview','poster_path','production_companies','production_countries','spoken_languages','status','tagline','video'],axis=1)
    d1 = d1.dropna(subset=['genre'])  # Drop rows with missing genres
    d1 = d1[d1['vote_count'] > 0]  # Filter by vote count
    d1 = d1[d1['vote_average'] > 0]  # Filter by vote average
    d1['popularity'] = d1['popularity'].astype(float)  # Convert popularity to float
    d1[['vote_count', 'vote_average', 'popularity']] = StandardScaler().fit_transform(d1[['vote_count', 'vote_average', 'popularity']])  # Scale features

    d2 = d2.drop(['MAL_ID','English name','Japanese name','Episodes','Producers','Licensors','Studios','Duration','Rating','Score-10','Score-9','Score-8','Score-7','Score-6','Score-5','Score-4','Score-3','Score-2','Score-1'], axis=1)
    d2 = d2.dropna(subset=['genre'])

    d3 = d3.dropna(subset=['genre'])
    d3['Runtime'] = d3['Runtime'].astype(str).str.replace(' min', '')
    d3['Runtime'] = pd.to_numeric(d3['Runtime'], errors='coerce')

    # Compute the mean of the column
    mean_value = d3['Runtime'].mean()

    # Replace NaN values with the mean value
    d3['Runtime'].fillna(mean_value, inplace=True)

    # Feature engineering
    d1['popularity_score'] = d1['popularity'] * (d1['vote_count'] / (d1['vote_count'] + 1000))
    d2['popularity_score'] = d2['Popularity'] * (d2['Members'] / (d2['Members'].max() + 1000))

    d1_genre = d1['genre']
    d2_genre = d2['genre']
    d3_genre = d3['genre']

    unique_genres_d1 = d1_genre.explode().unique()
    df = pd.DataFrame(unique_genres_d1, columns=['genres'])

    # Drop any NaN values from the DataFrame
    df.dropna(inplace=True)


    def recommend(features,df,input_movie):
        df['features'] = df[features].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
        tfidf_matrix = tfidf.fit_transform(df['features'])
        cosine_sim = cosine_similarity(tfidf_matrix)

    # Get the index of the input movie
        idx = df[df['title'] == input_movie].index[0]
        
        if idx >= len(cosine_sim):
          return "Data not found"
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sim_scores
    # Compute the feature vectors for each movie in the dataset
    tfidf = TfidfVectorizer()

    # Define Streamlit app
    st.title('Content Recommender')

    # Dataset selection
    dataset = st.selectbox('Select a dataset', ['Movies', 'Anime', 'Tv Series'])

    if dataset == 'Movies':
        data = d1
        features = ['budget', 'popularity', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count', 'genre', 'popularity_score']
        input_movie = st.selectbox('Enter Movie Name',data['title'])
        N = st.sidebar.slider('Select Number of Recommendations', 0, 100, 10)
        if st.button('Get Recommendations'):
            recommendations = recommend(features,data,input_movie)
            top_movies = [data.iloc[recommendations[i][0]]['title'] for i in range(1,N+1) if len(recommendations) > i]
            recommended_movies_df = data[data['title'].isin(top_movies)][['title', 'budget', 'popularity', 'release_date', 'revenue', 'runtime', 'vote_average', 'vote_count', 'genre', 'popularity_score']]
            if recommendations:
                st.write(f"Top {N} movies similar to {input_movie}:")
                st.write(recommended_movies_df)
            else:
                st.write("Sorry, no recommendations found.")
                with st.spinner('Wait for it...'):
                    time.sleep(0)
                    st.snow()

    elif dataset == 'Anime':
        data = d2
        features = ['title', 'Score', 'genre', 'Type', 'Aired', 'Premiered', 'Source', 'Ranked', 'Popularity', 'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped', 'Plan to Watch', 'popularity_score']
        input_movie = st.selectbox('Enter Movie Name',data['title'])
        N = st.sidebar.slider('Select Number of Recommendations', 0, 100, 10)
        if st.button('Get Recommendations'):
            recommendations = recommend(features,data,input_movie)
            top_movies = [data.iloc[recommendations[i][0]]['title'] for i in range(1,N+1) if len(recommendations) > i]
            # Filter the DataFrame to include only the recommended movies
            recommended_movies_df = data[data['title'].isin(top_movies)][['title', 'Score', 'genre', 'Type', 'Aired', 'Premiered', 'Source', 'Ranked', 'Popularity', 'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped', 'Plan to Watch', 'popularity_score']]
            if recommendations:
                st.write(f"Top {N} Anime similar to {input_movie}:")
                st.write(recommended_movies_df)
            else:
                st.write("Sorry, no recommendations found.")
                with st.spinner('Wait for it...'):
                    time.sleep(0)
                    st.snow()

    elif dataset == 'Tv Series':
        data = d3
        features = ['title', 'Release Year', 'Runtime', 'genre', 'Rating', 'Cast', 'Synopsis']
        input_movie = st.selectbox('Enter Movie Name',data['title'])
        N = st.sidebar.slider('Select Number of Recommendations', 0, 100, 10)
        if st.button('Get Recommendations'):
            recommendations = recommend(features,data,input_movie)
            top_movies = [data.iloc[recommendations[i][0]]['title'] for i in range(1,N+1) if len(recommendations) > i]
            recommended_movies_df = data[data['title'].isin(top_movies)][['title', 'Release Year', 'Runtime', 'genre', 'Rating', 'Cast', 'Synopsis']]
            if recommendations:
                st.write(f"Top {N} Tv shows similar to {input_movie}:")
                st.write(recommended_movies_df)
            else:
                st.write("Sorry, no recommendations found.")
                with st.spinner('Wait for it...'):
                    time.sleep(0)
                    st.snow()
        
        


page_names_to_funcs = {
    "Introduction": page1,
    "recommendations": page2
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()