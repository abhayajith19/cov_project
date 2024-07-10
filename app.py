import streamlit as st
import pandas as pd
import joblib
import json

@st.cache_resource
def load_model_1():
    return joblib.load('voting_regressor.pkl')

@st.cache_resource
def load_model_2():
    return joblib.load('gradient_boosting_regressor.pkl')

@st.cache_resource
def load_language_encoder():
    return joblib.load('label_encoder_languages.pkl')

@st.cache_resource
def load_genre_encoder():
    return joblib.load('genre_onehot_encoder.pkl')

@st.cache_resource
def load_languages():
    with open('language_mapping.json', 'r') as f:
        languages_dict = json.load(f)
    return list(languages_dict.values())

@st.cache_resource
def load_directors():
    df = pd.read_csv('directors_encoded_name.csv')
    return df['primaryName'].to_list()

@st.cache_resource
def load_actors():
    df = pd.read_csv('actors_encoded_name.csv')
    return df['primaryName'].to_list()

@st.cache_resource
def load_scaler():
    return joblib.load('standard_scaler.pkl')

@st.cache_resource
def load_scaler_2():
    return joblib.load('standard_scaler_2.pkl')

@st.cache_resource
def load_poly_features():
    return joblib.load('polynomial_features.pkl')

# Load the trained models



genres = ['Action','Adult','Adventure''Animation','Biography','Comedy','Crime','Documentary','Drama','Family','Fantasy',
          'History','Horror','Music','Musical','Mystery','News','Romance','Sci-Fi','Short','Sport','Thriller','War','Western']

# Streamlit interface
st.title('OTT Movie Success Predictor 🍿')
st.write("Predicts the **Total Viewing Hours** of the movie on streaming platforms based on various features.")

# Input fields
genres_selected = st.multiselect('Genre', genres)

languages = load_languages()
language = st.selectbox('Language', languages, placeholder="Select a language...")

directors = load_directors()
director = st.selectbox('Director', directors, placeholder="Director")

# Select 5 main actors/actresses using separate selectbox components
actors = load_actors()
actor_1 = st.selectbox('Actor 1', actors, key='actor_1')
actor_2 = st.selectbox('Actor 2', actors, key='actor_2')
actor_3 = st.selectbox('Actor 3', actors, key='actor_3')
actor_4 = st.selectbox('Actor 4', actors, key='actor_4')
actor_5 = st.selectbox('Actor 5', actors, key='actor_5')

runtime = st.number_input('Runtime (in minutes)', min_value=10, max_value=300, step=1)

# Add a predict button
if st.button('Predict Viewing Hours'):
    # Ensure the necessary selections are made
    if len(genres_selected) == 0:
        st.error('Please select at least one genre.')
    else:
        # Combine all features into a dictionary
        features_dict = {
            'startYear': [1],
            'runtimeMinutes': [runtime],
            'director_name': [director],
            'actor_names': [[actor_1, actor_2, actor_3, actor_4, actor_5]]
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(features_dict)

        mlb = load_genre_encoder()
        df_genres = pd.DataFrame(mlb.transform([genres_selected]), columns=mlb.classes_, index=df.index)
        # Combine the original DataFrame with the new one-hot encoded genres DataFrame
        df = pd.concat([df, df_genres], axis=1)

        language_encoder = load_language_encoder()
        df['original_language_encoded'] = language_encoder.transform([language])[0]
        df2 = pd.read_csv('directors_encoded_name.csv')
        df2.rename(columns={'primaryName': 'director_name'}, inplace=True)
        # Merging the new dataframe with the mean encoding dataframe to get the encoded values
        df = df.merge(df2, on='director_name', how='left')
        # Flatten the actors_encoded list to separate columns
        actors_df = pd.DataFrame(df['actor_names'].to_list(), columns=[f'actor_{i+1}' for i in range(5)])
        # Merge each actor column with the encoded_actors_df to get the encoded values
        df1 = pd.read_csv('actors_encoded_name.csv')
        for i in range(5):
            actors_df = actors_df.merge(df1, how='left', left_on=f'actor_{i+1}', right_on='primaryName')
            actors_df = actors_df.drop(columns=[f'actor_{i+1}', 'primaryName'])
            actors_df = actors_df.rename(columns={'actors_encoded': f'actor_{i+1}_encoded'})

        actors_df = actors_df.fillna(200)
        # Combine the encoded actors columns back with the original dataframe
        df = pd.concat([df, actors_df], axis=1)
        df = df.drop(columns=['Game-Show', 'Reality-TV', 'Talk-Show', 'director_name', 'actor_names'])
        df_model1 = df.copy()

        scaler = load_scaler()
        df_model1 = scaler.transform(df_model1)

        model_1 = load_model_1()
        df['predicted_numVotes'] = model_1.predict(df_model1)
        df = df.drop(columns=['startYear', 'directors_encoded', 'actor_1_encoded', 'actor_2_encoded',
                              'actor_3_encoded', 'actor_4_encoded', 'actor_5_encoded'])
        df.insert(25, 'Days', [180])
        poly = load_poly_features()
        df_poly = poly.transform(df)

        scaler_2 = load_scaler_2()
        df_scaled = scaler_2.transform(df_poly)

        model_2 = load_model_2()
        viewing_hours = model_2.predict(df_scaled)
        rounded_viewing_hours = round(viewing_hours[0] / 100000) * 100000
        st.write(f'Predicted Viewing Hours: {rounded_viewing_hours}')

# Add the detailed note about the prediction
st.markdown("""
**Note:**
- The predicted viewing hours are based on data from global streaming platforms like Netflix and are for the first 6 months after release. Actual viewing hours may vary depending on the OTT platform's subscriber base and global reach.
- Adjusting the runtime may not directly impact the predicted viewing hours, as the machine learning model accounts for complex and indirect relationships among various features.
""")
