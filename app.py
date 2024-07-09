import streamlit as st
import pandas as pd
import joblib
import json


# Load the trained models
model_1 = joblib.load('voting_regressor.pkl')
model_2 = joblib.load('gradient_boosting_regressor.pkl')

language_encoder = joblib.load('label_encoder_languages.pkl')

# Load the one-hot encoder
mlb = joblib.load('genre_onehot_encoder.pkl')

# Extract the genres
genres = mlb.classes_.tolist()

genres = [genre for genre in genres if genre not in ['Talk-Show', 'Game-Show', 'Reality-TV']]

# Load the JSON file
with open('language_mapping.json', 'r') as f:
    languages_dict = json.load(f)

# Convert dictionary values to a list of languages
languages = list(languages_dict.values())

df2 = pd.read_csv('directors_encoded_name.csv')
directors = df2['primaryName'].to_list()

df1 = pd.read_csv('actors_encoded_name.csv')
actors = df1['primaryName'].to_list()

# Streamlit interface
st.title('OTT Movie Success Predictor 🍿')
st.write("Predicts the **Total Viewing Hours** of the movie on streaming platforms based on various features.")


# Input fields
genres_selected = st.multiselect('Genre', genres)

language = st.selectbox('Language', languages, placeholder= "Select a language...")

director = st.selectbox('Director', directors, placeholder= "Director")

actors_selected = st.multiselect('Select 5 Main Actors/Actresses', actors, max_selections=5)

runtime = st.number_input('Runtime (in minutes)', min_value=10, max_value=300, step=1)

# Add a predict button
if st.button('Predict Viewing Hours'):
    # Ensure the necessary selections are made
    if len(genres_selected) == 0 or len(actors_selected) != 5:
        st.error('Please select at least one genre and exactly 5 actors/actresses.')
    else:


        # Combine all features into a dictionary
        features_dict = {
            'startYear': [1],
            'runtimeMinutes': [runtime],
            'director_name': [director],
            'actor_names': [actors_selected]
        }

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(features_dict)

        df_genres = pd.DataFrame(mlb.transform([genres_selected]), columns=mlb.classes_, index=df.index)



        # Combine the original DataFrame with the new one-hot encoded genres DataFrame
        df = pd.concat([df, df_genres], axis=1)

        df['original_language_encoded'] = language_encoder.transform([language])[0]

        df2.rename(columns={'primaryName': 'director_name'}, inplace=True)
        # Merging the new dataframe with the mean encoding dataframe to get the encoded values
        df = df.merge(df2, on='director_name', how='left')

        # Flatten the actors_encoded list to separate columns
        actors_df = pd.DataFrame(df['actor_names'].to_list(), columns=[f'actor_{i+1}' for i in range(5)])

        # Merge each actor column with the encoded_actors_df to get the encoded values
        for i in range(5):
            actors_df = actors_df.merge(df1, how='left', left_on=f'actor_{i+1}', right_on='primaryName')
            actors_df = actors_df.drop(columns=[f'actor_{i+1}', 'nconst','primaryName'])
            actors_df = actors_df.rename(columns={'actors_encoded': f'actor_{i+1}_encoded'})

        actors_df = actors_df.fillna(200)
        # Combine the encoded actors columns back with the original dataframe
        df = pd.concat([df, actors_df], axis=1) 

        df = df.drop(columns=['Game-Show','Reality-TV','Talk-Show','director_name', 'nconst', 'actor_names'])

        df_model1 = df.copy()

        scaler = joblib.load('standard_scaler.pkl')

        df_model1 = scaler.transform(df_model1)

        df['predicted_numVotes'] = model_1.predict(df_model1)

        df = df.drop(columns=['startYear','directors_encoded','actor_1_encoded','actor_2_encoded',
                               'actor_3_encoded', 'actor_4_encoded','actor_5_encoded'])
        
        df.insert(25, 'Days', [180])

        
        poly = joblib.load('polynomial_features.pkl')
        df_poly = poly.transform(df)

        scaler_2 = joblib.load('standard_scaler_2.pkl')
        df_scaled = scaler_2.transform(df_poly)

        viewing_hours = model_2.predict(df_scaled)
        rounded_viewing_hours = round(viewing_hours[0] / 100000) * 100000
        st.write(f'Predicted Viewing Hours: {rounded_viewing_hours}')
    
    # Add the detailed note about the prediction
st.markdown("""
**Note:**
The predicted viewing hours are for the first 6 months after its release and are based on data from global streaming platforms like Netflix. The actual viewing hours may vary depending on the number of subscribers of the OTT platform it gets released on and its global reach.
""")