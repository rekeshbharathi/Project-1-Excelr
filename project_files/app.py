import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =====================================================================
# 1. INITIAL SETUP AND CACHED LOADING
# =====================================================================
st.set_page_config(page_title="Course Recommender App", layout="centered")

@st.cache_resource
def load_saved_pipeline_objects():
    """Loads and holds the trained objects in memory so the app stays lightning-fast."""
    # Loading your raw dataset to read user histories
    data = pd.read_excel('online_course_recommendation.xlsx')
    
    # Pre-processing baseline steps to align with your notebook variables
    # (Setting up your ordinal experience tiers so the mapping stays intact)
    data['User_Experience_Tier'] = pd.cut(data['previous_courses_taken'], 
                                         bins=[-1, 2, 6, np.inf], 
                                         labels=[0, 1, 2]).astype(int)
    
    difficulty_map = {'Beginner': 0, 'Intermediate': 1, 'Advanced': 2}
    data['difficulty_level_enc'] = data['difficulty_level'].map(difficulty_map)
    
    # Loading your saved machine learning objects
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
        
    return data, model, scaler, tfidf

# Run the loader
try:
    df, gb_model, scaler, tfidf = load_saved_pipeline_objects()
    # Grabbing the structural feature order dynamically
    # This list must match the exact sequence you passed your scaler in the notebook
    feature_columns = ['previous_courses_taken', 'enrollment_numbers', 'course_price',
                       'User_Experience_Tier', 'certification_offered_enc', 
                       'difficulty_level_enc', 'study_material_available_enc', 
                       'Diligence_Ratio', 'Price_Per_Hour', 'Difficulty_vs_Experience'] + \
                      [f"word_{w}" for w in tfidf.get_feature_names_out()]
except FileNotFoundError:
    st.error("⚠️ Error: Could not find your saved export files (model.pkl, scaler.pkl, tfidf.pkl, or the excel file). Make sure they are saved in this same folder directory!")
    st.stop()


# =====================================================================
# 2. THE USER INTERFACE LAYOUT (FRONT-END)
# =====================================================================
st.title("🎯 Smart Online Course Recommendation Engine")
st.write("Enter a Student ID below to instantly scan the course catalog and compute personalized top-5 learning suggestions using our trained Gradient Boosting pipeline.")

# Input box for user query
student_id_input = st.number_input("Enter Student ID Number:", min_value=int(df['user_id'].min()), max_value=int(df['user_id'].max()), step=1)

# Action execution trigger button
if st.button("Generate Recommendations 🚀"):
    
    # Step A: Double check student existence
    if student_id_input not in df['user_id'].values:
        st.warning(f"Could not find any student records for ID: {student_id_input}")
    else:
        # Step B: Isolate user characteristics
        user_rows = df[df['user_id'] == student_id_input]
        current_tier = int(user_rows['User_Experience_Tier'].iloc[0])
        total_past_courses = int(user_rows['previous_courses_taken'].iloc[0])
        
        st.info(f"📋 **Student Profile Found:** Experience Tier {current_tier} | Total Completed Courses: {total_past_courses}")
        
        # Step C: Isolate available catalog paths
        course_catalog = df.drop_duplicates(subset=['course_name']).copy()
        past_completed = user_rows['course_name'].unique()
        available_courses = course_catalog[~course_catalog['course_name'].isin(past_completed)].copy()
        
        if available_courses.empty:
            st.success("This student has already finished every single course available in our catalog!")
        else:
            with st.spinner("Running Gradient Boosting Inference Math..."):
                # Step D: Apply your engineered cross-feature equations
                available_courses['Diligence_Ratio'] = 1.0
                available_courses['Price_Per_Hour'] = available_courses['course_price'] / (available_courses['course_duration_hours'] + 1e-5)
                available_courses['Difficulty_vs_Experience'] = available_courses['difficulty_level_enc'] - current_tier
                
                # Step E: Transform textual strings via TF-IDF weights
                text_matrix = tfidf.transform(available_courses['course_name'])
                text_df = pd.DataFrame(
                    text_matrix.toarray(),
                    columns=[f"word_{w}" for w in tfidf.get_feature_names_out()],
                    index=available_courses.index
                )
                
                # Step F: Assemble and scale the matrix to align with your model shapes
                base_drops = ['enrollment_numbers', 'previous_courses_taken', 'course_price', 'User_Experience_Tier', 'certification_offered_enc', 'difficulty_level_enc', 'study_material_available_enc']
                inference_X = available_courses[base_drops]
                inference_X = pd.concat([inference_X, text_df], axis=1)
                
                # Force exact training alignment column sequence
                inference_X['Diligence_Ratio'] = available_courses['Diligence_Ratio']
                inference_X['Price_Per_Hour'] = available_courses['Price_Per_Hour']
                inference_X['Difficulty_vs_Experience'] = available_courses['Difficulty_vs_Experience']
                inference_X = inference_X[feature_columns]
                
                # Scale values safely
                inference_X_scaled = scaler.transform(inference_X)
                
                # Step G: Compute prediction match probabilities using Column 1
                match_scores = gb_model.predict_proba(inference_X_scaled)[:, 1]
                available_courses['Match_Score'] = match_scores
                
                # Get top 5 results
                top_5_results = available_courses.sort_values(by='Match_Score', ascending=False).head(5)
                
            # Step H: Output clean visual cards to the browser screen
            st.markdown("### 🏆 Top 5 Personalized Recommendations")
            st.write("---")
            
            rank = 1
            for idx, row in top_5_results.iterrows():
                # Display individual course recommendation modules
                st.markdown(f"### {rank}. 🌟 {row['course_name']}")
                st.write(f"**Instructor:** {row['instructor']} | **Difficulty:** {row['difficulty_level']}")
                st.write(f"**Price:** ${row['course_price']:.2f} | **Match Confidence:** `{row['Match_Score']:.2%}`")
                st.markdown("---")
                rank += 1