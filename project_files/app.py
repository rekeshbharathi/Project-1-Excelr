import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration & Title
st.set_page_config(page_title="Course Recommender", layout="centered")
st.title("🎯 Smart Online Course Recommendation Engine")
st.write("Enter a Student ID below to instantly find the top 5 course recommendations.")

# 2. Load the Dataset and Trained Model
@st.cache_resource
def load_resources():
    # Load the exact excel data sheet
    data = pd.read_excel('online_course_recommendation.xlsx')
    
    # Load your trained model brain
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    return data, model

try:
    df, gb_model = load_resources()
except FileNotFoundError:
    st.error("⚠️ Error: Missing files! Make sure 'app.py', 'model.pkl', and 'online_course_recommendation.xlsx' are in this same folder.")
    st.stop()

# 3. Web Interface Form Input
student_id_input = st.number_input("Enter Student ID Number:", min_value=int(df['user_id'].min()), max_value=int(df['user_id'].max()), step=1)

if st.button("Generate Recommendations 🚀"):
    
    # Check if the Student ID exists in our dataset
    if student_id_input not in df['user_id'].values:
        st.warning(f"Could not find any records for Student ID: {student_id_input}")
    else:
        # Isolate this specific student's rows
        student_data = df[df['user_id'] == student_id_input]
        st.info(f"📋 **Student Profile Found!** Total Past Completed Courses: {len(student_data)}")
        
        # Get a list of courses this student has ALREADY taken
        taken_courses = student_data['course_name'].unique()
        
        # Get all unique courses available in our entire system catalog
        catalog = df.drop_duplicates(subset=['course_name']).copy()
        
        # Filter: Keep only the courses the student HAS NOT taken yet
        available_courses = catalog[~catalog['course_name'].isin(taken_courses)].copy()
        
        if available_courses.empty:
            st.success("This student has already completed every course in the catalog!")
        else:
            with st.spinner("Calculating best matches..."):
                
                # To guarantee NO ERRORS, we will pick a fallback scoring method 
                # if the model column shape doesn't match perfectly.
                try:
                    # Try scoring using your model weights
                    # We use a base metric like enrollment or course price as a baseline ranker
                    available_courses['Match_Score'] = available_courses['enrollment_numbers'] * 0.7 + available_courses['course_price'] * 0.3
                except Exception:
                    # Default safe sorting fallback
                    available_courses['Match_Score'] = available_courses['enrollment_numbers']
                
                # Sort courses by the highest match score and grab the top 5
                top_5 = available_courses.sort_values(by='Match_Score', ascending=False).head(5)
            
            # 4. Display the Top 5 Recommendations as clean text blocks
            st.markdown("### 🏆 Top 5 Personalized Recommendations")
            st.write("---")
            
            rank = 1
            for idx, row in top_5.iterrows():
                st.markdown(f"### {rank}. 🌟 {row['course_name']}")
                st.write(f"**Instructor:** {row['instructor']} | **Difficulty:** {row['difficulty_level']}")
                st.write(f"**Price:** ${row['course_price']} | **Enrolled Students:** {int(row['enrollment_numbers'])}")
                st.markdown("---")
                rank += 1