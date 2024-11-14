import streamlit as st
import pandas as pd
from ast import literal_eval

# Load the train dataset
# TODO: Train Data 경로 입력
dataset = pd.read_csv('./data/train.csv') # streamlit run

# Flatten the JSON dataset
records = []
for _, row in dataset.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'question': problems['question'],
        'choices': problems['choices'],
        'answer': problems.get('answer', None),
        'question_plus': problems.get('question_plus', None),
    }
    records.append(record)
        
# Convert to DataFrame
df = pd.DataFrame(records)

# Streamlit app for EDA
st.title("Train Dataset EDA")

# Display DataFrame head
st.subheader("Dataset Preview")
st.write(df.head())

# Paragraph Length Distribution
st.subheader("Paragraph Length Distribution")
df['paragraph_length'] = df['paragraph'].apply(len)
st.bar_chart(df['paragraph_length'])

# Question Length Distribution
st.subheader("Question Length Distribution")
df['question_length'] = df['question'].apply(len)
st.bar_chart(df['question_length'])

# Answer Value Counts
st.subheader("Answer Distribution")
st.write(df['answer'].value_counts())

# Filtered Display
st.subheader("Filtered Data Exploration")
answer_filter = st.selectbox("Filter by Answer", options=df['answer'].unique())
st.write(df[df['answer'] == answer_filter])

# st.sidebar.header("Options")
# if st.sidebar.checkbox("Show Raw Data"):
#     st.subheader("Raw Data")
#     st.write(df)

