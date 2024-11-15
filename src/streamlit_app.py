import streamlit as st
import pandas as pd
import os
import re
from ast import literal_eval
from transformers import AutoTokenizer

# Load the tokenizer
model_name = st.text_input("Enter the model name for tokenizer", "beomi/gemma-ko-2b")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Function to load and process selected CSV file
def load_data(file_path):
    dataset = pd.read_csv(file_path)
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
    return pd.DataFrame(records)

# Streamlit EDA App
st.title("Train Dataset EDA")

# Select CSV file from data folder
data_folder = './data/'
csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

# Set default index for 'train.csv' if it exists, otherwise default to 0
default_index = csv_files.index('train.csv') if 'train.csv' in csv_files else 0

# Select CSV file from data folder with 'train.csv' as the default if available
selected_file = st.selectbox("Select a CSV file", csv_files, index=default_index)
file_path = os.path.join(data_folder, selected_file)
# Load selected data
df = load_data(file_path)

st.subheader("Pattern-Based ID Lookup")
number = st.text_input("Enter a number to match in IDs (e.g., 'generation-for-nlp-xxx')")

if number.isdigit():
    # Create pattern based on the input number
    pattern = r'(^|[^0-9])(' + number + r')$'
    
    # Filter DataFrame by pattern
    matched_rows = df[df['id'].astype(str).str.contains(pattern, regex=True)]
    
    # Display matching rows with tokenization results
    if not matched_rows.empty:
        # Add tokenized paragraph and question columns for inspection
        matched_rows['paragraph_tokens'] = matched_rows['paragraph'].apply(lambda x: tokenizer.tokenize(x))
        matched_rows['question_tokens'] = matched_rows['question'].apply(lambda x: tokenizer.tokenize(x))
        
        for _, row in matched_rows.iterrows():
            st.markdown(f"### ID: {row['id']}")
            
            with st.expander("Paragraph"):
                st.write(row['paragraph'])
            
            with st.expander("Paragraph Tokens"):
                st.write(row['paragraph_tokens'])
            
            with st.expander("Question"):
                st.write(row['question'])
            
            with st.expander("Question Tokens"):
                st.write(row['question_tokens'])

            st.markdown("---")  # Separator for better readability
    else:
        st.write("No matching IDs found based on the pattern.")
else:
    st.write("Please enter a valid number for pattern matching.")

# Show all data
st.subheader("Full Dataset")
st.write(df.reset_index(drop=True))


# Length Distributions
st.subheader("Length Distribution for paragraph, question")

# Paragraph Length Distribution
df['paragraph_length'] = df['paragraph'].apply(len)
st.bar_chart(df['paragraph_length'], use_container_width=True)

# Question Length Distribution
df['question_length'] = df['question'].apply(len)
st.bar_chart(df['question_length'], use_container_width=True)

# Tokenized Length Distributions
st.subheader("Tokenized Length Distribution for paragraph, question")

# Tokenize paragraph and question, then calculate tokenized length using the tokenizer
df['paragraph_token_length'] = df['paragraph'].apply(lambda x: len(tokenizer.tokenize(x)))
df['question_token_length'] = df['question'].apply(lambda x: len(tokenizer.tokenize(x)))

# Display tokenized length distributions
st.bar_chart(df['paragraph_token_length'], use_container_width=True)
st.bar_chart(df['question_token_length'], use_container_width=True)

# Filter and display data by answer value
st.subheader("Answer Distribution")
st.write(df['answer'].value_counts())