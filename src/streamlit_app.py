import streamlit as st
import pandas as pd
import os
import re
from ast import literal_eval
from transformers import AutoTokenizer
from kiwipiepy import Kiwi

kiwi = Kiwi()

# Define fields to display
fields_to_display = {
    "Paragraph": "paragraph",
    "Paragraph Tokens": "paragraph_tokens",
    "Question": "question",
    "Question Tokens": "question_tokens",
    "Choices": "choices",
    "Answer": "answer",
    "Choices Tokens": "choices_tokens",
    "Common Nouns": "common_nouns",
}


def load_tokenizer():
    model_name = st.text_input("Enter the model name for tokenizer", "beomi/gemma-ko-2b")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer = load_tokenizer()

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

# 공통 명사 추출 함수
def extract_common_nouns(paragraph, question):
    # 형태소 분석 후 NNG, NNP 태깅된 단어 추출 - 일반명사, 고유명사
    paragraph_nouns = set(word[0] for word in kiwi.analyze(paragraph)[0][0] if word[1] in ['NNG', 'NNP'])
    question_nouns = set(word[0] for word in kiwi.analyze(question)[0][0] if word[1] in ['NNG', 'NNP'])
    
    # 공통 명사 추출
    common_nouns = paragraph_nouns & question_nouns
    return list(common_nouns)

def display_expander(title, content):
    """Streamlit expander helper to reduce repetitive code."""
    with st.expander(title):
        st.write(content)

def display_row_with_expanders(row, fields):
    """
    Display multiple sections in a single Streamlit row using expanders.

    Args:
        row (pd.Series): A row of the DataFrame.
        fields (dict): A dictionary mapping expander titles to row keys.
    """
    st.markdown(f"### ID: {row['id']}")
    for title, key in fields.items():
        display_expander(title, row[key])  # Reuse the expander function
    st.markdown("---")  # Separator for better readability

def pattern_based_id_lookup(df):
    """Pattern-based ID lookup and matching."""
    st.subheader("Pattern-Based ID Lookup")
    number = st.text_input("Enter a number to match in IDs (e.g., 'generation-for-nlp-xxx')")

    if number.isdigit():
        # Create pattern based on the input number
        pattern = r'(?:^|[^0-9])' + number + r'$'

        # Filter DataFrame by pattern
        matched_rows = df[df['id'].astype(str).str.contains(pattern, regex=True)]

        if not matched_rows.empty:
            # Add tokenized paragraph, question, and choices columns for inspection
            matched_rows['paragraph_tokens'] = matched_rows['paragraph'].apply(lambda x: tokenizer.tokenize(x))
            matched_rows['question_tokens'] = matched_rows['question'].apply(lambda x: tokenizer.tokenize(x))
            matched_rows['choices_tokens'] = matched_rows['choices'].apply(lambda choices: [tokenizer.tokenize(choice) for choice in choices])
            matched_rows['common_nouns'] = matched_rows.apply(lambda row: extract_common_nouns(row['paragraph'], row['question']), axis=1)

            for _, row in matched_rows.iterrows():
                display_row_with_expanders(row, fields_to_display)
                st.markdown("---")
        else:
            st.write("No matching IDs found based on the pattern.")
    else:
        st.write("Please enter a valid number for pattern matching.")

def display_full_dataset(df):
    """Display the full dataset."""
    st.subheader("Full Dataset")
    st.write(df.reset_index(drop=True))

def plot_length_distributions(df):
    """Plot length distributions for paragraphs and questions."""
    st.subheader("Length Distribution for paragraph, question")

    # Paragraph Length Distribution
    df['paragraph_length'] = df['paragraph'].apply(len)
    st.bar_chart(df['paragraph_length'], use_container_width=True)
    st.write(df['paragraph_length'].describe())

    # Question Length Distribution
    df['question_length'] = df['question'].apply(len)
    st.bar_chart(df['question_length'], use_container_width=True)
    st.write(df['question_length'].describe())

def plot_tokenized_length_distributions(df):
    """Plot tokenized length distributions for paragraphs and questions."""
    st.subheader("Tokenized Length Distribution for paragraph, question")

    # Tokenize paragraph and question, then calculate tokenized length
    df['paragraph_token_length'] = df['paragraph'].apply(lambda x: len(tokenizer.tokenize(x)))
    df['question_token_length'] = df['question'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Display tokenized length distributions
    st.bar_chart(df['paragraph_token_length'], use_container_width=True)
    st.write(df['paragraph_token_length'].describe())
    st.bar_chart(df['question_token_length'], use_container_width=True)
    st.write(df['question_token_length'].describe())

def display_answer_distribution(df):
    """Display the answer value distribution."""
    st.subheader("Answer Distribution")
    st.write(df['answer'].value_counts())

def plot_chat_template_token_length_distribution(df):
    """Plot tokenized length distributions after applying the chat template."""
    MAX_SEQ_LENGTH = 1024
    
    st.subheader(f"Tokenized Length Distribution After Applying Chat Template, MAX_SEQ_LENGTH : {MAX_SEQ_LENGTH}")

    # Define a default global system message
    DEFAULT_SYSTEM_MESSAGE = "지문을 읽고 질문의 답을 구하세요."


    # Function to apply the chat template
    def apply_chat_template(row):
        messages = [
            {"role": "system", "content": row.get('system_message', DEFAULT_SYSTEM_MESSAGE)},
            {"role": "user", "content": row['paragraph']},
            {"role": "user", "content": f"Question: {row['question']}"},
            {"role": "user", "content": f"Choices: {', '.join(row['choices'])}"},
            {"role": "assistant", "content": f"Answer: {row['answer']}"}
        ]

        # Apply the chat template logic
        result = []
        system_message = None
        if messages[0]['role'] == 'system':
            system_message = messages[0]['content']
        if system_message:
            result.append(system_message)
        for message in messages:
            content = message['content']
            if message['role'] == 'user':
                result.append(f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n")
            elif message['role'] == 'assistant':
                result.append(f"{content}<end_of_turn>\n")
        return ''.join(result)

    # Apply the chat template to generate combined text for each row
    df['chat_template_text'] = df.apply(apply_chat_template, axis=1)

    # Tokenize the combined text and calculate token lengths
    df['chat_template_token_length'] = df['chat_template_text'].apply(lambda x: len(tokenizer.tokenize(x)))

    # Display statistics
    st.write("Statistics for tokenized chat template lengths:")
    st.write(df['chat_template_token_length'].describe())

    # Identify samples exceeding 1024 tokens (or other limit)
    long_samples = df[df['chat_template_token_length'] > MAX_SEQ_LENGTH]
    st.write(f"Samples exceeding {MAX_SEQ_LENGTH} tokens: {len(long_samples)}")
    if not long_samples.empty:
        st.write(long_samples[['id', 'chat_template_token_length']].reset_index(drop=True))

    st.subheader("Lookup by ID")
    number = st.text_input("Enter a number to match in IDs (e.g., 'generation-for-nlp-xxx')", value=550)

    if number.isdigit():
        # Create pattern based on the input number
        pattern = r'(?:^|[^0-9])' + number + r'$'

        # Filter the row by ID
        matched_row = df[df['id'].astype(str).str.contains(pattern, regex=True)]

        if not matched_row.empty:
            # Convert Series to string for tokenization
            sample_text = matched_row.iloc[0]['chat_template_text']  # Extract the first matching row's text
            st.write("### Chat Template Applied Text")
            st.text(sample_text)

            st.write("### Tokenized Result")
            tokenized_result = tokenizer.tokenize(sample_text)
            st.write(tokenized_result)

            st.write("### Token Count")
            st.write(len(tokenized_result))
        else:
            st.write("No matching ID found.")


def main():
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

     # Define tabs and corresponding functions
    tabs = {
        "Pattern Lookup": pattern_based_id_lookup,
        "Full Dataset": display_full_dataset,
        "Length Distributions": plot_length_distributions,
        "Tokenized Distributions": plot_tokenized_length_distributions,
        "Chat Template Tokenization": plot_chat_template_token_length_distribution,
        "Answer Distribution": display_answer_distribution
    }

    # Create tabs
    tab_names = list(tabs.keys())
    selected_tab = st.tabs(tab_names)

    # Render content for each tab
    for name, tab in zip(tab_names, selected_tab):
        with tab:
            tabs[name](df)  # Call the corresponding function with `df`


if __name__ == '__main__':
    main()