import pandas as pd
from googletrans import Translator
from ast import literal_eval
import time
from tqdm import tqdm  # For progress visualization

# Initialize translator
translator = Translator()

# Function to load and process the dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row['problems'])
        record = {
            'id': row['id'],
            'paragraph': row['paragraph'],
            'problems': problems,
            'question_plus': row['question_plus'],
        }
        records.append(record)
    return pd.DataFrame(records)

# Function to calculate average lengths of choices
def calculate_average_choice_length(df):
    df['average_choice_length'] = df['problems'].apply(
        lambda x: sum(len(choice) for choice in x['choices']) / len(x['choices']) if x['choices'] else 0
    )
    return df

# Function to back-translate text with delay
def back_translate(text, src_language='en', intermediate_language='fr', delay=1):
    translated = translator.translate(text, src=src_language, dest=intermediate_language).text
    time.sleep(delay)
    back_translated = translator.translate(translated, src=intermediate_language, dest=src_language).text
    time.sleep(delay)
    return back_translated

# Main function to process data
def process_data(file_path, output_path):
    # Load data
    print("Loading data...")
    df = load_data(file_path)

    # Calculate average lengths of choices
    print("Calculating average choice lengths...")
    df = calculate_average_choice_length(df)

    # Prepare list for augmented data
    augmented_records = []

    print("Performing back-translation... This may take some time.")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Back-translating rows"):
        original_problems = row['problems']
        augmented_problems = original_problems.copy()

        # Back-translate question for all rows
        augmented_problems['question'] = back_translate(original_problems['question'])

        # Back-translate choices only if average length > 16
        if row['average_choice_length'] > 16:
            augmented_problems['choices'] = [back_translate(choice) for choice in original_problems['choices']]
        else:
            augmented_problems['choices'] = original_problems['choices']

        # Add augmented record
        augmented_records.append({
            'id': row['id'],
            'paragraph': row['paragraph'],
            'problems': augmented_problems,
            'question_plus': row['question_plus']
        })

    # Combine original and augmented data
    print("Combining original and augmented data...")
    original_records = df[['id', 'paragraph', 'problems', 'question_plus']].to_dict(orient='records')
    combined_records = original_records + augmented_records

    # Convert combined records to DataFrame and save
    print("Saving augmented dataset...")
    combined_df = pd.DataFrame(combined_records)
    combined_df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to {output_path}")

# Define file paths
input_file = './data/train.csv'  # Path to the input CSV file
output_file = './data/augmented_bt_question_choice_train.csv'  # Path to save the augmented CSV file

# Run the processing function
process_data(input_file, output_file)


#todo api호출이 있고 시간이 오래걸리므로 역번역 결과를 파일에 계속 갱신해줘서 중간에 멈춰도 큰 지장 없도록 할 것 -> 이전 성공을 기록해두고
#이어서 실행 가능하도록 수정하자