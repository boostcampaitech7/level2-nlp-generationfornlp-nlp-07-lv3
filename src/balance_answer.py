import pandas as pd
from ast import literal_eval
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Load dataset
dataset = pd.read_csv('./data/train.csv')

# Flatten the JSON dataset while keeping original structure
records = []
for _, row in dataset.iterrows():
    problems = literal_eval(row['problems'])
    record = {
        'id': row['id'],
        'paragraph': row['paragraph'],
        'problems': problems,
        'question_plus': row.get('question_plus', None),
    }
    records.append(record)

# Convert to DataFrame
df = pd.DataFrame(records)

# Filter for rows where choices have 5 elements
df_5_choices = df[df['problems'].apply(lambda x: len(x['choices']) == 5)].copy()

# Balance the answer distribution
def redistribute_answers(problems):
    """Redistribute the answer and adjust choices to balance the distribution."""
    choices = problems['choices']
    current_answer = problems['answer'] - 1  # Convert to 0-based index
    num_choices = len(choices)

    # Select a new answer index (ensure it is different from the current one)
    new_answer = random.choice([i for i in range(num_choices)])

    # Swap the current answer with the new answer in the choices
    choices = choices[:]
    choices[current_answer], choices[new_answer] = choices[new_answer], choices[current_answer]

    # Update the problems dictionary
    problems['choices'] = choices
    problems['answer'] = new_answer + 1  # Convert back to 1-based index

    return problems

# Apply redistribution to problems column
df_5_choices['problems'] = df_5_choices['problems'].apply(redistribute_answers)

# Update the original DataFrame with modified problems
df.loc[df['problems'].apply(lambda x: len(x['choices']) == 5), 'problems'] = df_5_choices['problems']

# Save the updated DataFrame
df.to_csv('./data/updated_train.csv', index=False)

