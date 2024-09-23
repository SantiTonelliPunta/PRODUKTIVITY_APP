import pandas as pd
from .gpt_question_generator import generate_gpt_questions

def load_embeddings(file_path):
    return pd.read_csv(file_path)

def select_random_embeddings(embeddings_df, num_embeddings=30):
    return embeddings_df.sample(n=num_embeddings)

def generate_random_questions(embeddings_df, num_questions=30):
    return generate_gpt_questions(embeddings_df, num_questions)