
import pandas as pd
import random

def load_embeddings(file_path):
    # Cargar los embeddings desde el archivo CSV
    return pd.read_csv(file_path)

def generate_random_questions(embeddings_df, num_questions=30):
    # Generar preguntas aleatorias basadas en el campo 'title'
    questions = ["Pregunta sobre: " + row['title'] for _, row in embeddings_df.iterrows()]
    return random.sample(questions, num_questions)
