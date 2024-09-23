
import pandas as pd
import random

# Función para cargar los embeddings desde un archivo CSV
def load_embeddings(file_path):
    return pd.read_csv(file_path)

# Función para seleccionar embeddings aleatoriamente
def select_random_embeddings(embeddings_df, num_embeddings=30):
    return embeddings_df.sample(n=num_embeddings)

# Función para generar preguntas aleatorias a partir de los títulos
def generate_random_questions(embeddings_df, num_questions=30):
    questions = ["Pregunta sobre: " + row['title'] for _, row in embeddings_df.iterrows()]
    return random.sample(questions, num_questions)
