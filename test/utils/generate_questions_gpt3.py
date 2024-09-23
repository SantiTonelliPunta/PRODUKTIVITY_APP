
import openai
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Asignar la clave de OpenAI desde el archivo .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cargar los embeddings desde el archivo CSV
def load_embeddings(file_path):
    return pd.read_csv(file_path)

# Seleccionar 30 embeddings aleatoriamente
def select_random_embeddings(embeddings_df, num_embeddings=30):
    return embeddings_df.sample(n=num_embeddings)

# Función para generar preguntas utilizando GPT-3.5 Turbo con el nuevo prompt
def generate_questions_with_gpt3(embedding_data, model="gpt-3.5-turbo"):
    questions = []
    
    # Prompt base según las indicaciones
    prompt_base = ("Eres un asistente especializado en generar respuestas creativas y variadas. "
                   "Tienes que proporcionar treinta respuestas diferentes, y todas deben estar relacionadas con las palabras clave proporcionadas. "
                   "Las respuestas deben ser claras, creativas, y pueden variar en longitud y estilo. "
                   "Cada respuesta debe ofrecer información útil o una reflexión interesante sobre cada palabra clave."
                  )

    for _, row in embedding_data.iterrows():
        # Usamos tanto el título como el texto completo de la reseña para proporcionar más contexto
        review_title = row['title']
        review_text = row['text']
        prompt = f"{prompt_base}

Palabras clave: {review_title}, {review_text}"
        
        # Llamada al API de OpenAI para generar preguntas basadas en las palabras clave
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un asistente que genera preguntas detalladas basadas en productos."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extraemos la respuesta generada
            question = response['choices'][0]['message']['content']
            questions.append(question)
        except Exception as e:
            questions.append(f"Error generating question: {e}")
    
    return questions

# Integramos todo en el flujo
def main():
    # Cargamos los embeddings
    embeddings_df = load_embeddings('embeddings/1000_embeddings_store.csv')
    
    # Seleccionamos 30 embeddings aleatoriamente
    selected_embeddings = select_random_embeddings(embeddings_df, num_embeddings=30)
    
    # Generamos preguntas usando GPT-3.5 Turbo
    questions = generate_questions_with_gpt3(selected_embeddings)
    
    # Mostramos las preguntas generadas
    for i, question in enumerate(questions):
        print(f"Pregunta {i+1}: {question}")

# Ejecutamos el flujo principal
if __name__ == "__main__":
    main()
