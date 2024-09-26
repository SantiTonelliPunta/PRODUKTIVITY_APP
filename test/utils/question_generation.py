import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from ..config import GPT_MODEL, MAX_TOKENS

# Cargar variables de entorno desde .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Inicializar el cliente de OpenAI con la API key del archivo .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_questions(embeddings, num_questions):
    """
    Genera preguntas utilizando GPT-3.5 Turbo basadas en los embeddings.

    Args:
    embeddings (list): Lista de embeddings.
    num_questions (int): Número de preguntas a generar.

    Returns:
    list: Lista de diccionarios, cada uno conteniendo una pregunta y su embedding asociado.
    """
    questions = []
    for _ in range(num_questions):
        embedding = random.choice(embeddings)
        prompt = f"Genera una pregunta detallada sobre el siguiente producto o servicio: {embedding['title']}"
        
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un asistente que genera preguntas detalladas sobre productos y servicios."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS
            )
            
            question = response.choices[0].message.content.strip()
            questions.append({"question": question, "embedding": embedding})
        except Exception as e:
            print(f"Error al generar pregunta: {str(e)}")
            questions.append({"question": "Error al generar pregunta", "embedding": embedding})
    
    return questions
