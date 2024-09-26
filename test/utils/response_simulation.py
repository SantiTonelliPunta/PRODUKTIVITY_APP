import os
from dotenv import load_dotenv
from openai import OpenAI
from ..config import GPT_MODEL, MAX_TOKENS

# Cargar variables de entorno desde .env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Inicializar el cliente de OpenAI con la API key del archivo .env
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def simulate_responses(questions):
    """
    Simula respuestas utilizando GPT-3.5 Turbo para las preguntas generadas.

    Args:
    questions (list): Lista de diccionarios, cada uno conteniendo una pregunta y su embedding asociado.

    Returns:
    list: Lista de diccionarios, cada uno conteniendo la pregunta original, la respuesta simulada y el embedding.
    """
    responses = []
    for q in questions:
        prompt = f"Responde a la siguiente pregunta sobre un producto o servicio: {q['question']}"
        
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un asistente que responde preguntas sobre productos y servicios basándote en reseñas de usuarios."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=MAX_TOKENS
            )
            
            answer = response.choices[0].message.content.strip()
            responses.append({"question": q['question'], "response": answer, "embedding": q['embedding']})
        except Exception as e:
            print(f"Error al generar respuesta: {str(e)}")
            responses.append({"question": q['question'], "response": "Error al generar respuesta", "embedding": q['embedding']})
    
    return responses
