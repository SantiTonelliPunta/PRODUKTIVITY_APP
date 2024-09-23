from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_gpt_questions(embeddings_df, num_questions=30):
    questions = []
    for _, row in embeddings_df.sample(n=num_questions).iterrows():
        prompt = f"Genera una pregunta detallada sobre el siguiente producto o reseña: '{row['title']}'. La pregunta debe ser relevante para una tienda online y útil para entender mejor el producto o la experiencia del cliente."
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en generar preguntas detalladas sobre productos y reseñas de una tienda online."},
                    {"role": "user", "content": prompt}
                ]
            )
            question = response.choices[0].message.content.strip()
            questions.append(question)
        except Exception as e:
            print(f"Error al generar pregunta: {e}")
            questions.append(f"Pregunta sobre: {row['title']}")  # Fallback a la versión original
    
    return questions
