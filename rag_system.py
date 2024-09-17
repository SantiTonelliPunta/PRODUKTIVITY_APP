# utils/rag_system.py

import ast
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# Inicializar el modelo SBERT
model = SentenceTransformer('all-mpnet-base-v2')

# Configurar el cliente de OpenAI usando variables de entorno
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

# Ruta al archivo CSV (deberás asegurarte de que el archivo esté accesible)
datafile_path = "1000_embeddings_store.csv"  # Asegúrate de subir este archivo a tu repositorio o usar almacenamiento adecuado

def str_to_array(s):
    """Convierte una cadena de texto de embedding a un array numpy."""
    try:
        return np.array(ast.literal_eval(s))
    except:
        return np.array([])

# Cargar el CSV y convertir los embeddings
corpus_df = pd.read_csv(datafile_path)
corpus_df['embedding'] = corpus_df['embeddings_str'].apply(str_to_array)

print(f"Cargados {len(corpus_df)} documentos con embeddings.")

def obtener_embedding(texto):
    """Obtiene el embedding de un texto usando SBERT."""
    return model.encode([texto])[0]

def recuperar_documentos(query, corpus_df, top_n=5):
    """Recupera los documentos más similares a la consulta."""
    query_embedding = obtener_embedding(query)

    corpus_embeddings = np.vstack(corpus_df['embedding'].values)

    # Normalizar los embeddings
    query_embedding = normalize([query_embedding])[0]
    corpus_embeddings = normalize(corpus_embeddings)

    similitudes = cosine_similarity([query_embedding], corpus_embeddings)[0]
    corpus_df['similaridad'] = similitudes
    documentos_recuperados = corpus_df.sort_values(by='similaridad', ascending=False).head(top_n)

    return documentos_recuperados

def generar_respuesta_y_analizar_sentimiento(query, documentos_relevantes):
    """Genera una respuesta usando OpenAI basada en los documentos relevantes y analiza el sentimiento de las reseñas."""
    contexto = "\n".join(documentos_relevantes['text'].tolist())
    prompt = f"""
Eres un asistente virtual para el proyecto de análisis de reseñas de productos en Amazon. Tu objetivo es transformar el análisis de reseñas en una herramienta estratégica para el desarrollo de productos y la inteligencia de mercado. Proporcionas información precisa y relevante basada en las reseñas de productos, beneficiando a empresas B2B en España, como fabricantes, vendedores en Amazon, agencias de marketing digital, plataformas de e-commerce e inversores.

Debes utilizar las reseñas de productos de Amazon disponibles en el corpus, así como la base de datos de embeddings ubicada en `/content/drive/My Drive/001_AI_MASTER/TFM_grupo2_Master_Inesdi/9. Notebooks/data_sets_MVP/Embeddings/1000_embeddings_store.csv` para obtener información adicional.

Eres capaz de responder en diferentes idiomas y, si no conoces el idioma del usuario, debes traducir la entrada y la salida al idioma que te han preguntado.

Personalización de la comunicación:
- Siempre personalizarás la conversación utilizando un tono profesional y experto.
- Sé directo y claro; si alguien te solicita información, provee detalles específicos.
- Utiliza el nombre del usuario, si es proporcionado, para personalizar la interacción.
- Usa el corpus de embeddings para buscar información adicional cuando sea necesario.

Tono de Voz y Trato:
1. Debe ser formal, pero cordial.
2. Sus respuestas deben ser agradables, fluidas, cordiales y con la intención de ayudar
3. Debe responder en el idioma que hable el usuario.

Restricciones y Limitaciones del Chatbot:
1. No puedes citar las fuentes que no sean del corpus de reseñas de Amazon y la base de datos de embeddings especificada.
2. No debes mencionar ni promocionar productos o servicios específicos que no estén basados en las reseñas proporcionadas.
3. No debes ser irrespetuoso, violento, sexista, y no debes tener comentarios impropios o que hieran los sentimientos de los usuarios.

Basándote en la siguiente información:

{contexto}

1. Responde a la siguiente pregunta: {query}

2. Analiza el sentimiento general de las reseñas proporcionadas y proporciona un resumen. Indica si las reseñas tienen un sentimiento positivo, negativo o neutral, y proporciona una puntuación de polaridad para reflejar la intensidad del sentimiento.
"""

    try:
        respuesta = client.chat.completions.create(
            model="gpt-4",  # Usar GPT-4
            messages=[
                {"role": "system", "content": "Eres un asistente experto en análisis de mercado basado en reseñas de productos de Amazon."},
                {"role": "user", "content": prompt}
            ]
        )
        respuesta_texto = respuesta.choices[0].message.content

        # Analizar sentimiento y polaridad
        sentimiento_prompt = f"""
A continuación, se presentan algunas reseñas de productos en Amazon:

{contexto}

Analiza el sentimiento de estas reseñas y proporciona una evaluación general. Indica si el sentimiento es positivo, negativo o neutral. Además, proporciona una puntuación de polaridad en una escala de -1 a 1, donde -1 es completamente negativo, 0 es neutral, y 1 es completamente positivo.
"""
        sentimiento = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un analista de sentimientos experto."},
                {"role": "user", "content": sentimiento_prompt}
            ]
        )
        sentimiento_texto = sentimiento.choices[0].message.content

        # Combinar respuestas
        respuesta_final = f"{respuesta_texto}\n\nAnálisis de Sentimiento:\n{sentimiento_texto}"
        return respuesta_final
    except Exception as e:
        return f"Error al generar la respuesta: {str(e)}"

# Exportar corpus_df para uso en app.py
__all__ = ['recuperar_documentos', 'generar_respuesta_y_analizar_sentimiento', 'corpus_df']