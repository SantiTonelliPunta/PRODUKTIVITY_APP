# utils/rag_system.py

import ast
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import time
import logging
from functools import lru_cache

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar el modelo SBERT
model = SentenceTransformer('all-mpnet-base-v2')

# Configurar el cliente de OpenAI usando variables de entorno
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Ruta al archivo CSV (usando ruta relativa)
base_dir = os.path.dirname(os.path.abspath(__file__))
datafile_path = os.path.join(base_dir, '..', 'embeddings', '1000_embeddings_store.csv')

def str_to_array(s):
    """Convierte una cadena de texto de embedding a un array numpy."""
    try:
        return np.array(ast.literal_eval(s))
    except:
        return np.array([])

# Variable global para almacenar los datos
corpus_df = None

def load_data():
    global corpus_df
    if corpus_df is None:
        start_time = time.time()
        try:
            corpus_df = pd.read_csv(datafile_path)
            corpus_df['embedding'] = corpus_df['embeddings_str'].apply(str_to_array)
            logging.info(f"Cargados {len(corpus_df)} documentos con embeddings en {time.time() - start_time:.2f} segundos.")
        except FileNotFoundError:
            logging.error(f"Error: El archivo {datafile_path} no fue encontrado.")
            corpus_df = pd.DataFrame(columns=['text', 'embeddings_str', 'embedding'])

# Cargar datos al inicio
load_data()

@lru_cache(maxsize=100)
def obtener_embedding(texto):
    """Obtiene el embedding de un texto usando SBERT con caché."""
    return tuple(model.encode([texto])[0])  # Convertir a tupla para que sea hashable

def recuperar_documentos(query, top_n=5):
    global corpus_df
    start_time = time.time()
    query_embedding = obtener_embedding(query)
    
    corpus_embeddings = np.vstack(corpus_df['embedding'].values)

    # Normalizar los embeddings
    query_embedding = normalize([query_embedding])[0]
    corpus_embeddings = normalize(corpus_embeddings)

    similitudes = cosine_similarity([query_embedding], corpus_embeddings)[0]
    corpus_df['similaridad'] = similitudes
    documentos_recuperados = corpus_df.sort_values(by='similaridad', ascending=False).head(top_n)

    logging.info(f"Recuperación de documentos completada en {time.time() - start_time:.2f} segundos.")
    return documentos_recuperados

@lru_cache(maxsize=100)
def generar_respuesta_y_analizar_sentimiento(query, documentos_relevantes_tuple):
    start_time = time.time()
    
    documentos_relevantes = list(documentos_relevantes_tuple)
    contexto = "\n".join(documentos_relevantes)

    # Verificar si es un saludo simple
    saludos = ["hola", "hello", "hi", "buenos días", "buenas tardes", "buenas noches"]
    if query.lower().strip() in saludos:
        respuesta = "Hola, ¿en qué puedo ayudarte hoy con respecto a la búsqueda y análisis de productos?"
        tiempo_total = time.time() - start_time
        return respuesta, tiempo_total

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
        respuesta_start_time = time.time()
        respuesta = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de reseñas de Amazon y en proporcionar información precisa y relevante sobre productos y mercado."},
                {"role": "user", "content": prompt}
            ]
        )
        respuesta_texto = respuesta.choices[0].message.content
        tiempo_total = time.time() - start_time
        return respuesta_texto, tiempo_total
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {str(e)}")
        return f"Error al generar la respuesta: {str(e)}", time.time() - start_time

def procesar_consulta(query):
    logging.info(f"Iniciando procesamiento de consulta: {query}")
    try:
        if query.lower().strip() in ["hola", "hello", "hi", "buenos días", "buenas tardes", "buenas noches"]:
            logging.info("Detectado saludo simple")
            respuesta, tiempo = generar_respuesta_y_analizar_sentimiento(query, tuple([]))
            return respuesta, tiempo

        logging.info("Recuperando documentos relevantes")
        documentos_relevantes = recuperar_documentos(query)
        if documentos_relevantes.empty:
            logging.warning("No se encontraron documentos relevantes para la consulta.")
            return "Lo siento, no pude encontrar información relevante para tu consulta.", 0

        documentos_relevantes_tuple = tuple(documentos_relevantes['text'].tolist())
        logging.info(f"Generando respuesta con {len(documentos_relevantes_tuple)} documentos relevantes")
        respuesta, tiempo = generar_respuesta_y_analizar_sentimiento(query, documentos_relevantes_tuple)
        return respuesta, tiempo
    except Exception as e:
        logging.error(f"Error en procesar_consulta: {str(e)}", exc_info=True)
        return f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}", 0
# Exportar funciones necesarias para app.py
__all__ = ['procesar_consulta']