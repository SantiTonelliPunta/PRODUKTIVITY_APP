import os
import ast
import time
import logging
import aiohttp
import asyncio
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score, recall_score, ndcg_score
from sklearn.metrics.pairwise import cosine_similarity
from utils.evaluation_metrics import evaluate_and_save_metrics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar el modelo SBERT solo una vez
start_time = time.time()
model = SentenceTransformer('all-mpnet-base-v2')
logging.info(
    f"Modelo SBERT cargado en {time.time() - start_time:.2f} segundos")

# Configurar la clave de API de OpenAI
api_key = os.getenv('OPENAI_API_KEY')

# Ruta al archivo CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
datafile_path = os.path.join(base_dir, '..', 'embeddings',
                             '1000_embeddings_store.csv')

def str_to_array(s):
    try:
        return np.array(ast.literal_eval(s))
    except:
        return np.array([])

corpus_df = None

def load_data():
    global corpus_df
    if corpus_df is None:
        start_time = time.time()
        corpus_df = pd.read_csv(datafile_path)
        corpus_df['embedding'] = corpus_df['embeddings_str'].apply(
            str_to_array)
        logging.info(
            f"Cargados {len(corpus_df)} documentos con embeddings en {time.time() - start_time:.2f} segundos."
        )

load_data()

@lru_cache(maxsize=100)
def obtener_embedding(texto):
    return tuple(model.encode([texto])[0])

def recuperar_documentos(query, top_n=5):
    start_time = time.time()
    query_embedding = obtener_embedding(query)

    corpus_embeddings = np.vstack(corpus_df['embedding'].values)
    query_embedding = normalize([query_embedding])[0]
    corpus_embeddings = normalize(corpus_embeddings)

    similitudes = cosine_similarity([query_embedding], corpus_embeddings)[0]
    corpus_df['similaridad'] = similitudes
    documentos_recuperados = corpus_df.sort_values(by='similaridad',
                                                   ascending=False).head(top_n)

    logging.info(
        f"Recuperación de documentos completada en {time.time() - start_time:.2f} segundos."
    )
    return documentos_recuperados, similitudes

def calcular_precision(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='binary')
    logging.info(f"Precisión calculada: {precision:.4f}")
    return precision

def calcular_ndcg(y_true, y_score):
    ndcg = ndcg_score([y_true], [y_score])
    logging.info(f"NDCG calculado: {ndcg:.4f}")
    return ndcg

def calcular_recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='binary')
    logging.info(f"Recall calculado: {recall:.4f}")
    return recall

def calcular_cosine_similarity(embedding1, embedding2):
    cosine_sim = cosine_similarity([embedding1], [embedding2])[0][0]
    logging.info(f"Cosine Similarity calculado: {cosine_sim:.4f}")
    return cosine_sim

def evaluar_query(query, ground_truth):
    logging.info(f"Evaluando query: {query} con ground_truth: {ground_truth}")

    documentos_recuperados, similitudes = recuperar_documentos(query)
    logging.info(f"Documentos recuperados: {documentos_recuperados}")

    ground_truth_embeddings = [
        obtener_embedding(text) for text in ground_truth
    ]
    y_true = [1 if text in ground_truth else 0 for text in corpus_df['text']]
    y_pred = [
        1 if df.text in documentos_recuperados['text'].values else 0
        for df in corpus_df.itertuples()
    ]

    logging.info(f"y_true: {y_true}")
    logging.info(f"y_pred: {y_pred}")

    precision = calcular_precision(y_true, y_pred)
    ndcg = calcular_ndcg(y_true, similitudes)
    recall = calcular_recall(y_true, y_pred)
    coherence = np.mean([
        calcular_cosine_similarity(obtener_embedding(query), gt_emb)
        for gt_emb in ground_truth_embeddings
    ])

    # Loggear todas las métricas
    logging.info(
        f"Evaluación de Query - Precisión: {precision:.4f}, NDCG: {ndcg:.4f}, Recall: {recall:.4f}, Coherencia: {coherence:.4f}"
    )

    return {
        "precision": precision,
        "ndcg": ndcg,
        "recall": recall,
        "coherence": coherence
    }

async def generar_respuesta_y_analizar_sentimiento(query, documentos_relevantes_tuple):
    start_time = time.time()

    documentos_relevantes = list(documentos_relevantes_tuple)
    contexto = "\n".join(documentos_relevantes)

    saludos = [
        "hola", "hello", "hi", "buenos días", "buenas tardes", "buenas noches"
    ]
    if query.lower().strip() in saludos:
        respuesta = "Hola, ¿en qué puedo ayudarte hoy con respecto a la búsqueda y análisis de productos?"
        total_duration = time.time() - start_time
        evaluate_and_save_metrics(query, respuesta, documentos_relevantes, total_duration)
        return format_response(respuesta, total_duration), total_duration

    prompt = f"""
    Eres un asistente virtual para el proyecto de análisis de reseñas de productos en Amazon. Tu objetivo es transformar el análisis de reseñas en una herramienta estratégica para el desarrollo de productos y la inteligencia de mercado. Proporcionas información precisa y relevante basada en las reseñas de productos, beneficiando a empresas B2B en España, como fabricantes, vendedores en Amazon, agencias de marketing digital, plataformas de e-commerce e inversores.

    Debes utilizar las reseñas de productos de Amazon disponibles en el corpus, así como la base de datos de embeddings ubicada en `/content/drive/My Drive/001_AI_MASTER/TFM_grupo2_Master_Inesdi/9. Notebooks/data_sets_MVP/Embeddings/1000_embeddings_store.csv` para obtener información adicional.

    Eres capaz de responder en diferentes idiomas y, si no conoces el idioma del usuario, debes traducir la entrada y la salida al idioma que te han preguntado.

    Personalización de la comunicación:
    - Utiliza un tono profesional y experto.
    - Sé directo y claro; si alguien te solicita información, provee detalles específicos.
    - Usa el corpus de embeddings para buscar información adicional cuando sea necesario.

    Tono de Voz y Trato:
    - Debe ser formal, pero cordial.
    - Sus respuestas deben ser agradables, fluidas, cordiales y con la intención de ayudar.
    - Debe responder en el idioma que hable el usuario.

    Restricciones y Limitaciones del Chatbot:
    - No puedes citar las fuentes que no sean del corpus de reseñas de Amazon y la base de datos de embeddings especificada.
    - No debes mencionar ni promocionar productos o servicios específicos que no estén basados en las reseñas proporcionadas.
    - No debes ser irrespetuoso, violento, sexista, y no debes tener comentarios impropios o que hieran los sentimientos de los usuarios.
    - No enumeres tus respuestas.

    Basándote en la siguiente información:

    {contexto}

    Responde a la siguiente pregunta: {query}

    Después de tu respuesta, en un nuevo párrafo, analiza el sentimiento general de las reseñas proporcionadas (no de la pregunta del usuario) y proporciona un resumen. Indica si las reseñas tienen un sentimiento positivo, negativo o neutral, y proporciona una puntuación de polaridad para reflejar la intensidad del sentimiento de las reseñas.
    """

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    json_data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{
            "role": "system",
            "content": "Eres un experto en análisis de reseñas de Amazon y en proporcionar información precisa y relevante sobre productos y mercado."
        }, {
            "role": "user",
            "content": prompt
        }],
    }

    try:
        respuesta_start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=json_data) as resp:
                respuesta = await resp.json()
        api_duration = time.time() - respuesta_start_time
        
        # Verificar si la respuesta tiene la estructura esperada
        if 'choices' in respuesta and len(respuesta['choices']) > 0 and 'message' in respuesta['choices'][0]:
            respuesta_texto = respuesta['choices'][0]['message']['content']
        else:
            raise ValueError("La respuesta de la API no tiene la estructura esperada")
        
        total_duration = time.time() - start_time
        logging.info(
            f"Tiempo de la llamada a la API de OpenAI: {api_duration:.2f} segundos"
        )
        
        # Evaluamos y guardamos las métricas, incluyendo el tiempo de respuesta
        evaluate_and_save_metrics(query, respuesta_texto, documentos_relevantes, total_duration)
        
        return format_response(respuesta_texto, total_duration), total_duration
    except Exception as e:
        logging.error(f"Error al generar la respuesta: {str(e)}")
        error_message = f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}"
        total_duration = time.time() - start_time
        evaluate_and_save_metrics(query, error_message, documentos_relevantes, total_duration)
        return format_response(error_message, total_duration), total_duration

def format_response(response_text, duration):
    # Formatear la respuesta sin enumeración
    formatted_response = f"""<div style="margin-bottom: 20px;">
    {response_text.strip()}
</div>"""
    return formatted_response

async def procesar_consulta_async(query):
    logging.info(f"Iniciando procesamiento de consulta: {query}")
    try:
        if query.lower().strip() in [
                "hola", "hello", "hi", "buenos días", "buenas tardes",
                "buenas noches"
        ]:
            logging.info("Detectado saludo simple")
            respuesta, tiempo = await generar_respuesta_y_analizar_sentimiento(
                query, tuple([]))
            return respuesta, tiempo

        logging.info("Recuperando documentos relevantes")
        start = time.time()
        documentos_relevantes, _ = recuperar_documentos(query)
        doc_retrieve_time = time.time() - start
        logging.info(
            f"Tiempo en recuperar documentos: {doc_retrieve_time:.2f} segundos"
        )

        if documentos_relevantes.empty:
            logging.warning(
                "No se encontraron documentos relevantes para la consulta.")
            return "Lo siento, no pude encontrar información relevante para tu consulta.", 0

        documentos_relevantes_tuple = tuple(
            documentos_relevantes['text'].tolist())
        logging.info(
            f"Generando respuesta con {len(documentos_relevantes_tuple)} documentos relevantes"
        )
        start = time.time()
        respuesta, tiempo = await generar_respuesta_y_analizar_sentimiento(
            query, documentos_relevantes_tuple)
        total_time = time.time() - start
        logging.info(
            f"Tiempo total para generar respuesta: {total_time:.2f} segundos")
        return respuesta, tiempo
    except Exception as e:
        logging.error(f"Error en procesar_consulta: {str(e)}", exc_info=True)
        return f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}", 0

def procesar_consulta(query):
    return asyncio.run(procesar_consulta_async(query))

# Exportar funciones necesarias para main.py
__all__ = ['procesar_consulta', 'evaluar_query']

# Código de prueba
if __name__ == "__main__":
    print("Probando rag_system.py")
    query = "¿Cuál es el mejor producto?"
    resultado, tiempo = procesar_consulta(query)
    print(f"Consulta: {query}")
    print(f"Respuesta: {resultado}")
    print(f"Tiempo de procesamiento: {tiempo} segundos")