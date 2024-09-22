import os
import csv
import math
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

# Variable global para contar las interacciones
interaction_counter = 0

# Función para calcular el MRR (Mean Reciprocal Rank)
def calculate_mrr(retrieved_documents, relevant_documents):
    for rank, doc in enumerate(retrieved_documents, 1):
        if doc in relevant_documents:
            return 1 / rank
    return 0

# Función para calcular la precisión
def calculate_precision(relevant_retrieved, total_retrieved):
    return relevant_retrieved / total_retrieved if total_retrieved > 0 else 0

# Función para calcular la similaridad de coseno entre la consulta y la respuesta generada
def calculate_cosine_similarity(query, generated_response):
    vectorizer = TfidfVectorizer().fit_transform([query, generated_response])
    vectors = vectorizer.toarray()
    cos_sim = cosine_similarity(vectors)
    return cos_sim[0][1]  # Devolvemos el valor de similitud entre query y respuesta

# Función para calcular ROUGE
def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

# Función para calcular BLEU
def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis)

# Función para verificar si existe la carpeta y escribir el archivo con encabezados
def create_folder_and_write_csv(interaction_number, timestamp, query, mrr, precision, cosine_similarity, rouge1, rougeL, bleu_score, response_time):
    # Nombre del archivo CSV
    filename = 'interaction_analysis.csv'
    
    # Ruta del directorio
    directory = 'data'
    
    # Verificar si la carpeta data existe, si no, crearla
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Definir el path completo para el archivo
    filepath = os.path.join(directory, filename)
    
    # Verificar si el archivo ya existe o no, para escribir los encabezados la primera vez
    file_exists = os.path.isfile(filepath)
    
    # Guardar los resultados en el archivo CSV
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Si el archivo no existe, escribir la fila de encabezados
        if not file_exists:
            writer.writerow(['Interaction Number', 'Timestamp', 'Query', 'MRR', 'Precision', 'Cosine Similarity', 'ROUGE-1', 'ROUGE-L', 'BLEU', 'Response Time (s)'])
        
        # Escribir la fila con los resultados
        writer.writerow([interaction_number, timestamp, query, mrr, precision, cosine_similarity, rouge1, rougeL, bleu_score, response_time])

# Función que agrupa todas las métricas y las guarda en CSV
def evaluate_and_save_metrics(query, generated_response, relevant_documents, response_time):
    global interaction_counter
    
    # Incrementar el contador de interacciones
    interaction_counter += 1
    
    # Obtener el timestamp actual
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # Calcular métricas
        mrr = calculate_mrr([generated_response], relevant_documents)
        precision = calculate_precision(1 if generated_response in relevant_documents else 0, 1)  # Si es relevante
        cosine_sim = calculate_cosine_similarity(query, generated_response)
        rouge1, rougeL = calculate_rouge(relevant_documents[0] if relevant_documents else "", generated_response)
        bleu_score = calculate_bleu(relevant_documents[0] if relevant_documents else "", generated_response)

        # Guardar las métricas en el archivo CSV dentro de la carpeta 'data'
        create_folder_and_write_csv(interaction_counter, timestamp, query, mrr, precision, cosine_sim, rouge1, rougeL, bleu_score, response_time)
    except Exception as e:
        print(f"Error al evaluar y guardar métricas: {str(e)}")
        # En caso de error, guardamos la interacción con valores nulos para las métricas
        create_folder_and_write_csv(interaction_counter, timestamp, query, None, None, None, None, None, None, response_time)

# Exportar funciones necesarias
__all__ = ['evaluate_and_save_metrics']