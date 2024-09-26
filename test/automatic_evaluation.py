import os
import pandas as pd
from datetime import datetime
from .config import *
from .utils import (
    load_embeddings,
    generate_questions,
    simulate_responses,
    calculate_metrics,
    create_visualizations
)

def run_evaluation():
    """
    Función principal que ejecuta el proceso de evaluación completo.
    """
    print("Iniciando proceso de evaluación...")

    # Cargar embeddings
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"Embeddings cargados: {len(embeddings)}")

    # Generar preguntas
    questions = generate_questions(embeddings, NUM_QUESTIONS)
    print(f"Preguntas generadas: {len(questions)}")

    # Simular respuestas
    responses = simulate_responses(questions)
    print(f"Respuestas simuladas: {len(responses)}")

    # Calcular métricas
    results = calculate_metrics(questions, responses)
    print("Métricas calculadas")

    # Crear timestamp para los nombres de archivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Guardar resultados en CSV
    csv_path = os.path.join(CSV_OUTPUT_PATH, f"evaluation_results_{timestamp}.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")

    # Crear visualizaciones
    html_path = os.path.join(HTML_OUTPUT_PATH, f"evaluation_visuals_{timestamp}")
    html_file, image_files = create_visualizations(results, html_path)
    print(f"Visualizaciones guardadas en: {html_file}")
    print(f"Imágenes guardadas en: {', '.join(image_files)}")

    # Mostrar resultados por consola
    print("\nResumen de resultados:")
    for result in results:
        print(f"Pregunta: {result['question']}")
        print(f"Respuesta: {result['response']}")
        print(f"NDCG: {result['ndcg']:.4f}")
        print(f"Similaridad de coseno: {result['cosine_similarity']:.4f}")
        print(f"MRR: {result['mrr']:.4f}")
        print(f"ROUGE-L: {result['rouge_l']:.4f}")
        print("--------------------")

    print("Proceso de evaluación completado.")

if __name__ == "__main__":
    run_evaluation()
