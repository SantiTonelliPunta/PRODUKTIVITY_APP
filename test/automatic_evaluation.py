from datetime import datetime
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics
from utils.data_processing import load_embeddings, generate_random_questions
from utils.visualization import plot_results_to_html
import os

# Crear carpeta 'test/results' si no existe
results_dir = 'test/results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Cargar embeddings
embeddings_df = load_embeddings('embeddings/1000_embeddings_store.csv')

# Generar preguntas aleatorias
questions = generate_random_questions(embeddings_df, num_questions=30)

# Inicializar lista de resultados
results = []

# Calcular métricas para cada pregunta
for question in questions:
    result = calculate_metrics(question, embeddings_df)
    results.append(result)

# Crear DataFrame de resultados
results_df = pd.DataFrame(results)

# Guardar resultados en un archivo CSV
csv_path = os.path.join(results_dir, 'results_evaluacion.csv')
results_df.to_csv(csv_path, index=False)

# Mostrar resultados en una tabla
print("Resultados:")
print(results_df)

# Generar visualizaciones en un archivo HTML
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
html_file_path = os.path.join(results_dir, f"autoanalisis_{datetime.now().strftime('%Y%m%d_%H%M')}.html")
plot_results_to_html(results_df, html_file_path, title=f"AUTOANALISIS {timestamp}")

print(f"Evaluación completada. Resultados guardados en '{csv_path}' y gráficos en '{html_file_path}'")