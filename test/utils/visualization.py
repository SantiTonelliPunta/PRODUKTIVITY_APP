import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from ..config import PLOT_WIDTH, PLOT_HEIGHT

def create_visualizations(results, output_path):
    """
    Crea visualizaciones para las métricas calculadas y las guarda en archivos PNG y un HTML.

    Args:
    results (list): Lista de diccionarios con los resultados de las métricas.
    output_path (str): Ruta base donde se guardarán los archivos.
    """
    df = pd.DataFrame(results)
    metrics = ['ndcg', 'cosine_similarity', 'mrr', 'rouge_l']

    # Crear directorio para las imágenes si no existe
    img_dir = os.path.join(os.path.dirname(output_path), 'images')
    os.makedirs(img_dir, exist_ok=True)

    # Crear una figura con subplots para cada métrica
    fig, axes = plt.subplots(2, 2, figsize=(PLOT_WIDTH*2, PLOT_HEIGHT*2))
    fig.suptitle('Distribución de Métricas', fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        sns.histplot(df[metric], kde=True, ax=ax)
        ax.set_title(f'Distribución de {metric.upper()}')
        ax.set_xlabel(metric.upper())
        ax.set_ylabel('Frecuencia')

    # Ajustar el layout y guardar la figura
    plt.tight_layout()
    metrics_plot_path = os.path.join(img_dir, 'metrics_distribution.png')
    plt.savefig(metrics_plot_path)
    plt.close()

    # Crear un gráfico de correlación entre métricas
    plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    sns.heatmap(df[metrics].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlación entre Métricas')
    correlation_plot_path = os.path.join(img_dir, 'metrics_correlation.png')
    plt.savefig(correlation_plot_path)
    plt.close()

    # Crear archivo HTML con las imágenes
    html_content = f"""
    <html>
    <head><title>Visualizaciones de Métricas</title></head>
    <body>
        <h1>Distribución de Métricas</h1>
        <img src="images/metrics_distribution.png" alt="Distribución de Métricas">
        <h1>Correlación entre Métricas</h1>
        <img src="images/metrics_correlation.png" alt="Correlación entre Métricas">
    </body>
    </html>
    """

    html_file_path = os.path.splitext(output_path)[0] + '.html'
    with open(html_file_path, 'w') as f:
        f.write(html_content)

    print(f"Visualizaciones guardadas en: {html_file_path}")
    print(f"Imágenes guardadas en: {img_dir}")

    return html_file_path, [metrics_plot_path, correlation_plot_path]
