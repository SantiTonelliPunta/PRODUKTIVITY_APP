
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_results_to_html(results_df, file_path, title="AUTOANALISIS"):
    # Crear el HTML para visualización con dos filas de gráficos
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Cosine Similarity", "ROUGE y Tasa de Cobertura"))

    # Graficar Cosine Similarity usando scatter.Line
    fig.add_trace(go.Scatter(x=results_df['Pregunta'], y=results_df['Cosine Similarity'], mode='markers', name='Cosine Similarity'), row=1, col=1)

    # Graficar ROUGE y Tasa de Cobertura de Features en la segunda fila
    fig.add_trace(go.Bar(x=results_df['Pregunta'], y=results_df['ROUGE'], name='ROUGE'), row=2, col=1)
    fig.add_trace(go.Scatter(x=results_df['Pregunta'], y=results_df['Tasa de Cobertura de Features'], name='Tasa de Cobertura de Features'), row=2, col=1)

    # Actualizar diseño
    fig.update_layout(height=600, width=800, title_text=title)

    # Guardar en archivo HTML
    fig.write_html(file_path)

    print(f"Gráficos guardados en: {file_path}")
