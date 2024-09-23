import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_results_to_html(results_df, file_path, title="AUTOANALISIS"):
    # Crear el HTML para visualización con dos filas de gráficos
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Métricas de Similitud", "Métricas de Cobertura"))

    # Graficar Cosine Similarity y ROUGE usando scatter
    fig.add_trace(go.Scatter(x=results_df['Pregunta'], y=results_df['Cosine Similarity'], mode='markers', name='Cosine Similarity'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df['Pregunta'], y=results_df['ROUGE'], mode='markers', name='ROUGE'), row=1, col=1)

    # Graficar Exact Match y Tasa de Cobertura de Features en la segunda fila
    fig.add_trace(go.Bar(x=results_df['Pregunta'], y=results_df['Exact Match'], name='Exact Match'), row=2, col=1)
    fig.add_trace(go.Bar(x=results_df['Pregunta'], y=results_df['Tasa de Cobertura de Features'], name='Tasa de Cobertura de Features'), row=2, col=1)

    # Actualizar diseño
    fig.update_layout(height=800, width=1000, title_text=title)
    fig.update_xaxes(tickangle=45, tickfont=dict(size=8))

    # Guardar en archivo HTML
    fig.write_html(file_path)

    print(f"Gráficos guardados en: {file_path}")
