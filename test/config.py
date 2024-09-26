import os

# Número de preguntas a generar
NUM_QUESTIONS = 30

# Rutas de archivos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings', '1000_embeddings_store.csv')
CSV_OUTPUT_PATH = os.path.join(BASE_DIR, 'test', 'data', 'csv')
HTML_OUTPUT_PATH = os.path.join(BASE_DIR, 'test', 'data', 'html')

# Asegurarse de que los directorios de salida existan
os.makedirs(CSV_OUTPUT_PATH, exist_ok=True)
os.makedirs(HTML_OUTPUT_PATH, exist_ok=True)

# Configuración de GPT-3.5 Turbo
GPT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 150

# Configuración de visualización
PLOT_WIDTH = 10
PLOT_HEIGHT = 6
