import pandas as pd

def load_embeddings(file_path):
    """
    Carga los embeddings desde un archivo CSV.

    Args:
    file_path (str): Ruta al archivo CSV que contiene los embeddings.

    Returns:
    list: Lista de diccionarios, cada uno representando un embedding.
    """
    try:
        df = pd.read_csv(file_path)
        # Convertir el DataFrame a una lista de diccionarios
        embeddings = df.to_dict('records')
        return embeddings
    except FileNotFoundError:
        print(f"Error: El archivo {file_path} no fue encontrado.")
        return []
    except pd.errors.EmptyDataError:
        print(f"Error: El archivo {file_path} está vacío.")
        return []
    except Exception as e:
        print(f"Error al cargar los embeddings: {str(e)}")
        return []

def preprocess_text(text):
    """
    Realiza el preprocesamiento del texto si es necesario.

    Args:
    text (str): Texto a preprocesar.

    Returns:
    str: Texto preprocesado.
    """
    # Por ahora, solo realizamos una limpieza básica
    return text.strip().lower()
