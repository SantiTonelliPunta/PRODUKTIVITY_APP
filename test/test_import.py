# Guarda esto como test_import.py en el directorio raíz
try:
    from test import automatic_evaluation
    print("Importación exitosa")
except ImportError as e:
    print(f"Error de importación: {e}")