# app.py
import os
import sys

# Añade el directorio raíz del proyecto al sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from flask import Flask, render_template, request, jsonify
from utils.rag_system import procesar_consulta
from dotenv import load_dotenv
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar variables de entorno desde .env sólo si no está en Replit
if not os.getenv('REPLIT_ENVIRONMENT'):
    load_dotenv()

# Verificar que la clave API se ha cargado correctamente
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    logging.error("API key not found in environment variables!")
else:
    logging.info("API key loaded successfully.")

# Configurar el entorno para los tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    
    if not message:
        logging.warning("Se recibió una solicitud sin mensaje.")
        return jsonify({"error": "No se proporcionó ningún mensaje."}), 400

    try:
        # Procesar la consulta
        respuesta, tiempo_total = procesar_consulta(message)
        logging.info(f"Consulta procesada. Tiempo total: {tiempo_total:.2f} segundos")
        
        return jsonify({"respuesta": respuesta, "tiempo_total": tiempo_total})
    except Exception as e:
        logging.error(f"Error al procesar la consulta: {str(e)}")
        return jsonify({"error": "Ocurrió un error al procesar la consulta."}), 500

@app.route("/history", methods=["GET"])
def get_history():
    try:
        with open('qa_history.csv', 'r', encoding='utf-8') as file:
            history = file.read()
        return jsonify({"history": history})
    except Exception as e:
        logging.error(f"Error al obtener el historial: {str(e)}")
        return jsonify({"error": "Ocurrió un error al obtener el historial."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Iniciando la aplicación en el puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=True)