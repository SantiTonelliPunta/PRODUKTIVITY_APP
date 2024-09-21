# main.py

from flask import Flask, render_template, request, jsonify
from utils.rag_system import procesar_consulta
from dotenv import load_dotenv
import os
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
        return jsonify({"error": "No se proporcionó ningún mensaje."}), 400

    # Procesar la consulta
    respuesta, tiempo_total = procesar_consulta(message)
    logging.info(f"Respuesta generada: {respuesta}")
    
    return jsonify({"respuesta": respuesta, "tiempo_total": tiempo_total})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)