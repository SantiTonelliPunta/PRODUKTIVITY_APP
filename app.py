# app.py

from flask import Flask, render_template, request, jsonify
from utils.rag_system import procesar_consulta
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

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
    respuesta = procesar_consulta(message)
    
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)