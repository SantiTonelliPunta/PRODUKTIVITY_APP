# app.py

from flask import Flask, render_template, request, jsonify
from utils.rag_system import recuperar_documentos, generar_respuesta_y_analizar_sentimiento
import os

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

    # Recuperar documentos relevantes
    documentos_relevantes = recuperar_documentos(message, corpus_df=corpus_df, top_n=5)
    
    # Generar respuesta y analizar sentimiento
    respuesta = generar_respuesta_y_analizar_sentimiento(message, documentos_relevantes)
    
    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
