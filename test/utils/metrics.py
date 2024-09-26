import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge

def calculate_ndcg(relevance_scores, k=10):
    """Calcula el NDCG."""
    dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(sorted(relevance_scores, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0

def calculate_mrr(relevance_binary):
    """Calcula el MRR."""
    for i, rel in enumerate(relevance_binary):
        if rel == 1:
            return 1 / (i + 1)
    return 0

def extract_embedding(embedding_dict):
    """Extrae el vector de embedding del diccionario."""
    if 'embeddings_str' in embedding_dict:
        return np.array([float(x) for x in embedding_dict['embeddings_str'].strip('[]').split(',')])
    elif 'embedding' in embedding_dict:
        return np.array(embedding_dict['embedding'])
    else:
        raise ValueError("Formato de embedding no reconocido")

def calculate_metrics(questions, responses):
    """
    Calcula las métricas NDCG, similaridad de coseno, MRR y ROUGE-L.
    """
    rouge = Rouge()
    results = []
    
    for q, r in zip(questions, responses):
        # Extraer los embeddings
        q_embedding = extract_embedding(q['embedding'])
        r_embedding = extract_embedding(r['embedding'])
        
        # Simular relevancia y calcular NDCG y MRR
        relevance_scores = np.random.rand(10)  # Simulación de scores de relevancia
        ndcg = calculate_ndcg(relevance_scores)
        mrr = calculate_mrr([1 if score > 0.5 else 0 for score in relevance_scores])
        
        # Calcular similaridad de coseno
        cos_sim = cosine_similarity([q_embedding], [r_embedding])[0][0]
        
        # Calcular ROUGE-L
        rouge_scores = rouge.get_scores(r['response'], q['question'])
        rouge_l = rouge_scores[0]['rouge-l']['f']
        
        results.append({
            "question": q['question'],
            "response": r['response'],
            "ndcg": ndcg,
            "cosine_similarity": cos_sim,
            "mrr": mrr,
            "rouge_l": rouge_l
        })
    
    return results
