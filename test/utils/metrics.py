import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

def cosine_similarity_score(embedding1, embedding2):
    score = cosine_similarity([embedding1], [embedding2])[0][0]
    return round(min(max(score * 10, 0), 10), 2)

def rouge_score(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return round(min(max(scores['rouge1'].fmeasure * 10, 0), 10), 2)

def exact_match_score(reference, generated_response):
    return 10.0 if reference.strip() == generated_response.strip() else 0.0

def tasa_cobertura_features(reference_features, generated_response):
    covered_features = len(set(reference_features).intersection(set(generated_response.split())))
    total_features = len(reference_features)
    if total_features == 0:
        return 0
    coverage_ratio = covered_features / total_features
    return round(min(max(coverage_ratio * 10, 0), 10), 2)

def perplexity_score(model, sentence):
    # Esta es una implementación simulada. En un caso real, usarías el modelo para calcular la perplejidad.
    perplexity = np.random.uniform(0, 100)  # Simulación
    return round(min(max((100 - perplexity) / 10, 0), 10), 2)

def calculate_metrics(question, embeddings_df):
    # Seleccionar una fila de embeddings aleatoriamente como referencia
    random_row = embeddings_df.sample(n=1).iloc[0]
    reference = random_row['cleaned_tokens']
    generated_response = random_row['lemmatized_tokens']
    embedding = np.fromstring(random_row['embeddings_str'][1:-1], sep=' ')
    
    # Comparar con otro embedding
    comparison_row = embeddings_df.sample(n=1).iloc[0]
    comparison_embedding = np.fromstring(comparison_row['embeddings_str'][1:-1], sep=' ')
    
    # Calcular las métricas
    cosine_sim = cosine_similarity_score(embedding, comparison_embedding)
    rouge = rouge_score(reference, generated_response)
    exact_match = exact_match_score(reference, generated_response)
    tasa_cobertura = tasa_cobertura_features(reference.split(), generated_response)
    perplexity = perplexity_score(None, generated_response)

    return {
        'Pregunta': question,
        'Cosine Similarity': cosine_sim,
        'ROUGE': rouge,
        'Exact Match': exact_match,
        'Tasa de Cobertura de Features': tasa_cobertura,
        'Perplexity': perplexity
    }
