"""
Option 3: Ranking with Cosine Similarity (L2)
This script takes tf-idf scores for 10 documents over 6 terms,
defines a query vector, and computes cosine similarity for ranking.
"""

import numpy as np
import pandas as pd


def main():
    # 10 documents, 6 terms, tf-idf values
    docs = [f"D{i}" for i in range(1, 11)]
    terms = ["t1", "t2", "t3", "t4", "t5", "t6"]

    tfidf_matrix = np.array([
        [0.3, 0.0, 0.1, 0.0, 0.5, 0.2],
        [0.0, 0.4, 0.0, 0.3, 0.1, 0.0],
        [0.2, 0.1, 0.3, 0.0, 0.0, 0.4],
        [0.5, 0.2, 0.0, 0.1, 0.0, 0.0],
        [0.1, 0.3, 0.2, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.4, 0.1, 0.3, 0.2],
        [0.2, 0.2, 0.0, 0.0, 0.4, 0.1],
        [0.3, 0.1, 0.1, 0.2, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.3, 0.2, 0.1],
        [0.4, 0.0, 0.2, 0.0, 0.1, 0.3],
    ], dtype=float)

    df_input = pd.DataFrame(tfidf_matrix, index=docs, columns=terms)
    print("Input: tf-idf scores (documents x terms)")
    print(df_input)

    # Query tf-idf vector (3 terms)
    query = {"t1": 0.5, "t3": 0.4, "t5": 0.7}
    q_vec = np.zeros(len(terms), dtype=float)
    for t, w in query.items():
        if t in terms:
            j = terms.index(t)
            q_vec[j] = w

    print("\nQuery tf-idf vector:")
    print(query)

    # Cosine similarity computation
    doc_norms = np.linalg.norm(tfidf_matrix, axis=1)
    q_norm = np.linalg.norm(q_vec)
    scores = []
    for i in range(len(docs)):
        dot = float(tfidf_matrix[i, :] @ q_vec)
        denom = (doc_norms[i] * q_norm) if doc_norms[i] > 0 and q_norm > 0 else 0.0
        score = dot / denom if denom > 0 else 0.0
        scores.append(score)

    df_scores = pd.DataFrame({
        "doc_id": docs,
        "cosine_score": scores
    })

    # Sort by score descending
    df_scores_sorted = df_scores.sort_values(by="cosine_score", ascending=False)
    print("\nCosine similarity scores per document (sorted)")
    print(df_scores_sorted.reset_index(drop=True))


if __name__ == "__main__":
    main()
