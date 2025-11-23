import numpy as np
import pandas as pd
import os

def load_embeddings_metadata():
    base_dir = os.path.dirname(os.path.dirname(__file__))  # go up one folder

    metadata_path = os.path.join(base_dir, "arxiv_papers_metadata.csv")
    embeddings_path = os.path.join(base_dir, "embeddings_cohere.npy")

    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} papers")

    embeddings = np.load(embeddings_path)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    print(f"Each paper is represented by a {embeddings.shape[1]}-dimensional vector")

    # Verify the data loaded correctly33666
    print(f"\nFirst paper title: {df['title'].iloc[0]}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")

    return df, embeddings


def euclidean_distance_manual(vec1, vec2):
    """
    Calculate Euclidean distance between two vectors.
    :param vec1 & vec2: numpy array - The vectors to compare
    :return: float - Euclidean distance (lower means more similar)
    """

    # np.linalg.norm computes the square root of sum of squared differences
    # This implements the Euclidean distance formula directly
    return np.linalg.norm(vec1-vec2)


def paper_comparisons():
    df, embeddings = load_embeddings_metadata()

    # assign position numbers (according to what they stored as in df)
    paper_idx_1 = 492
    paper_idx_2 = 493

    distance = euclidean_distance_manual(embeddings[paper_idx_1],
                                         embeddings[paper_idx_2])

    print("Comparing two papers:")
    print(f"Paper 1: {df['title'].iloc[paper_idx_1][:50]}...")
    print(f"Paper 1: {df['title'].iloc[paper_idx_2][:50]}...")
    print(f"Euclidean distance: {distance:.4f}")


if __name__ == "__main__":
    paper_comparisons()


