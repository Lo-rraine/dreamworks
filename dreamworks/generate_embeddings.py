from clients import cohere_api
from preprocess_arxiv_data import clean_papers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import numpy as np


def generate_embedding():
    df = clean_papers()
    abstracts = df['abstract'].tolist()

    print("Generate embeddings using Cohere API...")
    print(f"Processing {len(abstracts)} abstracts...")

    start_time = time.time()
    actual_api_time = 0  # Track time spent on actual API calls

    # Cohere recommends processing in batches for efficiency
    # Their API accepts up to 96 texts per request
    batch_size = 90
    all_embeddings = []

    co = cohere_api()

    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(abstracts) + batch_size - 1) // batch_size
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} abstracts...")

        # Add retry logic for rate limits
        max_retries = 3
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                api_start = time.time()

                # Generate embeddings for this batch using V2 API
                response = co.embed(
                    texts=batch,
                    model='embed-v4.0',
                    input_type='search_document',
                    embedding_types=['float']
                )

                actual_api_time += time.time() - api_start
                # V2 API returns embeddings in a different structure
                all_embeddings.extend(response.embeddings.float_)
                break  # Success, move to next batch

            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    print(f"  Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    raise  # Re-raise if it's not a rate limit error or we're out of retries

    # Add a delay between batches to avoid hitting rate limits
    # Wait 12 seconds between batches (spreads 500 papers over ~1 minute)
    if i + batch_size < len(abstracts):  # Don't wait after the last batch
        time.sleep(12)

    # Convert to numpy array for consistency with local models
    embeddings_cohere = np.array(all_embeddings)
    elapsed_time = time.time() - start_time

    print(f"\nCompleted in {elapsed_time:.2f} seconds (includes rate limit delays)")
    print(f"Actual API processing time: {actual_api_time:.2f} seconds")
    print(f"Time spent waiting for rate limits: {elapsed_time - actual_api_time:.2f} seconds")
    print(f"Embedding shape: {embeddings_cohere.shape}")
    print(f"Each abstract is now a {embeddings_cohere.shape[1]}-dimensional vector")
    print(f"Average time per abstract (API only): {actual_api_time / len(abstracts):.3f} seconds")

    # Add to DataFrame
    df['embedding_cohere'] = list(embeddings_cohere)
    return df, embeddings_cohere


def visualise_embeddings():
    # Reduce embeddings so it's easier to visualise
    df, embeddings_cohere = generate_embedding()
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_cohere)

    print(f"Original embedding dimensions: {embeddings_cohere.shape[1]}")
    print(f"Reduced embedding dimensions: {embeddings_2d.shape[1]}")

    plt.figure(figsize=(12, 8))

    # Define colors for our visualisation
    colours = ['#C8102E', '#003DA5', '#00843D', '#FF8200', '#6A1B9A']
    category_names = ['Machine Learning', 'Computer Vision', 'Comp. Linguistics', 'Databases', 'Software Eng.']
    category_codes = ['cs.LG', 'cs.CV', 'cs.CL', 'cs.DB', 'cs.SE']

    # Plot each category
    for i, (cat_code, cat_name, colour) in enumerate(zip(category_codes, category_names, colours)):
        # Get papers from this category
        mask = df['category'] == cat_code
        cat_embeddings = embeddings_2d[mask]

        plt.scatter(cat_embeddings[:, 0], cat_embeddings[:, 1], c=colour, label=cat_name,
                    s=50, alpha=0.6, edgecolors="black", linewidths=0.5)

    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Second Principal Component', fontsize=12)
    plt.title('500 arXiv Papers Across Five Computer Science CAtegories\n '
              'Embeddings showing overlapping cluster', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_embeddings_to_csv():
    df, embeddings_cohere = generate_embedding()
    df_metadata = df[['title', 'abstract', 'authors', 'published', 'category', 'arxiv_id', 'abstract_length']]
    df_metadata.to_csv('arxiv_papers_metadata.csv', index=False)
    print("Saved metadata to 'arvix_papers.csv")

    # Save embeddings as numpy arrays
    np.save('embeddings_cohere.npy', embeddings_cohere)



if __name__ == "__main__":
    visualise_embeddings()
