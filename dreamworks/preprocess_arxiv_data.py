import pandas as pd
from collect_arxiv_papers import collect_all_paper_category


def clean_papers():
    all_papers = collect_all_paper_category()

    # Convert to Dataframe for easier data manipulation
    df = pd.DataFrame(all_papers)

    print("Dataset before cleaning:")
    print(f"Total papers: {len(df)}")
    print(f"Papers with abstract: {df['abstract'].notna().sum()}")

    # Check for missing abstracts
    missing_abstracts = df['abstract'].isna().sum()
    if missing_abstracts > 0:
        print(f"\nWarning: {missing_abstracts} papers have missing abstracts")
        df = df.dropna(subset=['abstract'])

    # Filter out papers with short abstracts (less than 100 characters)
    # These are often placeholders or incomplete entries
    df['abstract_length'] = df['abstract'].str.len()
    df = df[df['abstract_length'] >= 100].copy()

    print(f"\nDataset after cleaning: ")
    print(f"Total papers: {len(df)}")
    print(f"Average abstract length: {df['abstract_length'].mean():.0f} characters")

    # Show the distribution across categories
    print(f"\nPapers per category: ")
    print(df['category'].value_counts().sort_index())

    separator = "=" * 80
    print(f"\n{separator}", "FIRST 3 PAPERS IN CLEANED DATASET", f"{separator}", sep="\n")
    for idx, row in df.head(3).iterrows():
        print(f"\n{idx + 1}. {row['title']}")
        print(f"    Category: {row['category']}")
        print(f"    Abstract length: {row['abstract_length']} characters")

    return df


clean_papers()
