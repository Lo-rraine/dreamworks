import arxiv

client = arxiv.Client()


def collect_arxiv_papers(category, max_results):
    """
    Collect research papers from arXiv by category.

    :param category:
        str
        arXiv category code (e.g., 'cs.LG', 'cs.CV')

    :param max_results:
        int
        Maximum number of papers to retrieve

    :return:
        list of dict
        List of paper dictionaries containing title, abstract, authors, etc.
    """

    # Construct the search query for the category
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    papers = []
    for result in client.results(search):
        paper = {
            'title': result.title,
            'abstract': result.summary,
            'authors': [author.name for author in result.authors],
            'published': result.published,
            'category': result.primary_category,
            'arxiv_id': result.entry_id.split('/')[-1]
        }

        papers.append(paper)

    return papers

# Define the categories we want to collect from
categories = [
    ('cs.LG', 'Machine Learning'),
    ('cs.CV', 'Computer Vision'),
    ('cs.CL', 'Computational Linguistics'),
    ('cs.DB', 'Databases'),
    ('cs.SE', 'Software Engineering')
]

# Collect all 100 papers from each category
all_papers = []
for category_code, category_name in categories:
    print(f"Collecting papers from {category_name} ({category_code})...")

    papers = collect_arxiv_papers(category_code, 100)
    all_papers.extend(papers)
    print(f"\nTotal papers collected: {len(all_papers)}")

separator = "=" * 80
print(f"\n{separator}", "SAMPLE PAPERS (one from each category)", f"{separator}", sep="\n")
for i, (_, category_name) in enumerate(categories):
    paper = all_papers[i * 100]
    print(f"\n{category_name}:")
    print(f"  Title: {paper['title']}")
    print(f"  Abstract (first 150 chars): {paper['abstract'][:150]}...")