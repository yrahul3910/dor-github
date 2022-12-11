import json
import sys
import numpy as np
import seaborn as sns
from pprint import pprint
from matplotlib import pyplot as plt
from collections import Counter


with open('reuse.json', 'r') as f:
    reuse = json.load(f)

with open('works-cache.json', 'r') as f:
    works = json.load(f)


def get_years_range():
    """Gets the range of years of the works."""
    # First, get the reused DOIs
    reused_dois = map(lambda p: p['reusedDOI'], reuse)

    # Get their dates
    years = map(
        lambda p: list(map(
            # (2) ...and get its year
            lambda r: r['created']['date-parts'][0][0],
            # (1) Get the record for the work with the current DOI...
            filter(lambda q: q['DOI'] == p, works)
        )),
        reused_dois)

    years = list(map(
        lambda q: q[0],  # (2) ...and get the only element
        filter(lambda p: list(p) != [], years)  # (1) Remove empty lists...
    ))

    plt.figure(dpi=150, figsize=(8, 6))
    sns.set()
    plt.title('Distribution of years')
    plt.xlabel('Year created')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(min(years), max(years)+1, 2))
    sns.histplot(years, kde=True, stat='count', alpha=0.7)
    plt.savefig('years.svg')
    plt.show()


def get_number_nodes_by_type():
    """Gets the number of nodes of each type."""
    groups = [x['source'] for x in works]
    with open('arxiv-cache.json', 'r') as f:
        arxiv = json.load(f)

    mappings = Counter(groups)
    mappings['arxiv'] = len(arxiv)

    return mappings


def get_number_edges():
    """Gets the number of edges in the graph."""
    pruned_reuse = [x['sourceDOI'] + x['reusedDOI'] for x in reuse]
    return len(set(pruned_reuse))


def get_most_and_least_reused(first, last):
    """Gets the most and least reused works for an author."""
    dois = []
    reuse_count = []

    # What DOIs were authored by the input?
    for obj in reuse:
        doi = obj['reusedDOI']
        for x in works:
            if x['DOI'] == doi:
                authors = [(y['given'], y['family']) for y in x['author']]
                if (first, last) in authors:
                    dois.append(str(doi))

    # How many times were each of those reused?
    for doi in dois:
        count = 0
        for obj in reuse:
            if obj['reusedDOI'] == doi:
                count += 1

        reuse_count.append(count)

    result = sorted(zip(dois, reuse_count), key=lambda p: p[1], reverse=True)
    final = result[:1]
    final.extend(result[-1:])
    pprint(final)


if __name__ == '__main__':
    get_years_range()
