import json
import numpy as np

def page_rank(fname, a):
    with open(fname, 'r') as f:
        papers = json.load(f)

    g = []
    urls = []
    references = []
    for paper in papers:
        urls.append(paper['url'])
        references.append(paper['references'])
    n = len(urls)
    print(n)
    for i in range(n):
        links = np.zeros(n)
        refs = references[i]
        for ref in refs:
            for j in range(n):
                if ref == urls[j] and j != i:
                    links[j] = 1
        c = np.sum(links)
        if c:
            links = (1/c) * links
        g.append(links)
    page_rank = np.sum(g, axis=0) * a
    return page_rank

print(page_rank('papers_data.json', 0.25))