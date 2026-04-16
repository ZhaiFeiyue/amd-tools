---
name: paper-search
description: >-
  Search arXiv for AI infrastructure papers by keyword, author, category, or
  date range. Supports advanced queries, result filtering, and batch reading.
  Use when the user wants to search for papers, find recent papers on a topic,
  look up an author's work, or browse new arXiv submissions. Also triggers on
  "搜paper", "搜论文", "搜一下", "search paper", "find paper", "最近有什么好论文".
---

# arXiv Paper Search

Search arXiv and present results. When the user picks papers from results,
hand off to the `paper-reader` skill for full analysis.

---

## Query Building

Translate the user's request into an arXiv API query. Combine prefixes with
boolean operators (`AND`, `OR`, `ANDNOT`).

| Prefix | Meaning | Example |
|--------|---------|---------|
| `ti:` | Title | `ti:FlashAttention` |
| `au:` | Author | `au:Tri Dao` |
| `abs:` | Abstract | `abs:speculative decoding` |
| `cat:` | Category | `cat:cs.LG` |
| `all:` | All fields | `all:KV cache compression` |

**Common AI infra categories**: `cs.LG`, `cs.CL`, `cs.AI`, `cs.DC`, `cs.AR`,
`cs.PF`, `cs.SE`

### Query Examples

| User says | Query |
|-----------|-------|
| "搜 FlashAttention" | `all:FlashAttention` |
| "Tri Dao 的论文" | `au:Tri Dao` |
| "最近的 MoE 推理优化" | `all:MoE AND all:inference AND all:optimization` |
| "cs.AR 上的 GPU 内存论文" | `cat:cs.AR AND all:GPU memory` |
| "量化相关，排除 NLP" | `all:quantization ANDNOT cat:cs.CL` |

---

## Search Execution

```bash
# Default: 15 results, sorted by submission date (newest first)
curl -s "https://export.arxiv.org/api/query?search_query=QUERY&start=0&max_results=15&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
total = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults')
if total is not None:
    print(f'Total results: {total.text}')
    print()
for i, entry in enumerate(root.findall('a:entry', ns)):
    eid = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
    title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
    authors = ', '.join(a.find('a:name', ns).text for a in entry.findall('a:author', ns))
    published = entry.find('a:published', ns).text[:10]
    updated = entry.find('a:updated', ns).text[:10]
    cats = ', '.join(c.get('term') for c in entry.findall('a:category', ns))
    abstract = entry.find('a:summary', ns).text.strip().replace('\n', ' ')[:200]
    print(f'{i+1}. [{eid}] {title}')
    print(f'   Authors: {authors}')
    print(f'   Published: {published} | Updated: {updated}')
    print(f'   Categories: {cats}')
    print(f'   {abstract}...')
    print()
"
```

### Sort Options

| User wants | `sortBy` | `sortOrder` |
|------------|----------|-------------|
| Newest first (default) | `submittedDate` | `descending` |
| Oldest first | `submittedDate` | `ascending` |
| Most relevant | `relevance` | `descending` |
| Recently updated | `lastUpdatedDate` | `descending` |

### Pagination

Change `start=0` to `start=15` for page 2, `start=30` for page 3, etc.
When the user says "more" or "next page", increment by `max_results`.

---

## Result Presentation

Present results as a numbered list in this format:

```markdown
### arXiv Search: "{user's query}"
> {total} results found | Showing {start+1}-{start+count} | Sorted by: {sort}

| # | ID | Title | Date | Categories |
|---|-----|-------|------|------------|
| 1 | 2401.xxxxx | Title here | 2024-01 | cs.LG, cs.CL |
| 2 | ... | ... | ... | ... |

**Pick papers to read**: give me the numbers (e.g. "1, 3, 5") or say
"read all" to deep-read them via paper-reader.
```

For each paper, also show a 1-2 sentence abstract summary below the table
if the user asks for details.

---

## After Selection

When the user picks papers:

1. For each selected paper, invoke the **paper-reader** skill pipeline
   by reading `~/.cursor/skills/paper-reader/SKILL.md` and running the
   full 11-phase process with the arXiv ID.
2. If multiple papers are selected, process them sequentially.
3. After all papers are read, report a summary of what was added to the
   knowledge base.

---

## Special Modes

### Trending / Recent

When the user asks "最近有什么好论文" or "trending papers":

Search recent papers in key AI infra categories:

```bash
curl -s "https://export.arxiv.org/api/query?search_query=cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.DC+OR+cat:cs.AR&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending" | python3 -c "
import sys, xml.etree.ElementTree as ET
ns = {'a': 'http://www.w3.org/2005/Atom'}
root = ET.parse(sys.stdin).getroot()
for i, entry in enumerate(root.findall('a:entry', ns)):
    eid = entry.find('a:id', ns).text.strip().split('/abs/')[-1]
    title = entry.find('a:title', ns).text.strip().replace('\n', ' ')
    published = entry.find('a:published', ns).text[:10]
    cats = ', '.join(c.get('term') for c in entry.findall('a:category', ns))
    print(f'{i+1}. [{eid}] {title}')
    print(f'   {published} | {cats}')
    print()
"
```

### Author Profile

When the user asks about a specific author's work, search by author and
group results by year:

Query: `au:"Author Name"`  
Sort: `submittedDate`, descending  
Max results: 30

### Check Knowledge Base

Before presenting results, cross-reference with `~/.cursor/paper-db/papers.json`
to mark papers already in the database:

```python
import json, os
DB = os.path.expanduser("~/.cursor/paper-db/papers.json")
with open(DB) as f:
    known = {p["id"] for p in json.load(f)["papers"]}
```

Mark known papers with `[已读]` in the results list.

### Semantic Search (Hugging Face)

If arXiv API results are unsatisfactory, try Semantic Scholar as fallback:

```bash
curl -s "https://api.semanticscholar.org/graph/v1/paper/search?query=QUERY&limit=10&fields=title,authors,year,externalIds,abstract,citationCount" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, p in enumerate(data.get('data', [])):
    arxiv = p.get('externalIds', {}).get('ArXiv', 'N/A')
    title = p['title']
    authors = ', '.join(a['name'] for a in (p.get('authors') or [])[:3])
    year = p.get('year', '?')
    cites = p.get('citationCount', 0)
    print(f'{i+1}. [{arxiv}] {title}')
    print(f'   {authors} | {year} | Citations: {cites}')
    abstract = (p.get('abstract') or '')[:200]
    if abstract:
        print(f'   {abstract}...')
    print()
"
```
