#!/usr/bin/env python3
"""Generate self-contained HTML pages from the paper knowledge base.

Reads ~/.cursor/paper-db/papers.json and notes/*.md.
Outputs overview.html and per-category HTML files.
All HTML is self-contained — inline CSS/JS, no external dependencies.

Usage: python3 generate_html.py
"""

import json, os, re, math, random, base64
import html as html_mod
from pathlib import Path

DB_DIR = Path.home() / ".cursor" / "paper-db"
NOTES_DIR = DB_DIR / "notes"
IMAGES_DIR = DB_DIR / "images"

CATEGORIES = {
    "algorithm": ("Algorithm",  "#16a34a", "Training algorithms, optimization, RL"),
    "kernel":    ("Kernel",     "#ea580c", "GPU kernels, hardware optimization"),
    "framework": ("Framework",  "#2563eb", "Serving & training systems"),
    "llm":       ("LLM",       "#9333ea", "Model architecture, scaling, serving"),
    "agent":     ("Agent",      "#dc2626", "Agentic systems, tool use, planning"),
}

def cat_info(cat):
    if cat in CATEGORIES:
        return CATEGORIES[cat]
    colors = ["#0891b2", "#65a30d", "#c026d3", "#0284c7", "#d97706"]
    return (cat.title(), colors[hash(cat) % len(colors)], "")


# ── Markdown → HTML ──────────────────────────────────────────────────

def inline(text):
    """Inline markdown: bold, italic, code, links, images."""
    # Images
    def img_replace(m):
        alt, src = m.group(1), m.group(2)
        src = src.replace("../images/", "images/")
        # Embed local images as base64 if they exist
        local = DB_DIR / src
        if local.exists() and local.stat().st_size < 2_000_000:
            suffix = local.suffix.lstrip(".").lower()
            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "gif": "image/gif", "webp": "image/webp", "svg": "image/svg+xml"
                    }.get(suffix, "image/png")
            b64 = base64.b64encode(local.read_bytes()).decode()
            return f'<img src="data:{mime};base64,{b64}" alt="{html_mod.escape(alt)}" style="max-width:100%;border-radius:8px;margin:8px 0;border:1px solid #e2e8f0">'
        return f'<img src="{html_mod.escape(src)}" alt="{html_mod.escape(alt)}" style="max-width:100%;border-radius:8px;margin:8px 0;border:1px solid #e2e8f0">'

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replace, text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`([^`]+)`', lambda m: f'<code style="background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:0.88em">{html_mod.escape(m.group(1))}</code>', text)
    return text


def md_to_html(text):
    """Convert markdown to HTML."""
    if not text or not text.strip():
        return ""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Code block
        if line.strip().startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(html_mod.escape(lines[i]))
                i += 1
            out.append('<pre style="background:#1e293b;color:#e2e8f0;padding:14px;border-radius:8px;'
                       'overflow-x:auto;font-size:0.85rem;margin:10px 0;font-family:\'SF Mono\','
                       'Consolas,monospace"><code>' + '\n'.join(code) + '</code></pre>')
            if i < len(lines):
                i += 1
            continue

        # HR
        if re.match(r'^-{3,}\s*$', line):
            out.append('<hr style="border:none;border-top:1px solid #e2e8f0;margin:20px 0">')
            i += 1
            continue

        # Headers
        m = re.match(r'^(#{1,6})\s+(.+)$', line)
        if m:
            lvl = len(m.group(1))
            sizes = {1: '1.4rem', 2: '1.25rem', 3: '1.1rem', 4: '1rem', 5: '0.95rem', 6: '0.9rem'}
            out.append(f'<h{lvl} style="font-size:{sizes[lvl]};margin:18px 0 8px;font-weight:700">'
                       f'{inline(m.group(2))}</h{lvl}>')
            i += 1
            continue

        # Blockquote
        if line.startswith('>'):
            bq = []
            while i < len(lines) and lines[i].startswith('>'):
                bq.append(inline(lines[i].lstrip('>').strip()))
                i += 1
            out.append('<blockquote style="border-left:3px solid #cbd5e1;padding:6px 14px;'
                       f'margin:10px 0;color:#64748b">{"<br>".join(bq)}</blockquote>')
            continue

        # Table
        if '|' in line and i + 1 < len(lines) and re.match(r'^[\s|:\-]+$', lines[i + 1]):
            hdrs = [c.strip() for c in line.strip().strip('|').split('|')]
            i += 2
            rows = []
            while i < len(lines) and '|' in lines[i] and lines[i].strip():
                rows.append([c.strip() for c in lines[i].strip().strip('|').split('|')])
                i += 1
            t = '<table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:0.88rem">'
            t += '<tr>' + ''.join(
                f'<th style="padding:8px 10px;border:1px solid #e2e8f0;background:#f8fafc;'
                f'font-weight:600;text-align:left">{inline(h)}</th>' for h in hdrs) + '</tr>'
            for r in rows:
                t += '<tr>' + ''.join(
                    f'<td style="padding:6px 10px;border:1px solid #e2e8f0">{inline(c)}</td>'
                    for c in r) + '</tr>'
            out.append(t + '</table>')
            continue

        # Unordered list
        if re.match(r'^[-*]\s', line):
            out.append('<ul style="margin:8px 0;padding-left:22px">')
            while i < len(lines) and re.match(r'^[-*]\s', lines[i]):
                content = re.sub(r'^[-*]\s+', '', lines[i])
                out.append(f'<li style="margin:3px 0">{inline(content)}</li>')
                i += 1
            out.append('</ul>')
            continue

        # Ordered list
        if re.match(r'^\d+\.\s', line):
            out.append('<ol style="margin:8px 0;padding-left:22px">')
            while i < len(lines) and re.match(r'^\d+\.\s', lines[i]):
                content = re.sub(r'^\d+\.\s+', '', lines[i])
                out.append(f'<li style="margin:3px 0">{inline(content)}</li>')
                i += 1
            out.append('</ol>')
            continue

        # Empty line
        if not line.strip():
            i += 1
            continue

        # Paragraph
        para = []
        while (i < len(lines) and lines[i].strip()
               and not re.match(r'^(#{1,6}\s|[-*]\s|\d+\.\s|>|```|-{3,}\s*$|.*\|.*\|)', lines[i])):
            para.append(lines[i])
            i += 1
        if para:
            out.append(f'<p style="margin:8px 0;line-height:1.7">{inline(" ".join(para))}</p>')
            continue
        i += 1

    return '\n'.join(out)


# ── Force-Directed Graph Layout ──────────────────────────────────────

def force_layout(nodes, edges, w, h, iters=300):
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]["id"]: (w / 2, h / 2)}

    random.seed(42)
    pos = {n["id"]: [random.uniform(100, w - 100), random.uniform(80, h - 80)] for n in nodes}
    k = math.sqrt(w * h / max(len(nodes), 1)) * 0.8

    edge_idx = {}
    for e in edges:
        edge_idx.setdefault(e["source"], set()).add(e["target"])
        edge_idx.setdefault(e["target"], set()).add(e["source"])

    for it in range(iters):
        disp = {n["id"]: [0.0, 0.0] for n in nodes}
        # Repulsion
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                dx = pos[a["id"]][0] - pos[b["id"]][0]
                dy = pos[a["id"]][1] - pos[b["id"]][1]
                d = max(math.hypot(dx, dy), 0.5)
                f = (k * k) / d
                fx, fy = f * dx / d, f * dy / d
                disp[a["id"]][0] += fx
                disp[a["id"]][1] += fy
                disp[b["id"]][0] -= fx
                disp[b["id"]][1] -= fy
        # Attraction
        for e in edges:
            s, t = e["source"], e["target"]
            if s not in pos or t not in pos:
                continue
            dx = pos[t][0] - pos[s][0]
            dy = pos[t][1] - pos[s][1]
            d = max(math.hypot(dx, dy), 0.5)
            f = (d * d) / k
            fx, fy = f * dx / d, f * dy / d
            disp[s][0] += fx
            disp[s][1] += fy
            disp[t][0] -= fx
            disp[t][1] -= fy
        # Centering
        for nid in pos:
            disp[nid][0] -= (pos[nid][0] - w / 2) * 0.01
            disp[nid][1] -= (pos[nid][1] - h / 2) * 0.01
        # Apply
        temp = max(0.05, 1.0 - it / iters)
        for n in nodes:
            nid = n["id"]
            d = max(math.hypot(*disp[nid]), 0.1)
            scale = min(d, temp * 40) / d
            pos[nid][0] = max(90, min(w - 90, pos[nid][0] + disp[nid][0] * scale))
            pos[nid][1] = max(60, min(h - 60, pos[nid][1] + disp[nid][1] * scale))

    return pos


def make_graph_svg(papers, w=950, h=480):
    if not papers:
        return (f'<svg width="{w}" height="100" xmlns="http://www.w3.org/2000/svg">'
                f'<text x="{w // 2}" y="50" text-anchor="middle" fill="#94a3b8" '
                f'font-family="system-ui" font-size="15">No papers yet</text></svg>')

    nodes = [{"id": p["id"], "title": p["title"], "cat": p["category"]} for p in papers]
    id_set = {p["id"] for p in papers}
    edges, seen = [], set()
    for p in papers:
        for rid in p.get("related_paper_ids", []):
            if rid in id_set:
                key = tuple(sorted([p["id"], rid]))
                if key not in seen:
                    edges.append({"source": p["id"], "target": rid})
                    seen.add(key)

    pos = force_layout(nodes, edges, w, h)
    svg = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
           f'style="font-family:system-ui,sans-serif">']

    # Edges
    for e in edges:
        if e["source"] in pos and e["target"] in pos:
            x1, y1 = pos[e["source"]]
            x2, y2 = pos[e["target"]]
            svg.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
                       f'stroke="#94a3b8" stroke-width="1.5" opacity="0.5"/>')
    # Nodes
    for n in nodes:
        if n["id"] not in pos:
            continue
        x, y = pos[n["id"]]
        color = cat_info(n["cat"])[1]
        title_esc = html_mod.escape(n["title"])
        short = html_mod.escape(n["title"][:30] + ('...' if len(n["title"]) > 30 else ''))
        cat_page = n["cat"]
        svg.append(
            f'<a href="{cat_page}.html#{html_mod.escape(n["id"])}" style="text-decoration:none">'
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="10" fill="{color}" '
            f'stroke="white" stroke-width="2.5" style="cursor:pointer">'
            f'<title>{title_esc}</title></circle>'
            f'<text x="{x:.0f}" y="{y + 24:.0f}" text-anchor="middle" '
            f'font-size="10" fill="#64748b">{short}</text></a>')

    # Legend
    ly = h - 25
    lx = 15
    for i, (cat, (label, color, _)) in enumerate(CATEGORIES.items()):
        ox = lx + i * 110
        svg.append(f'<circle cx="{ox}" cy="{ly}" r="6" fill="{color}"/>')
        svg.append(f'<text x="{ox + 11}" y="{ly + 4}" font-size="12" fill="#64748b">{label}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


# ── HTML Page ────────────────────────────────────────────────────────

CSS = """
:root { --bg:#f0f2f5; --sf:#fff; --tx:#0f172a; --mt:#64748b; --bd:#e2e8f0; }
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--tx);line-height:1.7}
.w{max-width:1100px;margin:0 auto;padding:32px 24px}
a{color:#2563eb}a:hover{text-decoration:underline}
h1{font-size:1.8rem;font-weight:800;letter-spacing:-0.02em}
.sub{color:var(--mt);margin-bottom:28px;font-size:0.95rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:32px}
.cd{background:var(--sf);border-radius:14px;padding:22px;text-decoration:none;color:var(--tx);
  border:2px solid var(--bd);transition:transform .15s,box-shadow .15s;display:block}
.cd:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(0,0,0,0.08);text-decoration:none}
.cd .n{font-size:2.2rem;font-weight:800}
.cd .l{font-weight:700;font-size:1.05rem;margin-top:2px}
.cd .d{font-size:0.8rem;color:var(--mt);margin-top:4px}
.sec{background:var(--sf);border-radius:14px;padding:28px;margin-bottom:22px;
  box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.sec h2{font-size:1.2rem;margin-bottom:16px}
.pp{background:var(--sf);border-radius:14px;padding:28px;margin-bottom:18px;
  border-left:5px solid var(--bd);box-shadow:0 1px 3px rgba(0,0,0,0.04)}
.pp .pt{font-size:1.25rem;font-weight:700;margin-bottom:6px}
.pp .pm{font-size:0.82rem;color:var(--mt);margin-bottom:14px}
.pp .cr{background:#f1f5f9;padding:12px 16px;border-radius:8px;margin-bottom:16px;font-weight:500}
details{margin:8px 0}
summary{cursor:pointer;font-weight:600;padding:7px 0;user-select:none;font-size:0.95rem}
summary:hover{color:#3b82f6}
details[open]>summary{margin-bottom:8px;color:#2563eb}
.dt{padding:6px 0 6px 18px;border-left:2px solid var(--bd)}
.tg{display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.73rem;font-weight:600;margin:0 3px 4px 0}
.bk{display:inline-block;color:var(--mt);text-decoration:none;margin-bottom:22px;font-weight:500;font-size:0.95rem}
.bk:hover{color:var(--tx)}
.tl{margin-top:8px}
.tr{display:flex;gap:14px;padding:9px 0;border-bottom:1px solid var(--bd);align-items:baseline}
.td{font-size:0.82rem;color:var(--mt);min-width:85px;flex-shrink:0}
.tc{font-size:0.72rem;font-weight:600;padding:2px 9px;border-radius:999px;flex-shrink:0}
.tt{font-size:0.92rem}.tt a{text-decoration:none}.tt a:hover{text-decoration:underline}
.empty{text-align:center;padding:48px;color:var(--mt);font-size:1rem}
.gbox{overflow-x:auto;margin-top:8px}
img{max-width:100%}
"""


def html_page(title, body):
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{html_mod.escape(title)}</title>
<style>{CSS}</style>
</head>
<body><div class="w">
{body}
</div></body>
</html>'''


# ── Generate Overview ────────────────────────────────────────────────

def generate_overview(papers):
    cats = {}
    for p in papers:
        cats.setdefault(p["category"], []).append(p)
    for c in CATEGORIES:
        cats.setdefault(c, [])

    # Category cards
    cards = ''
    ordered = list(CATEGORIES.keys()) + sorted(c for c in cats if c not in CATEGORIES)
    for cat in ordered:
        label, color, desc = cat_info(cat)
        n = len(cats.get(cat, []))
        cards += (f'<a class="cd" href="{cat}.html" style="border-color:{color}40">'
                  f'<div class="n" style="color:{color}">{n}</div>'
                  f'<div class="l">{label}</div>'
                  f'<div class="d">{html_mod.escape(desc)}</div></a>\n')

    # Graph
    graph = make_graph_svg(papers)

    # Timeline
    recent = sorted(papers, key=lambda p: p.get("date_read", p.get("date", "")), reverse=True)
    rows = ''
    for p in recent[:30]:
        label, color, _ = cat_info(p["category"])
        date = p.get("date_read", p.get("date", ""))
        pid = html_mod.escape(p["id"])
        rows += (f'<div class="tr">'
                 f'<span class="td">{date}</span>'
                 f'<span class="tc" style="background:{color}15;color:{color}">{label}</span>'
                 f'<span class="tt"><a href="{p["category"]}.html#{pid}">'
                 f'{html_mod.escape(p["title"])}</a></span></div>\n')

    if not rows:
        rows = '<div class="empty">No papers yet. Use the paper-reader skill to start.</div>'

    total = len(papers)
    active = len([c for c in cats if cats[c]])
    body = f'''<h1>AI Infra Paper Knowledge Base</h1>
<p class="sub">{total} paper{"s" if total != 1 else ""} across {active} active categor{"ies" if active != 1 else "y"}</p>

<div class="grid">
{cards}
</div>

<div class="sec">
<h2>Paper Relationships</h2>
<div class="gbox">{graph}</div>
</div>

<div class="sec">
<h2>Reading Timeline</h2>
<div class="tl">{rows}</div>
</div>'''

    out = DB_DIR / "overview.html"
    out.write_text(html_page("AI Infra Papers", body))
    print(f"  wrote {out}")


# ── Generate Category Page ───────────────────────────────────────────

def generate_category_page(all_papers, category):
    label, color, desc = cat_info(category)
    papers = sorted(
        [p for p in all_papers if p["category"] == category],
        key=lambda p: p.get("date", ""),
    )

    cards = ''
    for p in papers:
        pid = p["id"]
        tags = ''.join(
            f'<span class="tg" style="background:{color}12;color:{color}">'
            f'{html_mod.escape(t)}</span>' for t in p.get("secondary_tags", []))

        # Notes
        notes_file = NOTES_DIR / f"{pid}.md"
        notes_html = ''
        if notes_file.exists():
            notes_html = md_to_html(notes_file.read_text())

        # Impact
        impact = p.get("infra_impact", {})
        impact_rows = ''
        for layer in ["algorithm", "kernel", "framework", "llm", "agent"]:
            if layer == category:
                continue
            val = impact.get(layer, "N/A")
            if val and val != "N/A":
                ll, lc, _ = cat_info(layer)
                impact_rows += (f'<tr><td style="padding:6px 10px;border:1px solid #e2e8f0;'
                                f'font-weight:600;color:{lc};width:100px">{ll}</td>'
                                f'<td style="padding:6px 10px;border:1px solid #e2e8f0">'
                                f'{html_mod.escape(val)}</td></tr>')
        impact_sec = ''
        if impact_rows:
            impact_sec = (f'<details><summary>Infrastructure Impact</summary>'
                          f'<div class="dt"><table style="width:100%;border-collapse:collapse;'
                          f'font-size:0.88rem">{impact_rows}</table></div></details>')

        # Open questions
        oq = p.get("open_questions", [])
        oq_sec = ''
        if oq:
            items = ''.join(f'<li style="margin:3px 0">{html_mod.escape(q)}</li>' for q in oq)
            oq_sec = (f'<details><summary>Open Questions</summary>'
                      f'<div class="dt"><ul style="padding-left:20px">{items}</ul></div></details>')

        # Related
        related = p.get("related_paper_ids", [])
        rel_sec = ''
        if related:
            items = ''
            for rid in related:
                rp = next((x for x in all_papers if x["id"] == rid), None)
                if rp:
                    rl = cat_info(rp["category"])[0]
                    items += (f'<li style="margin:3px 0"><a href="{rp["category"]}.html#{rid}">'
                              f'{html_mod.escape(rp["title"])}</a> '
                              f'<span class="tg" style="background:#f1f5f9;color:#64748b">'
                              f'{rl}</span></li>')
                else:
                    items += f'<li style="margin:3px 0">{html_mod.escape(rid)}</li>'
            rel_sec = (f'<details><summary>Related Papers</summary>'
                       f'<div class="dt"><ul style="padding-left:20px;list-style:none">'
                       f'{items}</ul></div></details>')

        # Summary
        summary_sec = ''
        if p.get("summary"):
            summary_sec = (f'<details open><summary>Summary</summary>'
                           f'<div class="dt">{md_to_html(p["summary"])}</div></details>')

        # Key findings
        kf = p.get("key_findings", [])
        kf_sec = ''
        if kf:
            items = ''.join(f'<li style="margin:3px 0">{html_mod.escape(f)}</li>' for f in kf)
            kf_sec = (f'<details><summary>Key Findings</summary>'
                      f'<div class="dt"><ul style="padding-left:20px">{items}</ul></div></details>')

        # Limitations
        lim = p.get("limitations", [])
        lim_sec = ''
        if lim:
            items = ''.join(f'<li style="margin:3px 0">{html_mod.escape(l)}</li>' for l in lim)
            lim_sec = (f'<details><summary>Limitations</summary>'
                       f'<div class="dt"><ul style="padding-left:20px">{items}</ul></div></details>')

        # Full notes
        notes_sec = ''
        if notes_html:
            notes_sec = (f'<details><summary>Full Deep Analysis</summary>'
                         f'<div class="dt">{notes_html}</div></details>')

        # Authors / source link
        authors = ', '.join(p.get("authors", []))
        url = p.get("url", "")
        src_link = f' | <a href="{html_mod.escape(url)}" target="_blank">Source</a>' if url else ''

        cards += f'''<div class="pp" id="{html_mod.escape(pid)}" style="border-left-color:{color}">
<div class="pt"><a href="{html_mod.escape(url)}" target="_blank" style="color:inherit;text-decoration:none">{html_mod.escape(p["title"])}</a></div>
<div class="pm">{html_mod.escape(authors)} | {p.get("date", "")}{src_link}</div>
<div style="margin-bottom:10px">{tags}</div>
<div class="cr">{html_mod.escape(p.get("core_contribution", ""))}</div>
{summary_sec}
{kf_sec}
{lim_sec}
{impact_sec}
{oq_sec}
{rel_sec}
{notes_sec}
</div>\n'''

    if not cards:
        cards = f'<div class="empty">No {label.lower()} papers yet.</div>'

    body = f'''<a class="bk" href="overview.html">&larr; Overview</a>
<h1 style="color:{color}">{label} Papers</h1>
<p class="sub">{len(papers)} paper{"s" if len(papers) != 1 else ""} &middot; {html_mod.escape(desc)}</p>
{cards}'''

    out = DB_DIR / f"{category}.html"
    out.write_text(html_page(f"{label} Papers", body))
    print(f"  wrote {out}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    db = json.loads((DB_DIR / "papers.json").read_text())
    papers = db.get("papers", [])
    print(f"Generating HTML from {len(papers)} papers...")

    generate_overview(papers)

    all_cats = set(CATEGORIES.keys())
    all_cats.update(p["category"] for p in papers)
    for cat in sorted(all_cats):
        generate_category_page(papers, cat)

    print(f"Done. Open file://{DB_DIR / 'overview.html'}")


if __name__ == "__main__":
    main()
