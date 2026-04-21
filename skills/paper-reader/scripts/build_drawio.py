#!/usr/bin/env python3
"""Declarative drawio (mxfile) builder for paper architecture diagrams.

Use this from a Python spec file (one per paper) to generate the
multi-page .drawio that gets dropped into
/apps/feiyue/upstream/zhaifeiyue.github.io/assets/<paper-id>_arch.drawio
and embedded into the notes via {{drawio:...#page=N}}.

Spec format (Python dict per page)::

    PAGES = [
        {
            "name": "1. Overview",
            "size": (1600, 1000),
            "boxes": [
                # (id, label, x, y, w, h, style_name)
                ("in_text", "Text Input", 40, 40, 180, 50, "input_blue"),
                ("backbone", "Decoder x28", 600, 200, 220, 70, "module"),
            ],
            "edges": [
                # (id, src_id, dst_id, label, style)
                ("e1", "in_text", "backbone", "[seq]", "edge"),
            ],
            "groups": [
                # (id, label, x, y, w, h, style)  --- swimlane container; children
                # are placed by absolute coords inside the page (the group is just visual)
            ],
            "texts": [
                # (id, text, x, y, w, h, font_size, bold)
            ],
        },
        ...
    ]

Then::

    from build_drawio import build, write
    write(PAGES, "/apps/feiyue/upstream/zhaifeiyue.github.io/assets/foo_arch.drawio")

Available named styles (extend STYLES dict if needed).
"""

from html import escape as h
from typing import Iterable

# ---------- styles -----------------------------------------------------------

STYLES = {
    # input chips ---------------------------------------------------------
    "input_blue":   "rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=11;",
    "input_green":  "rounded=1;whiteSpace=wrap;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=11;",
    "input_yellow": "rounded=1;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;",
    "input_red":    "rounded=1;whiteSpace=wrap;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;",
    "input_purple": "rounded=1;whiteSpace=wrap;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=11;",
    "output":       "rounded=1;whiteSpace=wrap;fillColor=#f5f5f5;strokeColor=#666666;fontSize=11;",
    # internal boxes -------------------------------------------------------
    "linear":       "rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;",
    "norm":         "rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;",
    "act":          "rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;",
    "core":         "rounded=1;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=11;fontStyle=1;",
    "core_red":     "rounded=1;whiteSpace=wrap;fillColor=#f8cecc;strokeColor=#b85450;fontSize=11;fontStyle=1;",
    "module":       "rounded=1;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;fontStyle=1;",
    "module_red":   "rounded=1;whiteSpace=wrap;fillColor=#f8cecc;strokeColor=#b85450;fontSize=12;fontStyle=1;",
    "module_green": "rounded=1;whiteSpace=wrap;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;fontStyle=1;",
    "module_blue":  "rounded=1;whiteSpace=wrap;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;fontStyle=1;",
    "residual":     "rounded=1;whiteSpace=wrap;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;",
    "note":         "rounded=0;whiteSpace=wrap;fillColor=#fafafa;strokeColor=#bbbbbb;fontSize=10;align=left;verticalAlign=top;",
    # ellipse / decision ---------------------------------------------------
    "ellipse":      "ellipse;whiteSpace=wrap;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;",
    # text labels ----------------------------------------------------------
    "title":        "text;fontSize=16;fontStyle=1;fillColor=none;strokeColor=none;",
    "subtitle":     "text;fontSize=13;fontStyle=1;fillColor=none;strokeColor=none;",
    "caption":      "text;fontSize=10;fontStyle=2;fillColor=none;strokeColor=none;align=left;",
    # swimlane (group container) ------------------------------------------
    "lane_yellow":  "swimlane;startSize=24;fillColor=#f5f5f5;strokeColor=#d6b656;fontStyle=1;fontSize=12;rounded=1;",
    "lane_red":     "swimlane;startSize=24;fillColor=#f5f5f5;strokeColor=#b85450;fontStyle=1;fontSize=12;rounded=1;",
    "lane_blue":    "swimlane;startSize=24;fillColor=#f5f5f5;strokeColor=#6c8ebf;fontStyle=1;fontSize=12;rounded=1;",
    "lane_green":   "swimlane;startSize=24;fillColor=#f5f5f5;strokeColor=#82b366;fontStyle=1;fontSize=12;rounded=1;",
    "lane_purple":  "swimlane;startSize=24;fillColor=#f5f5f5;strokeColor=#9673a6;fontStyle=1;fontSize=12;rounded=1;",
    # edges ----------------------------------------------------------------
    "edge":         "edgeStyle=orthogonalEdgeStyle;",
    "edge_dashed":  "edgeStyle=orthogonalEdgeStyle;dashed=1;",
    "edge_thick":   "edgeStyle=orthogonalEdgeStyle;strokeWidth=2;",
    "edge_red":     "edgeStyle=orthogonalEdgeStyle;strokeColor=#b85450;strokeWidth=2;",
    "edge_curved":  "edgeStyle=orthogonalEdgeStyle;curved=1;",
    "edge_loop":    "edgeStyle=orthogonalEdgeStyle;dashed=1;exitX=0;exitY=0.5;entryX=0;entryY=0.5;",
}


def _style(name: str) -> str:
    return STYLES.get(name, name)


def _vertex(cell_id: str, value: str, x: int, y: int, w: int, h_: int,
            style: str, parent: str = "1") -> str:
    return (
        f'        <mxCell id="{cell_id}" value="{h(value)}" '
        f'style="{_style(style)}" vertex="1" parent="{parent}">\n'
        f'          <mxGeometry x="{x}" y="{y}" width="{w}" height="{h_}" as="geometry"/>\n'
        f'        </mxCell>\n'
    )


def _edge(cell_id: str, src: str, dst: str, label: str = "", style: str = "edge",
          parent: str = "1") -> str:
    out = (
        f'        <mxCell id="{cell_id}" style="{_style(style)}" edge="1" '
        f'source="{src}" target="{dst}" parent="{parent}">\n'
        f'          <mxGeometry relative="1" as="geometry"/>\n'
        f'        </mxCell>\n'
    )
    if label:
        out += (
            f'        <mxCell id="{cell_id}_lbl" value="{h(label)}" '
            f'style="edgeLabel;fontSize=9;fontColor=#333333;" vertex="1" '
            f'connectable="0" parent="{cell_id}">\n'
            f'          <mxGeometry x="-0.2" relative="1" as="geometry">'
            f'<mxPoint as="offset"/></mxGeometry>\n'
            f'        </mxCell>\n'
        )
    return out


def _page(idx: int, page: dict) -> str:
    name = page.get("name", f"Page {idx+1}")
    pid = f"page{idx+1}"
    w, ht = page.get("size", (1600, 1200))
    out = [
        f'  <diagram id="{pid}" name="{h(name)}">\n',
        f'    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" '
        f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
        f'pageWidth="{w}" pageHeight="{ht}" math="0" shadow="0">\n',
        '      <root>\n',
        '        <mxCell id="0"/>\n',
        '        <mxCell id="1" parent="0"/>\n',
    ]
    # groups (swimlanes) -- emit FIRST so inner boxes can reference them
    group_parents = {}
    for grp in page.get("groups", []):
        gid, label, gx, gy, gw, gh, gstyle = grp
        out.append(_vertex(gid, label, gx, gy, gw, gh, gstyle))
        group_parents[gid] = True
    # boxes
    for box in page.get("boxes", []):
        if len(box) == 7:
            bid, label, x, y, w_, h_, style = box
            parent = "1"
        else:  # 8 elements -> last is parent group id
            bid, label, x, y, w_, h_, style, parent = box
        out.append(_vertex(bid, label, x, y, w_, h_, style, parent=parent))
    # text labels (no shape, just floating text)
    for txt in page.get("texts", []):
        if len(txt) == 6:
            tid, label, x, y, w_, h_ = txt
            style = "title"
        elif len(txt) == 7:
            tid, label, x, y, w_, h_, style = txt
        else:
            tid, label, x, y, w_, h_, style, parent = txt
            out.append(_vertex(tid, label, x, y, w_, h_, style, parent=parent))
            continue
        out.append(_vertex(tid, label, x, y, w_, h_, style))
    # edges
    for edge in page.get("edges", []):
        if len(edge) == 4:
            eid, src, dst, label = edge
            style = "edge"
        else:
            eid, src, dst, label, style = edge
        out.append(_edge(eid, src, dst, label, style))
    out.append('      </root>\n')
    out.append('    </mxGraphModel>\n')
    out.append('  </diagram>\n')
    return "".join(out)


def build(pages: Iterable[dict]) -> str:
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n']
    parts.append('<mxfile host="paper-reader" modified="2026-04-18" agent="build_drawio.py" version="24.0" type="device">\n')
    for i, p in enumerate(pages):
        parts.append(_page(i, p))
    parts.append('</mxfile>\n')
    return "".join(parts)


def write(pages: Iterable[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(build(pages))


# ---------- convenience helpers for layered/repeated structures -------------

def lane(gid: str, label: str, x: int, y: int, w: int, h_: int,
        color: str = "yellow") -> tuple:
    """A swimlane container."""
    return (gid, label, x, y, w, h_, f"lane_{color}")


def box(bid: str, label: str, x: int, y: int, w: int, h_: int,
        style: str = "linear", parent: str = "1") -> tuple:
    return (bid, label, x, y, w, h_, style, parent) if parent != "1" else (bid, label, x, y, w, h_, style)


def edge(eid: str, src: str, dst: str, label: str = "", style: str = "edge") -> tuple:
    return (eid, src, dst, label, style)


def title(tid: str, text: str, x: int, y: int, w: int = 600, h_: int = 30,
          style: str = "title") -> tuple:
    return (tid, text, x, y, w, h_, style)


if __name__ == "__main__":
    # simple self-test
    PAGES = [{
        "name": "Test",
        "size": (800, 400),
        "boxes": [
            ("in", "Input", 40, 40, 100, 40, "input_blue"),
            ("mid", "Linear(d→4d)", 200, 40, 140, 40, "linear"),
            ("out", "Output", 400, 40, 100, 40, "output"),
        ],
        "edges": [
            ("e1", "in", "mid", "[B,T,d]", "edge"),
            ("e2", "mid", "out", "[B,T,4d]", "edge"),
        ],
    }]
    print(build(PAGES))
