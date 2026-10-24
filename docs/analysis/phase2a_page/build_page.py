"""Build the Phase 2a results page.

Composes docs/phase2a_results.src.html from
  * the Phase 1a' page's head (tokens, components, SVG styles, markers, tooltip) and its shared chart helpers, so both
    pages share one visual system (the user's GDN-report style), found by text markers, not line numbers;
  * body.html (the Phase 2a content) and charts.js (its figures, charts, tables and slots);
then injects page_data.json (extract_page_data.py) into the /*__DATA__*/null placeholder -> docs/phase2a_results.html,
and writes a standalone copy with the publish skeleton for local rendering to $PAGE_RENDER (optional).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(DOCS, 'phase1a_prime_results.src.html')
SRC = os.path.join(DOCS, 'phase2a_results.src.html')
OUT = os.path.join(DOCS, 'phase2a_results.html')

old = open(OLD).read()
head_end = old.index('<div id="tip" hidden></div>') + len('<div id="tip" hidden></div>')
head = old[:head_end].replace('<title>Phase 1a′ Results</title>', '<title>Phase 2a Results</title>', 1)
assert '<title>Phase 2a Results</title>' in head
h0 = old.index("const NS='http://www.w3.org/2000/svg';")
h1 = old.index('/* ---------- figures ---------- */')
helpers = old[h0:h1]
body = open(os.path.join(HERE, 'body.html')).read()
charts = open(os.path.join(HERE, 'charts.js')).read()
src = (head + '\n\n' + body + '\n<script>\nconst D=/*__DATA__*/null;\n(function(){\n' + helpers + charts + '\n})();\n</script>\n')
open(SRC, 'w').write(src)

data = open(os.path.join(HERE, 'page_data.json')).read()
assert src.count('/*__DATA__*/null') == 1
out = src.replace('/*__DATA__*/null', data)
open(OUT, 'w').write(out)
render = os.environ.get('PAGE_RENDER')
if render:
    skel = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light;box-sizing:border-box}body{margin:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
    open(render, 'w').write(skel + out + '</body></html>')
print('src', len(src), 'bytes; page', len(out), 'bytes ->', OUT)
