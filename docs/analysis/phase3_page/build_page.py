"""Build the Phase 3 results page.

Composes docs/phase3_sensor_results.src.html from
  * the Phase 1a' page's head (tokens, components, SVG styles, markers, tooltip) and its shared chart helpers, so every
    results page shares one visual system (the user's GDN-report style), found by text markers, not line numbers;
  * body.html (the Phase 3 content) and charts.js (its figures, charts, tables, slots and data-bound values);
then injects page_data.json (extract_page_data.py) into the /*__DATA__*/null placeholder -> docs/phase3_sensor_results.html,
and writes a standalone copy with the publish skeleton for local rendering to $PAGE_RENDER (optional).

    python docs/analysis/phase3_page/extract_page_data.py [--verdict <file>]   # refresh the data first
    python docs/analysis/phase3_page/build_page.py [--extract]                  # --extract runs the extractor first
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(os.path.dirname(HERE))
OLD = os.path.join(DOCS, 'phase1a_prime_results.src.html')
SRC = os.path.join(DOCS, 'phase3_sensor_results.src.html')
OUT = os.path.join(DOCS, 'phase3_sensor_results.html')


def main():
    if '--extract' in sys.argv[1:]:
        subprocess.run([sys.executable, os.path.join(HERE, 'extract_page_data.py')]
                       + [a for a in sys.argv[1:] if a != '--extract'], check=True)
    old = open(OLD).read()
    head_end = old.index('<div id="tip" hidden></div>') + len('<div id="tip" hidden></div>')
    head = old[:head_end].replace('<title>Phase 1a′ Results</title>', '<title>Phase 3 Results</title>', 1)
    assert '<title>Phase 3 Results</title>' in head
    h0 = old.index("const NS='http://www.w3.org/2000/svg';")
    h1 = old.index('/* ---------- figures ---------- */')
    helpers = old[h0:h1]
    body = open(os.path.join(HERE, 'body.html')).read()
    charts = open(os.path.join(HERE, 'charts.js')).read()
    src = head + '\n\n' + body + '\n<script>\nconst D=/*__DATA__*/null;\n(function(){\n' + helpers + charts + '\n})();\n</script>\n'
    with open(SRC, 'w') as f:
        f.write(src)

    data = open(os.path.join(HERE, 'page_data.json')).read()
    page = json.loads(data)
    assert src.count('/*__DATA__*/null') == 1
    out = src.replace('/*__DATA__*/null', data.replace('</', '<\\/'))
    with open(OUT, 'w') as f:
        f.write(out)
    render = os.environ.get('PAGE_RENDER')
    if render:
        skel = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<style>:root{color-scheme:light;box-sizing:border-box}body{margin:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
        with open(render, 'w') as f:
            f.write(skel + out + '</body></html>')
        print('standalone render copy ->', render)
    print('src', len(src), 'bytes; page', len(out), 'bytes ->', OUT)
    st = page['status']
    print(f"data built {page['generated']}; verdict {st['verdict_file']} (B clauses current: {st['verdict_b_valid']})")
    print('PENDING: ' + (', '.join(st['pending']) if st['pending'] else 'none'))


if __name__ == '__main__':
    main()
