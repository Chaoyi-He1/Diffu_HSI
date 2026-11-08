"""Build the Phase 3 results page.

Composes docs/phase3_sensor_results.src.html from
  * the Phase 1a' page's head (tokens, components, SVG styles, markers, tooltip) and its shared chart helpers, so every
    results page shares one visual system (the user's GDN-report style), found by text markers, not line numbers;
  * body.html (the Phase 3 content) and charts.js (its figures, charts, tables, slots and data-bound values);
then injects page_data.json (extract_page_data.py) into the /*__DATA__*/null placeholder -> docs/phase3_sensor_results.html,
and writes a standalone copy with the publish skeleton for local rendering to $PAGE_RENDER (optional).

    python docs/analysis/phase3_page/extract_page_data.py [--verdict <file>]   # refresh the data first
    python docs/analysis/phase3_page/build_page.py [--extract]                  # --extract runs the extractor first
    python docs/analysis/phase3_page/build_page.py --lang zh                    # Chinese build: body.zh.html + charts.zh.js
                                                                                # -> docs/phase3_sensor_results.zh{.src,}.html
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
# per-language inputs/outputs; 'en' is the original page and must stay byte-identical
LANGS = {
    'en': dict(body='body.html', charts='charts.js', src=SRC, out=OUT, title='Phase 3 Results'),
    'zh': dict(body='body.zh.html', charts='charts.zh.js', src=os.path.join(DOCS, 'phase3_sensor_results.zh.src.html'),
               out=os.path.join(DOCS, 'phase3_sensor_results.zh.html'), title='Phase 3 结果'),
}
ZH_FONT = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap">'


def parse_lang(argv):
    """Return (lang, argv without --lang/<value>); default 'en'."""
    lang, rest, i = 'en', [], 0
    while i < len(argv):
        a = argv[i]
        if a == '--lang' and i + 1 < len(argv):
            lang = argv[i + 1]; i += 2; continue
        if a.startswith('--lang='):
            lang = a.split('=', 1)[1]; i += 1; continue
        rest.append(a); i += 1
    if lang not in LANGS:
        sys.exit(f'--lang must be one of {sorted(LANGS)}, got {lang!r}')
    return lang, rest


def zh_head(head):
    """Chinese head: Noto Sans SC stylesheet after the Latin faces' link, and Noto Sans SC in the font stacks after the
    Latin faces (so Latin text renders exactly as on the en page) and before the system fallbacks."""
    latin = 'family=JetBrains+Mono:wght@400;600;700&display=swap">'
    assert head.count(latin) == 1
    head = head.replace(latin, latin + '\n' + ZH_FONT, 1)
    for old, new in (('--sans:"IBM Plex Sans",system-ui,', '--sans:"IBM Plex Sans","Noto Sans SC",system-ui,'),
                     ('--mono:"JetBrains Mono",ui-monospace,', '--mono:"JetBrains Mono","Noto Sans SC",ui-monospace,')):
        assert head.count(old) == 1, old
        head = head.replace(old, new, 1)
    return head


ZH_HELPERS = (("sm.textContent='Show the data';", "sm.textContent='显示数据';"),
              ("'min '+mn[1].toFixed(4)+' @ '+mn[0]", "'最小 '+mn[1].toFixed(4)+' @ t = '+mn[0]"))


def zh_helpers(helpers):
    """Chinese shared helpers: the two user-visible strings of the Phase 1a' helper block (the data-table toggle and the
    line chart's minimum label); everything else in the block is code."""
    for old, new in ZH_HELPERS:
        assert helpers.count(old) == 1, old
        helpers = helpers.replace(old, new, 1)
    return helpers


def main():
    lang, args = parse_lang(sys.argv[1:])
    L = LANGS[lang]
    if '--extract' in args:
        subprocess.run([sys.executable, os.path.join(HERE, 'extract_page_data.py')]
                       + [a for a in args if a != '--extract'], check=True)
    old = open(OLD, encoding='utf-8').read()
    head_end = old.index('<div id="tip" hidden></div>') + len('<div id="tip" hidden></div>')
    title = '<title>' + L['title'] + '</title>'
    head = old[:head_end].replace('<title>Phase 1a′ Results</title>', title, 1)
    assert title in head
    if lang == 'zh':
        head = zh_head(head)
    h0 = old.index("const NS='http://www.w3.org/2000/svg';")
    h1 = old.index('/* ---------- figures ---------- */')
    helpers = old[h0:h1]
    if lang == 'zh':
        helpers = zh_helpers(helpers)
    body = open(os.path.join(HERE, L['body']), encoding='utf-8').read()
    charts = open(os.path.join(HERE, L['charts']), encoding='utf-8').read()
    src = head + '\n\n' + body + '\n<script>\nconst D=/*__DATA__*/null;\n(function(){\n' + helpers + charts + '\n})();\n</script>\n'
    with open(L['src'], 'w', encoding='utf-8') as f:
        f.write(src)

    data = open(os.path.join(HERE, 'page_data.json'), encoding='utf-8').read()
    page = json.loads(data)
    assert src.count('/*__DATA__*/null') == 1
    out = src.replace('/*__DATA__*/null', data.replace('</', '<\\/'))
    with open(L['out'], 'w', encoding='utf-8') as f:
        f.write(out)
    render = os.environ.get('PAGE_RENDER')
    if render:
        skel = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                '<style>:root{color-scheme:light;box-sizing:border-box}body{margin:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
        with open(render, 'w', encoding='utf-8') as f:
            f.write(skel + out + '</body></html>')
        print('standalone render copy ->', render)
    print(f'[{lang}] src', len(src), 'bytes; page', len(out), 'bytes ->', L['out'])
    st = page['status']
    print(f"data built {page['generated']}; verdict {st['verdict_file']} (B clauses current: {st['verdict_b_valid']})")
    print('PENDING: ' + (', '.join(st['pending']) if st['pending'] else 'none'))


if __name__ == '__main__':
    main()
