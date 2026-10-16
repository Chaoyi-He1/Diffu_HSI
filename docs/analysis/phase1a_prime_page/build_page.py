"""Inject compact.json into the page source → docs/phase1a_prime_results.html (and a scratch copy for rendering)."""
import shutil

W = '/data/chaoyi_he/HSI/Diffu/.claude/worktrees/phase1-plan/docs/'
P = '/home/staff/c/chaoyi_he/.claude/jobs/97dc7552/tmp/page/'
src = open(W + 'phase1a_prime_results.src.html').read()
data = open(P + 'compact.json').read()
assert src.count('/*__DATA__*/null') == 1
out = src.replace('/*__DATA__*/null', data)
open(W + 'phase1a_prime_results.html', 'w').write(out)
# a standalone copy with the publish skeleton for local rendering
skel = ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        '<style>:root{color-scheme:light;box-sizing:border-box}body{margin:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif}img{max-width:100%}[hidden]{display:none!important}</style></head><body>')
open(P + 'render.html', 'w').write(skel + out + '</body></html>')
print('built', len(out), 'bytes')
