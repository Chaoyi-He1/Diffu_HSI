"""The 24 pre-registered Phase 3b settings (spec docs/superpowers/specs/2026-11-16-b-plus-and-zero-shot-generality-design.md
Part B). SCENARIOS[name] = dict(sensor_device, snr_db, scenario); GROUPS maps B1 .. B4 to the names in spec order.
`sd_eval.py --scenario NAME` evaluates one of them."""

DEVICE = 'device1'


def _s(name, snr_db, R=None, states=None, pixels=None, noise=None):
    return name, dict(sensor_device=DEVICE, snr_db=float(snr_db),
                      scenario=dict(name=name, R=R or {'kind': 'device'}, states=states, pixels=pixels,
                                    noise=noise or {'kind': 'gaussian'}))


_B1 = [_s('blend_025', 30, R={'kind': 'blend', 'alpha': 0.25}),
       _s('blend_050', 30, R={'kind': 'blend', 'alpha': 0.5}),
       _s('blend_075', 30, R={'kind': 'blend', 'alpha': 0.75}),
       _s('shift_1', 30, R={'kind': 'shift', 'bands': 1}),
       _s('shift_2', 30, R={'kind': 'shift', 'bands': 2}),
       _s('shift_4', 30, R={'kind': 'shift', 'bands': 4}),
       _s('gain_02', 30, R={'kind': 'gain', 'delta': 0.2, 'seed': 0}),
       _s('gain_05', 30, R={'kind': 'gain', 'delta': 0.5, 'seed': 0})]
_B2 = [_s(f'states_{S}_{snr}', snr, states=S) for snr in (40, 20) for S in (15, 10, 5)]
_B3 = [_s('dead_50', 30, pixels={'kind': 'dead', 'p': 0.5}),
       _s('dead_90', 30, pixels={'kind': 'dead', 'p': 0.9}),
       _s('stride_4_60', 60, pixels={'kind': 'stride', 'd': 4}),
       _s('stride_8_60', 60, pixels={'kind': 'stride', 'd': 8}),
       _s('stride_4_30', 30, pixels={'kind': 'stride', 'd': 4})]
_B4 = [_s('poisson_30', 30, noise={'kind': 'poisson'}),
       _s('poisson_20', 20, noise={'kind': 'poisson'}),
       _s('sat_90', 30, noise={'kind': 'saturation', 'q': 0.9}),
       _s('sat_70', 30, noise={'kind': 'saturation', 'q': 0.7}),
       _s('quant_4', 40, noise={'kind': 'quantisation', 'bits': 4})]

GROUPS = {g: [n for n, _ in items] for g, items in (('B1', _B1), ('B2', _B2), ('B3', _B3), ('B4', _B4))}
SCENARIOS = dict(_B1 + _B2 + _B3 + _B4)
assert len(SCENARIOS) == 24
