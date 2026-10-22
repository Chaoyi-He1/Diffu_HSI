"""Phase 2a Task 1: the cross-fitting split keeps every patch of a source image in one half, balances the halves
within one image per class, is deterministic, and the selection rule picks the lowest monitoring loss."""
import json
import os

import pytest

from scripts.paired_stats import source_image
from scripts.uq.select_epoch import select_epoch
from scripts.uq.split_halves import read_split, split_by_source_image, write_split


def fake_files():
    """3 classes; class c has 5 + c images with 1-4 patches each (so halves cannot be balanced by patch count)."""
    files = []
    for c in range(3):
        for img in range(5 + c):
            for patch in range(1 + (img % 4)):
                files.append(f'MatFlower60/Train/P00{c + 1}/{img}_{patch}.mat')
    return files


def test_every_file_lands_in_exactly_one_half():
    files = fake_files()
    s = split_by_source_image(files, seed=0)
    assert sorted(s['half1'] + s['half2']) == sorted(files)
    assert not set(s['half1']) & set(s['half2'])


def test_patches_of_one_image_never_straddle_the_halves():
    s = split_by_source_image(fake_files(), seed=0)
    imgs1 = {source_image(f) for f in s['half1']}
    imgs2 = {source_image(f) for f in s['half2']}
    assert not imgs1 & imgs2


def test_halves_are_balanced_per_class_within_one_image():
    s = split_by_source_image(fake_files(), seed=0)
    for cls, n in s['images_per_class'].items():
        assert abs(n['half1'] - n['half2']) <= 1, (cls, n)
    assert set(s['images_per_class']) == {'P001', 'P002', 'P003'}
    assert s['images_per_class']['P003'] == {'half1': 4, 'half2': 3} or s['images_per_class']['P003'] == {'half1': 3, 'half2': 4}


def test_split_is_deterministic_and_depends_on_the_seed():
    a, b, c = (split_by_source_image(fake_files(), seed=s) for s in (0, 0, 1))
    assert a == b and a['half1'] != c['half1']
    assert a['seed'] == 0 and c['seed'] == 1


def test_halves_are_sorted_lists():
    s = split_by_source_image(fake_files(), seed=0)
    assert s['half1'] == sorted(s['half1']) and s['half2'] == sorted(s['half2'])


def test_write_split_round_trips_and_refuses_overwrite(tmp_path):
    s = split_by_source_image(fake_files(), seed=0)
    p = tmp_path / 'uq' / 'split_halves.json'
    write_split(str(p), s)
    assert read_split(str(p)) == s
    with pytest.raises(FileExistsError):
        write_split(str(p), s)
    assert not list(tmp_path.glob('**/*.tmp'))


def test_select_epoch_takes_the_lowest_monitoring_loss_ties_earlier(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = [dict(epoch=1, loss=0.5, val_loss=0.45), dict(epoch=2, loss=0.4, val_loss=0.40),
            dict(epoch=3, loss=0.3, val_loss=0.40), dict(epoch=4, loss=0.2, val_loss=0.42)]
    (run / 'train_log.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    sel = select_epoch(str(run))
    assert sel == dict(epoch=2, val_loss=0.40, last_epoch=4, last_val_loss=0.42, n_val=4)


def test_select_epoch_skips_epochs_without_a_monitoring_loss(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    recs = [dict(epoch=1, loss=0.5), dict(epoch=2, loss=0.4, val_loss=0.41), dict(epoch=3, loss=0.3)]
    (run / 'train_log.jsonl').write_text('\n'.join(json.dumps(r) for r in recs) + '\n')
    assert select_epoch(str(run))['epoch'] == 2
    (run / 'train_log.jsonl').write_text(json.dumps(dict(epoch=1, loss=0.5)) + '\n')
    with pytest.raises(ValueError, match='no monitoring'):
        select_epoch(str(run))
