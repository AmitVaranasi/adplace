import numpy as np

from pipeline.maskstore import MaskStore, MaskWriter


def _masks(n=6, h=40, w=60, seed=0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, h, w), bool)
    for i in range(n):
        if i == 2:
            continue  # an empty frame
        y, x = rng.integers(0, h - 10), rng.integers(0, w - 12)
        masks[i, y:y + rng.integers(3, 10), x:x + rng.integers(3, 12)] = rng.random() > 0.0
        masks[i, y, x] = False  # not a plain rectangle
    return masks


def test_round_trip_full_masks(tmp_path):
    masks = _masks()
    w = MaskWriter(*masks.shape)
    for i, m in enumerate(masks):
        w.add(i, mask=m)
    w.save(tmp_path / "m.npz")
    store = MaskStore(tmp_path / "m.npz")
    assert len(store) == len(masks)
    assert all(np.array_equal(a, b) for a, b in zip(store, masks))
    assert not store.any(2) and store.any(0)
    assert np.array_equal(store[-1], masks[-1])


def test_box_local_input_and_merging_objects(tmp_path):
    w = MaskWriter(2, 30, 30)
    a = np.ones((4, 5), bool)
    w.add(0, box=(2, 3), local=a)
    w.add(0, box=(20, 20), local=np.ones((3, 3), bool))  # a second object in the same frame
    w.save(tmp_path / "m.npz")
    m = MaskStore(tmp_path / "m.npz")[0]
    assert m[3:7, 2:7].all() and m[20:23, 20:23].all() and m.sum() == 20 + 9
    assert not MaskStore(tmp_path / "m.npz")[1].any()


def test_storage_is_much_smaller_than_dense(tmp_path):
    n, h, w = 50, 720, 1280
    wr = MaskWriter(n, h, w)
    for i in range(n):
        m = np.zeros((h, w), bool)
        m[200 + i:400 + i, 300:700] = True
        wr.add(i, mask=m)
    wr.save(tmp_path / "m.npz")
    assert (tmp_path / "m.npz").stat().st_size < n * h * w / 8 / 50
