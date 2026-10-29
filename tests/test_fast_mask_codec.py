from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pytest
from meta_sam_parser import SegmentationMask
from meta_sam_parser._errors import InvalidSegmentationMaskError

from pipeline import fast_mask_codec as fast

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", [
    "dundas-3ff676f856f3-fc7fe7ff-events.jsonl",
    "patio_waiter-a19a14a8dfcb-2b3fb1e9-events.jsonl",
])
def test_recorded_masks(filename):
    path = ROOT / "cache" / filename
    if not path.exists():
        pytest.skip("Recorded SAM events are not available")

    report = fast.verify([path])
    assert report["count"] > 0
    assert report["mismatches"] == 0

    # Independently check the original public decoder's binary threshold.
    for record in fast._records(path)[:3]:
        assert (
            fast.decode_mask_to_raster(record.mask)
            == fast.original_decode_mask_to_raster(record.mask)
        )


@pytest.mark.parametrize("selector", range(5))
def test_lossless_predictors(selector):
    rng = np.random.default_rng(100 + selector)
    # Enough entropy-coded bytes prevent truncation; exercise all predictors.
    packed = bytes([selector]) + rng.bytes(4096)
    payload = "~" + fast._codec._pack(packed)
    expected = fast._REFERENCE["_decode_lossless_raster"](payload, 32, 32)
    assert fast._decode_lossless_raster(payload, 32, 32) == expected


@pytest.mark.parametrize("shape", [(1, 1), (7, 9), (64, 64)])
def test_one_bit_and_canonicality(shape):
    h, w = shape
    values = np.random.default_rng(w).integers(
        0, 2, w * h, dtype=np.uint8
    )
    payload = fast._REFERENCE["_encode_raster"](values, w, h)
    assert fast._encode_raster(values, w, h) == payload
    assert fast._encode_raster(values.tobytes(), w, h) == payload

    mask = SegmentationMask("one_bit", payload, w, h)
    assert (
        fast.decode_mask_to_raster(mask)
        == fast.original_decode_mask_to_raster(mask)
        == values.tobytes()
    )


@pytest.mark.parametrize("encoding,payload,width,height", [
    ("lossless", "", 1, 1),
    ("lossless", "~", 1, 1),
    ("lossless", "~!!!!", 1, 1),
    ("lossless", "~!!!!!", 1, 1),
    ("lossless", "~!!!!,", 1, 1),
    ("lossless", "~😀!!!!", 1, 1),
    ("lossless", "~" + fast._codec._pack(bytes(4)), 1, 1),
    ("lossless", "~" + fast._codec._pack(b"\x05" + bytes(4)), 1, 1),
    ("lossless", "~" + fast._codec._pack(bytes(5)), 1000, 1000),
    ("one_bit", "!" + fast._codec._pack(bytes(3)), 1, 1),
    ("one_bit", "!" + fast._codec._pack(bytes(5)), 1, 1),
    ("lossless", "~", True, 1),
    ("lossless", "~", 0, 1),
])
def test_malformed_error_parity(encoding, payload, width, height):
    mask = SegmentationMask(encoding, payload, width, height)
    with pytest.raises(InvalidSegmentationMaskError) as reference:
        fast.original_decode_mask_to_raster(mask)
    with pytest.raises(type(reference.value)) as actual:
        fast.decode_mask_to_raster(mask)
    assert str(actual.value) == str(reference.value)
    assert actual.value.code == reference.value.code


def test_unpack():
    rng = np.random.default_rng(85)
    for size in range(128):
        raw = rng.bytes(size)
        payload = fast._codec._pack(raw)
        assert (
            fast._unpack(payload)
            == fast._REFERENCE["_unpack"](payload)
            == raw
        )


def test_install_updates_imported_aliases():
    import meta_sam_parser
    import meta_sam_parser._mask_conversion as conversion
    import meta_sam_parser._segmentation as segmentation
    from pipeline import sam_api

    consumer = ModuleType("_fast_codec_test_consumer")
    consumer.alias = fast._OLD["decode_mask_to_raster"]
    sys.modules[consumer.__name__] = consumer
    try:
        fast.install()
        fast.install()
        for module in (
            meta_sam_parser, fast._codec, conversion, segmentation, sam_api
        ):
            assert module.decode_mask_to_raster is fast.decode_mask_to_raster
        assert consumer.alias is fast.decode_mask_to_raster
        assert (
            fast._codec._decode_lossless_raster
            is fast._decode_lossless_raster
        )
        assert fast._codec._decode_raster is fast._decode_raster
    finally:
        del sys.modules[consumer.__name__]
