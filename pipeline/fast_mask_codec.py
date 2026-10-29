"""Bit-exact Numba implementation of meta_sam_parser's raster codecs."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter
from types import FunctionType

import numpy as np
from numba import njit

import meta_sam_parser._mask_codec as _codec
from meta_sam_parser import (
    SegmentationMaskRecord,
    parse_responses_stream,
    video_segmentation_format,
)
from meta_sam_parser._errors import InvalidSegmentationMaskError

# Freeze the reference functions' globals before install() changes the module.
_NAMES = (
    "_unpack",
    "_decode_lossless_raster",
    "_decode_raster",
    "_encode_raster",
    "decode_mask_to_raster",
)
_OLD = {name: getattr(_codec, name) for name in _NAMES}
_REFERENCE = dict(vars(_codec))
for _name, _function in _OLD.items():
    _REFERENCE[_name] = FunctionType(
        _function.__code__,
        _REFERENCE,
        _name,
        _function.__defaults__,
        _function.__closure__,
    )
original_decode_mask_to_raster = _REFERENCE["decode_mask_to_raster"]

_DIGITS = np.asarray(_codec._CHARACTER_VALUES, dtype=np.int16)
_MESSAGES = (
    "",
    "Mask payload is missing its length prefix.",
    "Mask payload contains an invalid character.",
    "Mask payload length is unsupported.",
    "Mask payload length does not match its prefix.",
    "Mask payload contains an out-of-range group.",
    "Mask payload contains an out-of-range tail.",
    "Mask payload is not in canonical form.",
    "Lossless mask payload is truncated.",
    "Mask payload is missing its finalization.",
    "Mask payload has invalid coding state.",
    "Mask payload ended before decoding completed.",
    "Lossless mask payload uses an invalid predictor.",
    "Lossless mask payload is malformed.",
    "Decoded mask is not binary.",
)


def _checked(result):
    output, error = result
    if error:
        raise InvalidSegmentationMaskError(_MESSAGES[error])
    return output.tobytes()


@njit(cache=True, nogil=True)
def _unpack_core(text):
    empty = np.empty(0, np.uint8)
    if len(text) < 5:
        return empty, 1

    length = 0
    for i in range(5):
        digit = _DIGITS[text[i]]
        if digit < 0 or digit >= 85:
            return empty, 2
        length = length * 85 + digit

    remainder = length % 4
    expected = 5 + length // 4 * 5
    if remainder:
        expected += remainder + 1
    if len(text) != expected:
        return empty, 4

    out = np.empty(length, np.uint8)
    source = 5
    for destination in range(0, length - remainder, 4):
        value = 0
        for i in range(5):
            digit = _DIGITS[text[source + i]]
            if digit < 0 or digit >= 85:
                return empty, 2
            value = value * 85 + digit
        if value > 0xFFFFFFFF:
            return empty, 5
        for i in range(4):
            out[destination + i] = (value >> (24 - i * 8)) & 255
        source += 5

    if remainder:
        value = 0
        for i in range(5):
            digit = 84
            if i < remainder + 1:
                digit = _DIGITS[text[source + i]]
                if digit < 0 or digit >= 85:
                    return empty, 2
            value = value * 85 + digit
        if value > 0xFFFFFFFF:
            return empty, 6

        canonical = 0
        for i in range(remainder):
            byte = (value >> (24 - i * 8)) & 255
            out[length - remainder + i] = byte
            canonical |= byte << (24 - i * 8)

        # Full groups and the fixed-width prefix are unique. Only a partial
        # group's discarded low bytes can make the spelling noncanonical.
        digits = np.empty(5, np.int64)
        for i in range(4, -1, -1):
            digits[i] = canonical % 85
            canonical //= 85
        for i in range(remainder + 1):
            if digits[i] != _DIGITS[text[source + i]]:
                return empty, 7

    return out, 0


def _unpack(payload):
    try:
        text = payload.encode("ascii")
    except UnicodeEncodeError:
        # Preserve the reference's UTF-16 length and error precedence.
        return _REFERENCE["_unpack"](payload)
    return _checked(_unpack_core(np.frombuffer(text, np.uint8)))


@njit(cache=True, nogil=True)
def _start(data):
    # low, range, code, position, scaled; signed int64 intermediates, explicit
    # 32-bit wrapping exactly where the reference uses _u32.
    state = np.array([0, 0xFFFFFFFF, 0, 4, 0], dtype=np.int64)
    for i in range(4):
        state[2] = ((state[2] << 8) | int(data[i])) & 0xFFFFFFFF
    return state


@njit(cache=True, inline="always")
def _frequency(state, total):
    state[4] = state[1] // total
    if state[4] == 0:
        return 0, 10
    value = ((state[2] - state[0]) & 0xFFFFFFFF) // state[4]
    return min(value, total - 1), 0


@njit(cache=True, inline="always")
def _advance(state, data, cumulative, frequency):
    low = (state[0] + state[4] * cumulative) & 0xFFFFFFFF
    interval = state[4] * frequency
    while True:
        if (low ^ ((low + interval) & 0xFFFFFFFF)) < (1 << 24):
            pass
        elif interval < (1 << 16):
            interval = (-low & 0xFFFFFFFF) & 0xFFFF
        else:
            break

        if state[3] >= len(data):
            return 11
        state[2] = (
            (state[2] << 8) | int(data[state[3]])
        ) & 0xFFFFFFFF
        state[3] += 1
        low = (low << 8) & 0xFFFFFFFF
        interval = (interval << 8) & 0xFFFFFFFF

    state[0], state[1] = low, interval
    return 0


@njit(cache=True, inline="always")
def _predict(out, offset, x, y, mode, width):
    left = int(out[offset - 1]) if x else 0
    up = int(out[offset - width]) if y else 0
    if mode == 1:
        return up
    if mode == 2:
        return left
    if mode == 3:
        return (left + up) >> 1

    upper_left = int(out[offset - width - 1]) if x and y else 0
    if mode == 0:
        estimate = left + up - upper_left
        a = abs(estimate - left)
        b = abs(estimate - up)
        c = abs(estimate - upper_left)
        if a <= b and a <= c:
            return left
        return up if b <= c else upper_left

    if upper_left >= max(left, up):
        return min(left, up)
    if upper_left <= min(left, up):
        return max(left, up)
    return left + up - upper_left


@njit(cache=True, nogil=True)
def _lossless_core(data, width, height, selector):
    state = _start(data)
    length = width * height
    out = np.empty(length, np.uint8)
    residuals = np.empty(length, np.uint8)
    zeros = np.ones(32, np.uint32)
    first = np.ones((256, 256), np.uint16)
    first_totals = np.full(256, 256, np.int32)
    zeroth = np.full(256, 2, np.uint16)
    zeroth_total = 512

    # _get_order_two key = (previous << 8) | above: exactly 16 bits.
    # Reserve 32.125 MiB, initializing only rows actually encountered.
    second = np.empty((65536, 257), np.uint16)
    seen = np.zeros(65536, np.uint8)
    previous = 0

    for offset in range(length):
        x, y = offset % width, offset // width
        left = int(residuals[offset - 1] == 0) if x else 1
        up = int(residuals[offset - width] == 0) if y else 1
        ul = (
            int(residuals[offset - width - 1] == 0)
            if x and y else 1
        )
        ur = (
            int(residuals[offset - width + 1] == 0)
            if y and x < width - 1 else 1
        )
        z = (left | (up << 1) | (ul << 2) | (ur << 3)) << 1
        zero, nonzero = int(zeros[z]), int(zeros[z + 1])
        target, error = _frequency(state, zero + nonzero)
        if error:
            return out, error

        bit = int(target >= zero)
        error = _advance(
            state, data, zero if bit else 0, nonzero if bit else zero
        )
        if error:
            return out, error

        zeros[z + bit] = (int(zeros[z + bit]) + 32) & 0xFFFFFFFF
        if int(zeros[z]) + int(zeros[z + 1]) >= 16384:
            zeros[z] = max(int(zeros[z]) >> 1, 1)
            zeros[z + 1] = max(int(zeros[z + 1]) >> 1, 1)

        symbol = 0
        if bit:
            above = (
                int(residuals[offset - width]) if offset >= width else 0
            )
            key = (previous << 8) | above
            row = second[key]
            if not seen[key]:
                row[:] = 0
                seen[key] = 1

            total = (
                int(row[256])
                + int(first_totals[previous])
                + zeroth_total
                - int(row[0])
                - int(first[previous, 0])
                - int(zeroth[0])
            )
            target, error = _frequency(state, total)
            if error:
                return out, error

            cumulative = 0
            symbol = 1
            while True:
                frequency = (
                    int(row[symbol])
                    + int(first[previous, symbol])
                    + int(zeroth[symbol])
                )
                if cumulative + frequency > target:
                    break
                cumulative += frequency
                symbol += 1
                if symbol >= 256:
                    return out, 13

            error = _advance(state, data, cumulative, frequency)
            if error:
                return out, error

            row[symbol] = (int(row[symbol]) + 616) & 65535
            row[256] = (int(row[256]) + 616) & 65535
            if row[256] >= 50176:
                total = 0
                for value in range(256):
                    row[value] = int(row[value]) >> 1
                    total += int(row[value])
                row[256] = total & 65535

            first[previous, symbol] = (
                int(first[previous, symbol]) + 56
            ) & 65535
            first_totals[previous] += 56
            if first_totals[previous] >= 8192:
                total = 0
                for value in range(256):
                    scaled = max(int(first[previous, value]) >> 1, 1)
                    first[previous, value] = scaled
                    total += scaled
                first_totals[previous] = total

            zeroth[symbol] = (int(zeroth[symbol]) + 32) & 65535
            zeroth_total += 32
            if zeroth_total >= 4096:
                zeroth_total = 0
                for value in range(256):
                    zeroth[value] = int(zeroth[value]) >> 1
                    zeroth_total += int(zeroth[value])

        residuals[offset] = symbol

        # Match the reference: validate the predictor after decoding a symbol.
        if selector >= 5:
            return out, 12
        prediction = _predict(out, offset, x, y, selector, width)
        delta = -((symbol + 1) >> 1) if symbol & 1 else symbol >> 1
        out[offset] = (prediction + delta) & 255
        previous = symbol

    return out, 0


def _decode_lossless_raster(payload, width, height):
    if not payload.startswith("~"):
        raise InvalidSegmentationMaskError(
            "lossless mask payloads must start with ~."
        )
    packed = _unpack(payload[1:])
    if len(packed) < 5:
        raise InvalidSegmentationMaskError(_MESSAGES[8])
    return _checked(
        _lossless_core(
            np.frombuffer(packed, np.uint8)[1:],
            width,
            height,
            packed[0],
        )
    )


@njit(cache=True, inline="always")
def _context(padded, p, stride):
    prior, earlier = p - stride, p - 2 * stride
    return (
        int(padded[p - 1])
        | (int(padded[p - 2]) << 1)
        | (int(padded[prior - 2]) << 2)
        | (int(padded[prior - 1]) << 3)
        | (int(padded[prior]) << 4)
        | (int(padded[prior + 1]) << 5)
        | (int(padded[prior + 2]) << 6)
        | (int(padded[earlier - 2]) << 7)
        | (int(padded[earlier - 1]) << 8)
        | (int(padded[earlier]) << 9)
        | (int(padded[earlier + 1]) << 10)
        | (int(padded[earlier + 2]) << 11)
    )


@njit(cache=True, inline="always")
def _update(counts, z, bit):
    counts[z + bit] = (int(counts[z + bit]) + 14) & 65535
    if int(counts[z]) + int(counts[z + 1]) >= 4096:
        counts[z] = max(int(counts[z]) >> 1, 1)
        counts[z + 1] = max(int(counts[z + 1]) >> 1, 1)


@njit(cache=True, nogil=True)
def _bits_core(data, width, height, encode):
    counts = np.ones(8192, np.uint16)
    stride = width + 4
    padded = np.zeros(stride * (height + 2), np.uint8)
    out = np.empty(
        4 * width * height + 4 if encode else width * height,
        np.uint8,
    )
    state = (
        np.array([0, 0xFFFFFFFF, 0, 0, 0], np.int64)
        if encode else _start(data)
    )
    destination = 0

    for y in range(height):
        for x in range(width):
            p = (y + 2) * stride + 2 + x
            z = _context(padded, p, stride) << 1
            zero, one = int(counts[z]), int(counts[z + 1])

            if encode:
                bit = int(data[y * width + x])
                if bit != 0 and bit != 1:
                    return out, 14

                scaled = state[1] // (zero + one)
                low = (
                    state[0] + scaled * (zero if bit else 0)
                ) & 0xFFFFFFFF
                interval = scaled * (one if bit else zero)

                while True:
                    if (low ^ ((low + interval) & 0xFFFFFFFF)) < (1 << 24):
                        pass
                    elif interval < (1 << 16):
                        interval = (-low & 0xFFFFFFFF) & 65535
                    else:
                        break
                    out[destination] = (low >> 24) & 255
                    destination += 1
                    low = (low << 8) & 0xFFFFFFFF
                    interval = (interval << 8) & 0xFFFFFFFF

                state[0], state[1] = low, interval
            else:
                target, error = _frequency(state, zero + one)
                if error:
                    return out, error
                bit = int(target >= zero)
                error = _advance(
                    state, data, zero if bit else 0, one if bit else zero
                )
                if error:
                    return out, error
                out[y * width + x] = bit

            padded[p] = bit
            _update(counts, z, bit)

    if encode:
        for i in range(4):
            out[destination] = (state[0] >> 24) & 255
            destination += 1
            state[0] = (state[0] << 8) & 0xFFFFFFFF
        return out[:destination], 0

    return out, 0


def _decode_raster(payload, width, height):
    packed = _unpack(payload[1:])
    if len(packed) < 4:
        raise InvalidSegmentationMaskError(_MESSAGES[9])
    return _checked(
        _bits_core(np.frombuffer(packed, np.uint8), width, height, False)
    )


def _encode_raster(input_, width, height):
    values = (
        np.frombuffer(input_, np.uint8)
        if isinstance(input_, (bytes, bytearray, memoryview))
        else np.asarray(input_, dtype=np.int64)
    )
    packed = _checked(_bits_core(values, width, height, True))
    return "!" + _codec._pack(packed)


def decode_mask_to_raster(mask):
    # Preserve the public decoder's validation order and exception policy.
    if mask.encoding not in ("one_bit", "lossless"):
        raise _codec._invalid(
            f"Unsupported complete mask encoding: {mask.encoding}."
        )

    width = _codec._as_safe_integer(mask.width)
    height = _codec._as_safe_integer(mask.height)
    if width is None or height is None or width <= 0 or height <= 0:
        raise _codec._invalid(
            "Mask dimensions must be positive safe integers."
        )
    if width * height > _codec._MAXIMUM_SAFE_INTEGER:
        raise _codec._invalid(
            "Mask dimensions produce an unsafe decoded area."
        )
    if not isinstance(mask.payload, str):
        raise _codec._invalid("Mask payload must be a string.")

    marker = "!" if mask.encoding == "one_bit" else "~"
    if not mask.payload.startswith(marker):
        raise _codec._invalid(
            f"{mask.encoding} mask payloads must start with {marker}."
        )

    try:
        if mask.encoding == "lossless":
            coverage = _decode_lossless_raster(mask.payload, width, height)
            return (np.frombuffer(coverage, np.uint8) >= 129).tobytes()

        raster = _decode_raster(mask.payload, width, height)
        if (
            _encode_raster(np.frombuffer(raster, np.uint8), width, height)
            != mask.payload
        ):
            raise _codec._invalid(
                "Mask payload is not canonical or has invalid finalization."
            )
        return raster
    except (InvalidSegmentationMaskError, MemoryError):
        raise
    except Exception as error:
        raise _codec._invalid(
            "Mask payload could not be decoded.", error
        ) from error


def install():
    """Patch the codec and existing by-name imports; safe to call repeatedly."""
    replacements = {id(old): globals()[name] for name, old in _OLD.items()}

    # Includes _segmentation, _mask_conversion, meta_sam_parser, sam_api and
    # other loaded consumers, including aliases with different names.
    for module in tuple(sys.modules.values()):
        if module is None or module is sys.modules.get(__name__):
            continue
        for name, value in tuple(vars(module).items()):
            replacement = replacements.get(id(value))
            if replacement is not None:
                setattr(module, name, replacement)


def warmup():
    """Compile both decoding paths and the one-bit encoder before timing."""
    _decode_lossless_raster("~" + _codec._pack(bytes(5)), 1, 1)
    payload = _encode_raster([0], 1, 1)
    _decode_raster(payload, 1, 1)


def _records(path):
    """Use the same final-result Responses protocol as sam_api.parse_masks."""
    async def source():
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    async def run():
        parsed = parse_responses_stream(source(), video_segmentation_format())
        async with parsed:
            result = await parsed.final_result()

        if result.outcome.status != "completed":
            raise RuntimeError(
                f"SAM 3.1 stream incomplete: {result.outcome!r}"
            )
        errors = [
            d for d in result.diagnostics if d.severity == "error"
        ]
        if errors:
            raise ValueError(f"{path}: parser diagnostics: {errors!r}")

        # Include every accepted revision, rather than dropping older masks.
        return [
            r for r in result.records
            if isinstance(r, SegmentationMaskRecord) and r.frame is not None
        ]

    return asyncio.run(run())


def verify(paths):
    """Compare coverage and binary bytes; times measure raw raster decoding."""
    warmup()
    install()
    report = {
        "files": 0,
        "count": 0,
        "mismatches": 0,
        "original_seconds": 0.0,
        "fast_seconds": 0.0,
    }

    for path in paths:
        records = _records(path)
        report["files"] += 1
        for record in records:
            mask = record.mask
            # Checking raw coverage catches errors hidden by thresholding.
            name = (
                "_decode_lossless_raster"
                if mask.encoding == "lossless" else "_decode_raster"
            )
            start = perf_counter()
            expected_raw = _REFERENCE[name](
                mask.payload, mask.width, mask.height
            )
            report["original_seconds"] += perf_counter() - start

            start = perf_counter()
            actual_raw = globals()[name](
                mask.payload, mask.width, mask.height
            )
            report["fast_seconds"] += perf_counter() - start

            expected = (
                (np.frombuffer(expected_raw, np.uint8) >= 129).tobytes()
                if mask.encoding == "lossless" else expected_raw
            )
            actual = decode_mask_to_raster(mask)
            report["count"] += 1
            report["mismatches"] += int(
                expected_raw != actual_raw or expected != actual
            )

    report["speedup"] = (
        report["original_seconds"] / report["fast_seconds"]
        if report["fast_seconds"] else None
    )
    return report
