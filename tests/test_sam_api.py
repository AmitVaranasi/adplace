"""Offline SAM API tests using genuine meta-sam-parser wire payloads."""
from dataclasses import replace
import json

import numpy as np
import pytest
from meta_sam_parser import (
    ResponsesStreamLaneError,
    SegmentationMaskBounds,
    decode_mask_to_raster,
)
# No public encoder exists in 0.1.0. This private encoder produces canonical
# one_bit payloads; the real parser validates and decodes every test record.
from meta_sam_parser._mask_codec import _encode_raster

from pipeline import sam_api


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected HTTP request")

    monkeypatch.setattr(sam_api.requests, "post", forbidden)


def record(oid, raster, box, confidence=0.8, source=(12, 10)):
    """box uses half-open corners; the real wire uses inclusive x2/y2."""
    raster = np.asarray(raster, dtype=np.uint8)
    h, w = raster.shape
    payload = _encode_raster(raster.ravel().tolist(), w, h)
    x0, y0, x1, y1 = box
    sw, sh = source
    c = "" if confidence is None else f";c={confidence}"
    return (
        f"{oid}<|box;x1={x0};y1={y0};x2={x1 - 1};y2={y1 - 1}"
        f";w={sw};h={sh};c=0.1|>"
        f"<|mask;x=0;y=0;data={h},{w},{payload}{c}|>"
    )


def frame(fi, *records, newline=True):
    return f"<{fi}f>" + ",".join(records) + ("\n" if newline else "")


def events(chunks, finalizer="content_part"):
    """Real Responses lane identity, deltas, final text, and terminal event."""
    lane = {"item_id": "msg_sam_test", "output_index": 0, "content_index": 0}
    text = "".join(chunks)
    result = [
        {"type": "response.created",
         "response": {"id": "resp_sam_test", "status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": lane["item_id"], "type": "message",
                  "role": "assistant", "status": "in_progress", "content": []}},
        {"type": "response.content_part.added", **lane,
         "part": {"type": "output_text", "text": "", "annotations": []}},
    ]
    result.extend(
        {"type": "response.output_text.delta", **lane, "delta": chunk}
        for chunk in chunks
    )
    if finalizer in ("output_text", "both"):
        result.append({"type": "response.output_text.done", **lane, "text": text})
    if finalizer in ("content_part", "both"):
        # The installed package documents this as the live SAM API finalizer.
        result.append({
            "type": "response.content_part.done", **lane,
            "part": {"type": "output_text", "text": text, "annotations": []},
        })
    result.append({
        "type": "response.completed",
        "response": {"id": "resp_sam_test", "status": "completed"},
    })
    for i, event in enumerate(result):
        event["sequence_number"] = i
    return result


def parsed_record(raster, box, fi=0, oid="7", **kwargs):
    return sam_api.parse_masks(events([frame(fi, record(
        oid, raster, box, **kwargs
    ))]))[(oid, fi)]


def mock_api(monkeypatch, response_events):
    calls = []

    def upload(clip):
        calls.append(("upload", clip))
        return "file_test"

    def stream(file_id, prompt):
        calls.append(("stream", file_id, prompt))
        return iter(response_events)

    monkeypatch.setattr(sam_api, "upload", upload)
    monkeypatch.setattr(sam_api, "stream_events", stream)
    return calls


@pytest.mark.parametrize("finalizer", ["content_part", "output_text", "both"])
def test_parse_real_dict_events_and_paste_box_local_mask(finalizer):
    local = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    line = frame(2, record("7", local, (2, 1, 5, 3), confidence=0.75))
    recs = sam_api.parse_masks(events([line], finalizer))
    assert set(recs) == {("7", 2)}
    rec = recs[("7", 2)]
    assert rec.frame.frame_index == 2
    assert rec.bounds == SegmentationMaskBounds(2, 1, 5, 3)
    assert rec.revision == 1
    assert rec.confidence == pytest.approx(0.75)  # mask c, not box c
    decoded = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8)
    np.testing.assert_array_equal(decoded.reshape(2, 3), local)
    expected = np.zeros((6, 8), bool)
    expected[1:3, 2:5] = local
    actual = sam_api.paste(rec, 6, 8)
    assert actual.dtype == np.bool_
    np.testing.assert_array_equal(actual, expected)


def test_resize_raster_to_box_without_extra_endpoint():
    local = np.array([[1, 0], [0, 1]], dtype=bool)
    rec = parsed_record(local, (1, 1, 5, 7))
    expected = np.zeros((8, 7), bool)
    expected[1:7, 1:5] = np.repeat(np.repeat(local, 3, axis=0), 2, axis=1)
    np.testing.assert_array_equal(sam_api.paste(rec, 8, 7), expected)


def test_one_pixel_mask_at_bottom_right_frame_edge():
    rec = parsed_record([[1]], (7, 5, 8, 6), source=(8, 6))
    expected = np.zeros((6, 8), bool)
    expected[5, 7] = True
    np.testing.assert_array_equal(sam_api.paste(rec, 6, 8), expected)


def test_paste_box_partly_outside_destination_resizes_before_clipping():
    # Valid in the wire's 12x10 source; exercise clipping to a smaller viewport.
    # Resizing the clipped box instead of the full box would move these pixels.
    rec = parsed_record([[1, 0], [0, 1]], (3, 2, 7, 6))
    expected = np.zeros((4, 5), bool)
    expected[2:4, 3:5] = True
    np.testing.assert_array_equal(sam_api.paste(rec, 4, 5), expected)


def test_paste_negative_origin_crops_correct_local_pixels():
    local = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 1]], bool)
    rec = parsed_record(local, (0, 0, 4, 3))
    # Negative source boxes are rejected by the wire parser. Construct this
    # public record directly to test paste's defensive left/top clipping.
    rec = replace(rec, bounds=SegmentationMaskBounds(-1, -1, 3, 2))
    expected = np.zeros((4, 5), bool)
    expected[:2, :3] = local[1:, 1:]
    np.testing.assert_array_equal(sam_api.paste(rec, 4, 5), expected)


@pytest.mark.parametrize("bounds", [
    (-5, 1, -1, 4), (6, 1, 10, 4),
    (1, -4, 5, -1), (1, 5, 5, 8), (2, 1, 2, 4),
])
def test_paste_disjoint_or_degenerate_box_is_empty(bounds):
    rec = parsed_record(np.ones((3, 4), np.uint8), (0, 0, 4, 3))
    rec = replace(rec, bounds=SegmentationMaskBounds(*bounds))
    actual = sam_api.paste(rec, 4, 5)
    assert actual.shape == (4, 5)
    assert not actual.any()


def test_longer_lived_object_wins_and_missing_frames_stay_empty(monkeypatch):
    short = record("99", [[1]], (0, 0, 1, 1), confidence=0.99)
    long = lambda c: record("7", [[1, 0], [0, 1]], (2, 1, 4, 3), confidence=c)
    # Frame order need not be monotonic; absent frames emit no record.
    response = events([frame(3, long(0.5)), frame(0, short),
                       frame(1, long(0.4)), frame(4, long(0.6))])
    calls = mock_api(monkeypatch, response)
    masks, scores, info = sam_api.segment_video_api(
        "unused.mp4", "billboard", 5, (8, 6)
    )
    assert calls == [("upload", "unused.mp4"), ("stream", "file_test", "billboard")]
    assert masks.shape == (5, 6, 8) and masks.dtype == np.bool_
    assert scores.shape == (5,) and np.issubdtype(scores.dtype, np.floating)
    assert info["chosen"] == "7"
    assert info["objects"]["7"]["frames"] == 3
    assert info["objects"]["99"]["frames"] == 1
    expected = np.zeros_like(masks)
    expected[[1, 3, 4], 1, 2] = True
    expected[[1, 3, 4], 2, 3] = True
    np.testing.assert_array_equal(masks, expected)
    np.testing.assert_allclose(scores, [np.nan, 0.4, np.nan, 0.5, 0.6],
                               equal_nan=True)


def test_revised_record_replaces_prior_mask_and_confidence(monkeypatch):
    old = frame(2, record("7", [[1, 0], [0, 0]], (1, 1, 3, 3), 0.2))
    other = frame(0, record("7", [[1]], (0, 0, 1, 1), 0.6))
    new = frame(2, record("7", [[0, 0], [0, 1]], (1, 1, 3, 3), 0.9),
                newline=False)
    # Split inside the genuine payload, and flush an unterminated final line.
    split = new.index("data=") + len("data=2,2,") + 4
    response = events([old, other, new[:split], new[split:]])
    recs = sam_api.parse_masks(response)
    assert len(recs) == 2
    assert recs[("7", 2)].revision == 2
    assert recs[("7", 2)].confidence == pytest.approx(0.9)
    calls = mock_api(monkeypatch, response)
    masks, scores, info = sam_api.segment_video_api(
        "unused.mp4", "screen", 3, (8, 6), file_id="already_uploaded"
    )
    assert calls == [("stream", "already_uploaded", "screen")]
    assert info["objects"]["7"]["frames"] == 2
    assert info["objects"]["7"]["mean_conf"] == pytest.approx(0.75)
    expected = np.zeros_like(masks)
    expected[0, 0, 0] = True
    expected[2, 2, 2] = True
    np.testing.assert_array_equal(masks, expected)
    np.testing.assert_allclose(scores, [0.6, np.nan, 0.9], equal_nan=True)


def test_out_of_range_records_do_not_win_selection(monkeypatch):
    outside = record("99", [[1]], (0, 0, 1, 1), 0.99)
    inside = record("7", [[1]], (2, 1, 3, 2), 0.5)
    mock_api(monkeypatch, events([
        frame(10, outside), frame(11, outside), frame(12, outside),
        frame(0, inside),
    ]))
    masks, scores, info = sam_api.segment_video_api("unused.mp4", "sign", 2, (8, 6))
    assert info["chosen"] == "7"
    assert "99" not in info["objects"]
    assert masks[0, 1, 2] and not masks[1].any()
    np.testing.assert_allclose(scores, [0.5, np.nan], equal_nan=True)


def test_empty_mask_revision_does_not_count_as_a_sighting(monkeypatch):
    visible = record("99", [[1]], (0, 0, 1, 1), 0.99)
    empty = record("99", [[0]], (0, 0, 1, 1), 0.99)
    actual = record("7", [[1]], (2, 1, 3, 2), 0.4)
    mock_api(monkeypatch, events([
        frame(0, visible), frame(0, empty), frame(1, empty), frame(2, actual),
    ]))
    masks, scores, info = sam_api.segment_video_api("unused.mp4", "sign", 3, (8, 6))
    assert info["chosen"] == "7" and "99" not in info["objects"]
    assert not masks[:2].any()
    np.testing.assert_allclose(scores, [np.nan, np.nan, 0.4], equal_nan=True)


@pytest.mark.parametrize("c7,c9,box9,winner", [
    (0.9, 0.7, (0, 0, 4, 4), "7"),  # confidence wins over area
    (0.7, 0.7, (0, 0, 4, 4), "9"),  # area breaks a confidence tie
])
def test_object_selection_ties(monkeypatch, c7, c9, box9, winner):
    mock_api(monkeypatch, events([frame(
        0, record("7", [[1]], (1, 1, 2, 2), c7),
        record("9", [[1]], box9, c9),
    )]))
    _, _, info = sam_api.segment_video_api("unused.mp4", "sign", 1, (8, 6))
    assert info["chosen"] == winner


def test_completed_empty_lane_returns_empty_arrays(monkeypatch):
    mock_api(monkeypatch, events([]))
    masks, scores, info = sam_api.segment_video_api("unused.mp4", "sign", 3, (8, 6))
    assert masks.shape == (3, 6, 8) and masks.dtype == np.bool_
    assert not masks.any() and np.isnan(scores).all()
    assert info == {"file_id": "file_test", "objects": {}}


def test_missing_confidence_uses_documented_presence_fallback(monkeypatch):
    mock_api(monkeypatch, events([frame(
        0, record("7", [[1]], (1, 1, 2, 2), confidence=None)
    )]))
    masks, scores, _ = sam_api.segment_video_api("unused.mp4", "sign", 2, (8, 6))
    assert masks[0, 1, 1]
    np.testing.assert_allclose(scores, [1.0, np.nan], equal_nan=True)


@pytest.mark.parametrize("terminal", ["eof", "response.incomplete"])
def test_incomplete_stream_is_not_silently_accepted(terminal):
    response = events([frame(0, record("7", [[1]], (1, 1, 2, 2)))])
    if terminal == "eof":
        response.pop()
    else:
        response[-1] = {
            "type": terminal,
            "response": {"incomplete_details": {"reason": "max_output_tokens"}},
        }
    with pytest.raises(RuntimeError, match="incomplete"):
        sam_api.parse_masks(response)


def test_completed_requires_lane_finalizer():
    response = events([frame(0, record("7", [[1]], (1, 1, 2, 2)))],
                      finalizer="none")
    with pytest.raises(ResponsesStreamLaneError):
        sam_api.parse_masks(response)


def test_delta_requires_complete_lane_identity():
    response = events([frame(0, record("7", [[1]], (1, 1, 2, 2)))])
    delta = next(e for e in response if e["type"] == "response.output_text.delta")
    del delta["item_id"]
    with pytest.raises(ResponsesStreamLaneError):
        sam_api.parse_masks(response)


def test_parse_closes_event_generator_on_terminal_event():
    closed = []

    def source():
        try:
            yield from events([])
            raise AssertionError("Parser read beyond response.completed")
        finally:
            closed.append(True)

    assert sam_api.parse_masks(source()) == {}
    assert closed == [True]


@pytest.mark.parametrize("use_event_field", [False, True])
def test_stream_events_reads_sse_and_passes_real_dicts_to_parser(
    monkeypatch, use_event_field
):
    response = events([frame(0, record("7", [[1]], (2, 1, 3, 2)))])
    lines = [": heartbeat", ""]
    for event in response:
        body = dict(event)
        if use_event_field:
            lines.append(f"event: {body.pop('type')}")
        # Multiple data fields form one JSON event, separated by a newline.
        lines.extend("data: " + s for s in json.dumps(body, indent=2).splitlines())
        lines.append("")
    lines.extend(["data: [DONE]", ""])
    calls = []
    closed = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def iter_lines(self, decode_unicode):
            assert decode_unicode is True
            return iter(s.encode("utf-8") for s in lines)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setenv("MODEL_API_KEY", "offline-test-key")
    monkeypatch.setattr(sam_api.requests, "post", post)
    decoded = list(sam_api.stream_events("file_test", "billboard"))
    assert decoded == response
    recs = sam_api.parse_masks(decoded)
    assert set(recs) == {("7", 0)}
    assert closed == [True]
    url, request = calls[0]
    assert url == f"{sam_api.BASE}/responses"
    assert request["json"]["metadata"] == {"include_confidence": "true"}
    assert request["json"]["input"][0]["content"][-1] == {
        "type": "input_video", "file_id": "file_test"
    }
    assert request["stream"] is True
