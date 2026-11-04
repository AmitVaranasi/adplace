from scripts.ingest import failure_message, plain


def test_the_gate_named_first_wins():
    # "logo too small: ... px tall on screen" is about the logo's size, not time on screen
    assert plain(["logo too small: the logo would be 13 px tall on screen, needs 28 px"]) == [
        "It's too small on screen for a readable logo."]
    assert plain(["on screen 0.79 s, needs at least 2.0 s"]) == ["It isn't on screen long enough to register."]
    assert plain(["failed final check: judge_vs_original 0.6"]) == [
        "A blind reviewer spotted it as inserted much more often than the original."]


def test_failures_read_as_sentences():
    import subprocess
    e = subprocess.CalledProcessError(1, ["ffmpeg", "-i", "/Users/someone/x.mp4"])
    msg = failure_message(e)
    assert "ffmpeg" not in msg and "/Users" not in msg and msg.startswith("We couldn't read")
