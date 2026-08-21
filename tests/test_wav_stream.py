from __future__ import annotations

import struct
from pathlib import Path

import pytest

from audio2face import wav_stream
from audio2face.wav_stream import (
    MAX_CHANNELS,
    MAX_CHUNK_FRAMES,
    WavStreamError,
    WavStreamSource,
)


def _pcm_bytes(sample_width: int, samples: list[int]) -> bytes:
    if sample_width == 1:
        return bytes(value + 128 for value in samples)
    return b"".join(
        value.to_bytes(sample_width, "little", signed=True) for value in samples
    )


def _wav_bytes(
    *,
    sample_rate: int,
    channels: int,
    sample_width: int,
    samples: list[int],
    audio_format: int = 1,
    declared_data_size: int | None = None,
    block_align: int | None = None,
    byte_rate: int | None = None,
) -> bytes:
    pcm = _pcm_bytes(sample_width, samples)
    data_size = len(pcm) if declared_data_size is None else declared_data_size
    alignment = channels * sample_width if block_align is None else block_align
    rate = sample_rate * alignment if byte_rate is None else byte_rate
    fmt = struct.pack(
        "<HHIIHH",
        audio_format,
        channels,
        sample_rate,
        rate,
        alignment,
        sample_width * 8,
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", data_size) + pcm
    if data_size & 1:
        chunks += b"\x00"
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def _float_wav_bytes(
    *,
    samples: list[float],
    channels: int = 1,
    sample_rate: int = 16_000,
    bits_per_sample: int = 32,
    extensible: bool = False,
    subformat: bytes | None = None,
    valid_bits: int | None = None,
) -> bytes:
    sample_width = bits_per_sample // 8
    if bits_per_sample == 32:
        pcm = struct.pack(f"<{len(samples)}f", *samples)
    elif bits_per_sample == 64:
        pcm = struct.pack(f"<{len(samples)}d", *samples)
    else:
        pcm = bytes(sample_width * len(samples))
    block_align = channels * sample_width
    if extensible:
        guid = subformat or bytes.fromhex("0300000000001000800000aa00389b71")
        fmt = struct.pack(
            "<HHIIHHH",
            0xFFFE,
            channels,
            sample_rate,
            sample_rate * block_align,
            block_align,
            bits_per_sample,
            22,
        )
        valid_width = bits_per_sample if valid_bits is None else valid_bits
        fmt += struct.pack("<HI", valid_width, 0) + guid
    else:
        fmt = struct.pack(
            "<HHIIHH",
            3,
            channels,
            sample_rate,
            sample_rate * block_align,
            block_align,
            bits_per_sample,
        )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(pcm)) + pcm
    if len(pcm) & 1:
        chunks += b"\x00"
    return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks


def _write_wav(path: Path, **kwargs: object) -> Path:
    path.write_bytes(_wav_bytes(**kwargs))
    return path


def _unpack_chunks(chunks: list[bytes]) -> tuple[float, ...]:
    payload = b"".join(chunks)
    return struct.unpack(f"<{len(payload) // 4}f", payload)


def test_metadata_chunks_and_context_lifecycle(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "voice.wav",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        samples=[-32_768, -16_384, 0, 16_384, 32_767],
    )

    with WavStreamSource(path, chunk_frames=2) as source:
        assert source.metadata.source_sample_rate == 16_000
        assert source.metadata.output_sample_rate == 16_000
        assert source.metadata.channels == 1
        assert source.metadata.bits_per_sample == 16
        assert source.metadata.sample_width_bytes == 2
        assert source.metadata.input_frames == 5
        assert source.metadata.output_frames == 5
        assert source.metadata.duration_seconds == 5 / 16_000
        chunks = list(source)

    assert source.closed
    assert all(chunk and len(chunk) <= 2 * 4 for chunk in chunks)
    assert all(len(chunk) % 4 == 0 for chunk in chunks)
    assert _unpack_chunks(chunks) == pytest.approx(
        (-1.0, -0.5, 0.0, 0.5, 32_767 / 32_768)
    )


@pytest.mark.parametrize(
    ("sample_width", "scale"),
    [(1, 128), (2, 32_768), (3, 8_388_608), (4, 2_147_483_648)],
)
def test_all_integer_pcm_widths_downmix_to_finite_mono(
    tmp_path: Path,
    sample_width: int,
    scale: int,
) -> None:
    path = _write_wav(
        tmp_path / f"stereo-{sample_width}.wav",
        sample_rate=16_000,
        channels=2,
        sample_width=sample_width,
        samples=[
            -scale // 2,
            scale // 2,
            -scale,
            0,
            scale // 4,
            scale // 4,
        ],
    )

    with WavStreamSource(path) as source:
        samples = _unpack_chunks(list(source))

    assert samples == pytest.approx((0.0, -0.5, 0.25))


def test_extensible_integer_pcm_is_supported_without_relaxing_subformat(
    tmp_path: Path,
) -> None:
    samples = _pcm_bytes(3, [-4_194_304, 4_194_304])
    pcm_guid = bytes.fromhex("0100000000001000800000aa00389b71")
    fmt = struct.pack("<HHIIHHH", 0xFFFE, 1, 16_000, 48_000, 3, 24, 22)
    fmt += struct.pack("<HI", 24, 0) + pcm_guid
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(samples)) + samples
    payload = b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks
    path = tmp_path / "extensible.wav"
    path.write_bytes(payload)

    with WavStreamSource(path) as source:
        assert _unpack_chunks(list(source)) == pytest.approx((-0.5, 0.5))

    unknown_guid = bytes.fromhex("0700000000001000800000aa00389b71")
    path.write_bytes(payload.replace(pcm_guid, unknown_guid))
    with pytest.raises(WavStreamError, match="unsupported audio subformat"):
        WavStreamSource(path)


@pytest.mark.parametrize("extensible", [False, True])
def test_ieee_float32_wav_downmixes_and_resamples(
    tmp_path: Path,
    extensible: bool,
) -> None:
    path = tmp_path / f"float-{extensible}.wav"
    path.write_bytes(
        _float_wav_bytes(
            samples=[-0.75, 0.25, 0.5, 1.0, -0.25, -0.75],
            channels=2,
            sample_rate=8_000,
            extensible=extensible,
        )
    )

    with WavStreamSource(
        path,
        output_sample_rate=16_000,
        chunk_frames=2,
    ) as source:
        assert source.metadata.bits_per_sample == 32
        assert source.metadata.input_frames == 3
        assert source.metadata.output_frames == 6
        chunks = list(source)

    assert all(0 < len(chunk) <= 8 for chunk in chunks)
    assert _unpack_chunks(chunks) == pytest.approx(
        (-0.25, 0.25, 0.75, 0.125, -0.5, -0.5)
    )


@pytest.mark.parametrize("extensible", [False, True])
@pytest.mark.parametrize("bad_sample", [float("nan"), float("inf"), -float("inf")])
def test_ieee_float32_wav_rejects_every_non_finite_sample(
    tmp_path: Path,
    extensible: bool,
    bad_sample: float,
) -> None:
    path = tmp_path / "non-finite.wav"
    path.write_bytes(
        _float_wav_bytes(samples=[0.0, bad_sample], extensible=extensible)
    )
    source = WavStreamSource(path)

    with pytest.raises(WavStreamError, match="non-finite sample"):
        list(source)
    assert source.closed


@pytest.mark.parametrize("bits_per_sample", [16, 24, 64])
@pytest.mark.parametrize("extensible", [False, True])
def test_ieee_float_wav_rejects_every_non_32_bit_width(
    tmp_path: Path,
    bits_per_sample: int,
    extensible: bool,
) -> None:
    path = tmp_path / "wrong-float-width.wav"
    path.write_bytes(
        _float_wav_bytes(
            samples=[0.0],
            bits_per_sample=bits_per_sample,
            extensible=extensible,
        )
    )

    with pytest.raises(WavStreamError, match="exactly 32 bits"):
        WavStreamSource(path)


def test_extensible_float32_rejects_a_mismatched_valid_width(
    tmp_path: Path,
) -> None:
    path = tmp_path / "wrong-valid-width.wav"
    path.write_bytes(
        _float_wav_bytes(
            samples=[0.0],
            extensible=True,
            valid_bits=24,
        )
    )

    with pytest.raises(WavStreamError, match="valid bits"):
        WavStreamSource(path)


def test_resampling_keeps_phase_across_read_and_output_chunk_boundaries(
    tmp_path: Path,
) -> None:
    integers = [((index % 101) - 50) * 512 for index in range(4_102)]
    path = _write_wav(
        tmp_path / "upsample.wav",
        sample_rate=8_000,
        channels=1,
        sample_width=2,
        samples=integers,
    )

    with WavStreamSource(
        path,
        output_sample_rate=16_000,
        chunk_frames=7,
    ) as source:
        assert source.metadata.output_frames == 8_204
        chunks = list(source)
    actual = _unpack_chunks(chunks)
    decoded = [value / 32_768 for value in integers]

    assert len(actual) == 8_204
    for output_index in (0, 1, 13, 8_189, 8_190, 8_191, 8_201, 8_203):
        source_position = output_index / 2
        left = min(int(source_position), len(decoded) - 1)
        right = min(left + 1, len(decoded) - 1)
        fraction = source_position - left
        expected = decoded[left] + (decoded[right] - decoded[left]) * fraction
        assert actual[output_index] == pytest.approx(expected)
    assert all(0 < len(chunk) <= 7 * 4 for chunk in chunks)


def test_resampling_has_a_predictable_rounded_output_count(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "downsample.wav",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        samples=[0, 1_000, 2_000, 3_000, 4_000],
    )

    with WavStreamSource(path, output_sample_rate=8_000) as source:
        assert source.metadata.output_frames == 3
        actual = _unpack_chunks(list(source))

    assert actual == pytest.approx((0, 2_000 / 32_768, 4_000 / 32_768))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"audio_format": 6}, "integer PCM or IEEE float32"),
        ({"channels": MAX_CHANNELS + 1}, "channel count"),
        ({"sample_rate": 1}, "sample rate"),
        ({"sample_width": 5}, "sample width"),
        ({"block_align": 1}, "block alignment"),
        ({"byte_rate": 1}, "byte rate"),
    ],
)
def test_rejects_unsupported_or_inconsistent_pcm_headers(
    tmp_path: Path,
    changes: dict[str, int],
    message: str,
) -> None:
    arguments = {
        "sample_rate": 16_000,
        "channels": 1,
        "sample_width": 2,
        "samples": [0],
        **changes,
    }
    path = _write_wav(tmp_path / "invalid.wav", **arguments)

    with pytest.raises(WavStreamError, match=message):
        WavStreamSource(path)


def test_rejects_non_frame_aligned_and_structurally_malformed_files(
    tmp_path: Path,
) -> None:
    unaligned = _write_wav(
        tmp_path / "unaligned.wav",
        sample_rate=16_000,
        channels=2,
        sample_width=2,
        samples=[0],
    )
    with pytest.raises(WavStreamError, match="whole number of sample frames"):
        WavStreamSource(unaligned)

    wrong_riff_size = tmp_path / "wrong-size.wav"
    payload = bytearray(
        _wav_bytes(
            sample_rate=16_000,
            channels=1,
            sample_width=2,
            samples=[0],
        )
    )
    struct.pack_into("<I", payload, 4, len(payload) + 100)
    wrong_riff_size.write_bytes(payload)
    with pytest.raises(WavStreamError, match="RIFF size"):
        WavStreamSource(wrong_riff_size)

    truncated_chunk = _write_wav(
        tmp_path / "truncated.wav",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        samples=[0],
        declared_data_size=100,
    )
    with pytest.raises(WavStreamError, match="extends beyond"):
        WavStreamSource(truncated_chunk)


def test_safety_limits_are_enforced_without_large_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_wav(
        tmp_path / "bounded.wav",
        sample_rate=8_000,
        channels=1,
        sample_width=1,
        samples=[0] * 16,
    )

    monkeypatch.setattr(wav_stream, "MAX_WAV_FILE_BYTES", path.stat().st_size - 1)
    with pytest.raises(WavStreamError, match="safety limit"):
        WavStreamSource(path)

    monkeypatch.setattr(wav_stream, "MAX_WAV_FILE_BYTES", 1_000_000)
    monkeypatch.setattr(wav_stream, "MAX_DURATION_SECONDS", 0)
    with pytest.raises(WavStreamError, match="duration exceeds"):
        WavStreamSource(path)


def test_constructor_and_iterator_limits_are_strict(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "voice.wav",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        samples=[0],
    )

    with pytest.raises(WavStreamError, match="output_sample_rate must be an integer"):
        WavStreamSource(path, output_sample_rate=True)
    with pytest.raises(WavStreamError, match="chunk_frames must be between"):
        WavStreamSource(path, chunk_frames=MAX_CHUNK_FRAMES + 1)

    source = WavStreamSource(path)
    assert len(list(source)) == 1
    assert source.closed
    with pytest.raises(WavStreamError, match="closed"):
        iter(source)


def test_truncation_after_open_is_detected_during_streaming(tmp_path: Path) -> None:
    path = _write_wav(
        tmp_path / "mutable.wav",
        sample_rate=16_000,
        channels=1,
        sample_width=2,
        samples=[0] * 32,
    )
    source = WavStreamSource(path, chunk_frames=4)
    path.write_bytes(path.read_bytes()[:-16])

    with pytest.raises(WavStreamError, match="truncated WAV audio data"):
        list(source)
    assert source.closed
