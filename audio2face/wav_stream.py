"""Bounded streaming conversion from uncompressed WAV files to mono float32."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import stat
import struct
from typing import BinaryIO, Iterator

from .path_contract import require_unaliased_path


MAX_CHUNK_FRAMES = 65_536
MAX_CHANNELS = 32
MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 384_000
MAX_DURATION_SECONDS = 6 * 60 * 60
MAX_WAV_FILE_BYTES = 512 * 1024 * 1024
MAX_OUTPUT_FRAMES = MAX_WAV_FILE_BYTES // 4

_READ_BLOCK_FRAMES = 4_096
_SUPPORTED_SAMPLE_WIDTHS = frozenset({1, 2, 3, 4})
_PCM_FORMAT = 0x0001
_IEEE_FLOAT_FORMAT = 0x0003
_EXTENSIBLE_FORMAT = 0xFFFE
_PCM_SUBFORMAT_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
_IEEE_FLOAT_SUBFORMAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


class WavStreamError(ValueError):
    """Raised when a WAV source is unsupported, malformed, or unreadable."""


@dataclass(frozen=True, slots=True)
class WavStreamMetadata:
    """Validated source and output timing known before streaming begins."""

    source_sample_rate: int
    output_sample_rate: int
    channels: int
    bits_per_sample: int
    input_frames: int
    output_frames: int

@dataclass(frozen=True, slots=True)
class _ParsedWav:
    metadata: WavStreamMetadata
    data_offset: int
    block_align: int
    is_float: bool


def _require_bounded_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WavStreamError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise WavStreamError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )
    return value


def _read_exact(handle: BinaryIO, size: int, description: str) -> bytes:
    try:
        payload = handle.read(size)
    except (OSError, ValueError) as exc:
        raise WavStreamError(f"could not read {description}: {exc}") from exc
    if len(payload) != size:
        raise WavStreamError(f"truncated {description}")
    return payload


def _parse_wav(
    handle: BinaryIO,
    *,
    output_sample_rate: int,
) -> _ParsedWav:
    try:
        file_status = os.fstat(handle.fileno())
    except OSError as exc:
        raise WavStreamError(f"could not inspect WAV file: {exc}") from exc
    if not stat.S_ISREG(file_status.st_mode):
        raise WavStreamError("WAV source must be a regular file")
    file_size = file_status.st_size
    if file_size < 12:
        raise WavStreamError("WAV file is too small to contain a RIFF header")
    if file_size > MAX_WAV_FILE_BYTES:
        raise WavStreamError(
            f"WAV file exceeds the {MAX_WAV_FILE_BYTES}-byte safety limit"
        )

    handle.seek(0)
    riff_header = _read_exact(handle, 12, "RIFF header")
    if riff_header[:4] != b"RIFF" or riff_header[8:] != b"WAVE":
        raise WavStreamError("source must be a little-endian RIFF/WAVE file")
    riff_size = struct.unpack_from("<I", riff_header, 4)[0]
    riff_end = riff_size + 8
    if riff_end != file_size:
        raise WavStreamError("RIFF size does not match the WAV file size")

    fmt_payload: bytes | None = None
    fmt_chunk_size: int | None = None
    data_offset: int | None = None
    data_size: int | None = None
    position = 12
    while position < riff_end:
        if riff_end - position < 8:
            raise WavStreamError("truncated WAV chunk header")
        handle.seek(position)
        chunk_header = _read_exact(handle, 8, "WAV chunk header")
        chunk_id = chunk_header[:4]
        chunk_size = struct.unpack_from("<I", chunk_header, 4)[0]
        chunk_offset = position + 8
        chunk_end = chunk_offset + chunk_size
        padded_end = chunk_end + (chunk_size & 1)
        if padded_end > riff_end:
            raise WavStreamError("WAV chunk extends beyond the RIFF container")

        if chunk_id == b"fmt ":
            if fmt_payload is not None:
                raise WavStreamError("WAV contains more than one format chunk")
            if chunk_size < 16:
                raise WavStreamError("WAV format chunk is shorter than 16 bytes")
            handle.seek(chunk_offset)
            fmt_payload = _read_exact(handle, min(chunk_size, 40), "WAV format chunk")
            fmt_chunk_size = chunk_size
        elif chunk_id == b"data":
            if data_offset is not None:
                raise WavStreamError("WAV contains more than one data chunk")
            data_offset = chunk_offset
            data_size = chunk_size

        position = padded_end

    if fmt_payload is None or fmt_chunk_size is None:
        raise WavStreamError("WAV is missing its format chunk")
    if data_offset is None or data_size is None:
        raise WavStreamError("WAV is missing its data chunk")

    (
        audio_format,
        channels,
        source_sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_payload)
    is_float = False
    if audio_format == _EXTENSIBLE_FORMAT:
        if len(fmt_payload) < 40:
            raise WavStreamError("extensible WAV format chunk is shorter than 40 bytes")
        extension_size, valid_bits = struct.unpack_from("<HH", fmt_payload, 16)
        if extension_size < 22 or extension_size + 18 > fmt_chunk_size:
            raise WavStreamError("extensible WAV format extension has an invalid size")
        subformat = fmt_payload[24:40]
        if subformat == _IEEE_FLOAT_SUBFORMAT_GUID:
            is_float = True
        elif subformat != _PCM_SUBFORMAT_GUID:
            raise WavStreamError("extensible WAV has an unsupported audio subformat")
        if valid_bits != bits_per_sample:
            raise WavStreamError(
                "extensible WAV valid bits must equal its sample container width"
            )
    elif audio_format == _IEEE_FLOAT_FORMAT:
        is_float = True
    elif audio_format != _PCM_FORMAT:
        raise WavStreamError("WAV must contain integer PCM or IEEE float32 audio")
    if not 1 <= channels <= MAX_CHANNELS:
        raise WavStreamError(
            f"WAV channel count must be between 1 and {MAX_CHANNELS}, got {channels}"
        )
    if not MIN_SAMPLE_RATE <= source_sample_rate <= MAX_SAMPLE_RATE:
        raise WavStreamError(
            "WAV sample rate must be between "
            f"{MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}, got {source_sample_rate}"
        )
    if bits_per_sample % 8 != 0:
        raise WavStreamError("WAV bits per sample must be byte-aligned")
    sample_width = bits_per_sample // 8
    if is_float and bits_per_sample != 32:
        raise WavStreamError("IEEE float WAV samples must be exactly 32 bits")
    if not is_float and sample_width not in _SUPPORTED_SAMPLE_WIDTHS:
        raise WavStreamError("WAV sample width must be 8, 16, 24, or 32 bits")

    expected_block_align = channels * sample_width
    if block_align != expected_block_align:
        raise WavStreamError("WAV block alignment does not match its sample format")
    if byte_rate != source_sample_rate * block_align:
        raise WavStreamError("WAV byte rate does not match its sample format")
    if data_size == 0:
        raise WavStreamError("WAV data chunk is empty")
    if data_size % block_align != 0:
        raise WavStreamError("WAV data size is not a whole number of sample frames")

    input_frames = data_size // block_align
    if input_frames > source_sample_rate * MAX_DURATION_SECONDS:
        raise WavStreamError(
            f"WAV duration exceeds the {MAX_DURATION_SECONDS}-second safety limit"
        )
    output_frames = max(1, input_frames * output_sample_rate // source_sample_rate)
    if output_frames > MAX_OUTPUT_FRAMES:
        raise WavStreamError(
            "resampled WAV exceeds the 512 MiB decoded-audio safety limit"
        )

    return _ParsedWav(
        metadata=WavStreamMetadata(
            source_sample_rate=source_sample_rate,
            output_sample_rate=output_sample_rate,
            channels=channels,
            bits_per_sample=bits_per_sample,
            input_frames=input_frames,
            output_frames=output_frames,
        ),
        data_offset=data_offset,
        block_align=block_align,
        is_float=is_float,
    )


def _decode_mono(
    payload: bytes,
    *,
    channels: int,
    sample_width: int,
    is_float: bool,
) -> list[float]:
    sample_count = len(payload) // sample_width

    if is_float:
        decoded_float = struct.unpack(f"<{sample_count}f", payload)
        if not all(math.isfinite(value) for value in decoded_float):
            raise WavStreamError("IEEE float WAV contains a non-finite sample")
        decoded_float = tuple(max(-1.0, min(1.0, value)) for value in decoded_float)
        if channels == 1:
            return list(decoded_float)

        mono_float: list[float] = []
        append_float = mono_float.append
        for offset in range(0, sample_count, channels):
            append_float(
                sum(decoded_float[offset : offset + channels]) / channels
            )
        return list(
            struct.unpack(
                f"<{len(mono_float)}f",
                struct.pack(f"<{len(mono_float)}f", *mono_float),
            )
        )

    if sample_width == 1:
        integer_samples = [value - 128 for value in payload]
        scale = 128.0
    elif sample_width == 2:
        integer_samples = struct.unpack(f"<{sample_count}h", payload)
        scale = 32_768.0
    elif sample_width == 3:
        integer_samples = []
        append = integer_samples.append
        for offset in range(0, len(payload), 3):
            value = (
                payload[offset]
                | (payload[offset + 1] << 8)
                | (payload[offset + 2] << 16)
            )
            if value & 0x80_0000:
                value -= 0x100_0000
            append(value)
        scale = 8_388_608.0
    else:
        integer_samples = struct.unpack(f"<{sample_count}i", payload)
        scale = 2_147_483_648.0

    if channels == 1:
        mono = [value / scale for value in integer_samples]
    else:
        channel_scale = scale * channels
        mono = []
        append_mono = mono.append
        for offset in range(0, sample_count, channels):
            total = 0
            for channel in range(channels):
                total += integer_samples[offset + channel]
            append_mono(total / channel_scale)
    return list(
        struct.unpack(
            f"<{len(mono)}f",
            struct.pack(f"<{len(mono)}f", *mono),
        )
    )


def _pack_f32le(samples: list[float]) -> bytes:
    return struct.pack(f"<{len(samples)}f", *samples)


class WavStreamSource:
    """One-shot iterator producing bounded mono f32le chunks from a WAV."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        output_sample_rate: int,
        chunk_frames: int,
    ) -> None:
        output_sample_rate = _require_bounded_int(
            output_sample_rate,
            name="output_sample_rate",
            minimum=MIN_SAMPLE_RATE,
            maximum=MAX_SAMPLE_RATE,
        )
        self.chunk_frames = _require_bounded_int(
            chunk_frames,
            name="chunk_frames",
            minimum=1,
            maximum=MAX_CHUNK_FRAMES,
        )
        source_path = require_unaliased_path(
            path,
            description="WAV path",
            error_type=WavStreamError,
        )

        try:
            handle = source_path.open("rb", buffering=0)
        except OSError as exc:
            raise WavStreamError(f"could not open WAV file: {exc}") from exc
        self._handle: BinaryIO | None = handle
        self._iteration_started = False
        try:
            parsed = _parse_wav(
                handle,
                output_sample_rate=output_sample_rate,
            )
        except Exception:
            handle.close()
            self._handle = None
            raise

        self.metadata = parsed.metadata
        self._data_offset = parsed.data_offset
        self._block_align = parsed.block_align
        self._is_float = parsed.is_float

    def close(self) -> None:
        """Close the source. Calling this more than once is safe."""

        handle = self._handle
        self._handle = None
        if handle is not None:
            handle.close()

    def __enter__(self) -> WavStreamSource:
        if self._handle is None:
            raise WavStreamError("WAV source is closed")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def _mono_frames(self) -> Iterator[float]:
        handle = self._handle
        if handle is None:
            raise WavStreamError("WAV source is closed")
        try:
            handle.seek(self._data_offset)
        except (OSError, ValueError) as exc:
            raise WavStreamError(f"could not seek to WAV audio data: {exc}") from exc

        remaining_frames = self.metadata.input_frames
        sample_width = self.metadata.bits_per_sample // 8
        while remaining_frames:
            if self._handle is None:
                raise WavStreamError("WAV source was closed while streaming")
            block_frames = min(_READ_BLOCK_FRAMES, remaining_frames)
            byte_count = block_frames * self._block_align
            payload = _read_exact(handle, byte_count, "WAV audio data")
            yield from _decode_mono(
                payload,
                channels=self.metadata.channels,
                sample_width=sample_width,
                is_float=self._is_float,
            )
            remaining_frames -= block_frames

    def __iter__(self) -> Iterator[bytes]:
        """Yield each f32le audio chunk once, then close the source."""

        if self._handle is None:
            raise WavStreamError("WAV source is closed")
        if self._iteration_started:
            raise WavStreamError("WAV source can only be iterated once")
        self._iteration_started = True

        def chunks() -> Iterator[bytes]:
            try:
                mono_frames = iter(self._mono_frames())
                previous = next(mono_frames)

                source_rate = self.metadata.source_sample_rate
                output_rate = self.metadata.output_sample_rate
                output_count = self.metadata.output_frames
                output_index = 0
                source_index = 0
                pending: list[float] = []

                for current in mono_frames:
                    next_source_numerator = (source_index + 1) * output_rate
                    while (
                        output_index < output_count
                        and output_index * source_rate < next_source_numerator
                    ):
                        fraction_numerator = (
                            output_index * source_rate
                            - source_index * output_rate
                        )
                        fraction = fraction_numerator / output_rate
                        pending.append(previous + (current - previous) * fraction)
                        output_index += 1
                        if len(pending) == self.chunk_frames:
                            yield _pack_f32le(pending)
                            pending.clear()
                    previous = current
                    source_index += 1

                while output_index < output_count:
                    pending.append(previous)
                    output_index += 1
                    if len(pending) == self.chunk_frames:
                        yield _pack_f32le(pending)
                        pending.clear()
                if pending:
                    yield _pack_f32le(pending)
            finally:
                self.close()

        return chunks()
