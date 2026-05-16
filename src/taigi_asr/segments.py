"""Timestamped transcript segment — the canonical data structure passed between engines,
formatters, and the UI layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimestampedSegment:
    """A transcript line with start/end seconds.

    Immutable so formatters can hash/compare safely.

    ``speaker`` is optional and only populated when diarization has been
    attached (see :mod:`taigi_asr.diarize`). When ``speaker is None`` every
    formatter renders the segment in the legacy (pre-diarization) format —
    callers that didn't ask for speaker labels see no behavioural change.
    """

    start_time: float
    end_time: float
    text: str
    speaker: str | None = None

    def with_speaker(self, speaker: str) -> TimestampedSegment:
        """Return a copy with ``speaker`` set; immutability preserved."""
        return TimestampedSegment(
            start_time=self.start_time,
            end_time=self.end_time,
            text=self.text,
            speaker=speaker,
        )

    @staticmethod
    def format_time(seconds: float | None, srt_format: bool = False) -> str:
        """Format seconds as HH:MM:SS (wall clock) or HH:MM:SS,mmm (SRT).

        None / negative are clamped to zero to keep formatters defensive against
        upstream decoder quirks (Whisper sometimes emits None timestamps).

        Carries millisecond overflow into minutes/hours so values like 59.9996
        don't render as an invalid ``HH:MM:60,000``.
        """
        if seconds is None or seconds < 0:
            seconds = 0.0

        if srt_format:
            # Round to the nearest millisecond, then decompose — avoids the
            # float-formatting "60.000" overflow when input is epsilon below 60.
            total_ms = int(round(seconds * 1000))
            ms = total_ms % 1000
            total_s = total_ms // 1000
            s = total_s % 60
            total_m = total_s // 60
            m = total_m % 60
            h = total_m // 60
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        # Wall clock: truncate seconds (matches prior behaviour).
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _text_with_speaker_prefix(self) -> str:
        """Bracket-prefix the text with the speaker when set.

        Returns the bare text when ``speaker is None`` so legacy outputs stay
        byte-for-byte identical.
        """
        if self.speaker is None:
            return self.text
        return f"[{self.speaker}] {self.text}"

    def to_timestamp_line(self) -> str:
        start = self.format_time(self.start_time)
        end = self.format_time(self.end_time)
        return f"[{start} - {end}] {self._text_with_speaker_prefix()}"

    def to_srt_block(self, index: int) -> str:
        start = self.format_time(self.start_time, srt_format=True)
        end = self.format_time(self.end_time, srt_format=True)
        return f"{index}\n{start} --> {end}\n{self._text_with_speaker_prefix()}\n"

    def to_vtt_block(self) -> str:
        start = self.format_time(self.start_time, srt_format=True).replace(",", ".")
        end = self.format_time(self.end_time, srt_format=True).replace(",", ".")
        # WebVTT has a native voice tag (<v Speaker>...) that compliant players
        # render with per-speaker colors. Use it when we have a speaker; fall
        # back to bare text otherwise to preserve legacy output. We escape `<`,
        # `>`, and `&` in the label so an externally-sourced speaker name
        # (e.g. from a hostile RTTM) can't inject markup into the cue.
        if self.speaker is None:
            body = self.text
        else:
            safe_spk = self.speaker.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            body = f"<v {safe_spk}>{self.text}"
        return f"{start} --> {end}\n{body}\n"

    def to_json_dict(self) -> dict[str, float | str | None]:
        out: dict[str, float | str | None] = {
            "start": self.start_time,
            "end": self.end_time,
            "text": self.text,
        }
        if self.speaker is not None:
            out["speaker"] = self.speaker
        return out
