"""Windows audio through WASAPI (sounddevice/PortAudio). Same interface as audio_linux.py.

Windows can't create a microphone by itself: DECtalk is played into VB-Audio Virtual Cable's "CABLE Input"
(installed separately by the user), and Discord, OBS etc. use "CABLE Output" as the microphone. Without the
cable, DECtalk plays on the speakers. Sample rates and channel counts are converted by Windows
(WASAPI auto-convert), so the mic is opened at 16 kHz mono and DECtalk is played at 11025 Hz mono.
"""
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from dectalk import RATE as TTS_RATE

CABLE = "CABLE Input"                     # VB-Audio Virtual Cable's playback side
DESCRIPTION = "CABLE Output (VB-Audio Virtual Cable)"
MIC_RATE = 16000
MIC_BLOCK = 320                           # 20 ms blocks


def _wasapi():
    for i, h in enumerate(sd.query_hostapis()):
        if "WASAPI" in h["name"]:
            return i, h
    raise RuntimeError("Windows audio (WASAPI) is not available.")


def _devices(kind):
    """[(index, device)] of WASAPI input or output devices."""
    api, _ = _wasapi()
    return [(i, d) for i, d in enumerate(sd.query_devices())
            if d["hostapi"] == api and d[f"max_{kind}_channels"] > 0]


def _find(kind, name):
    return next((i for i, d in _devices(kind) if d["name"] == name), None)


def _settings():
    return sd.WasapiSettings(auto_convert=True)


def list_mics():
    # the cable's recording side carries DECtalk itself: listening to it would loop
    return [dict(name=d["name"], description=d["name"]) for _, d in _devices("input")
            if not d["name"].startswith("CABLE Output")]


def default_mic():
    mics = list_mics()
    i = _wasapi()[1]["default_input_device"]
    if i >= 0 and any(m["name"] == sd.query_devices(i)["name"] for m in mics):
        return sd.query_devices(i)["name"]
    return mics[0]["name"] if mics else ""


def list_outputs():
    return [dict(name=d["name"], description=d["name"]) for _, d in _devices("output")]


class _Player:
    """Plays queued clips on one output device. A clip's `on_start` gets the time its first sample is heard."""

    def __init__(self, index):
        self._q, self._lock = deque(), threading.Lock()      # [samples, on_start, position]
        self.stream = sd.OutputStream(device=index, samplerate=TTS_RATE, channels=1, dtype="float32",
                                      latency="low", callback=self._fill, extra_settings=_settings())
        self.stream.start()

    def _fill(self, out, frames, t, status):
        out.fill(0)
        delay = t.outputBufferDacTime - t.currentTime if t.outputBufferDacTime else self.stream.latency
        n = 0
        with self._lock:
            while n < frames and self._q:
                clip = self._q[0]
                if clip[2] == 0 and clip[1]:
                    clip[1](time.monotonic() + max(0.0, delay) + n / TTS_RATE)
                take = min(frames - n, len(clip[0]) - clip[2])
                out[n:n + take, 0] = clip[0][clip[2]:clip[2] + take]
                clip[2] += take
                n += take
                if clip[2] >= len(clip[0]):
                    self._q.popleft()

    def play(self, pcm, on_start=None):
        with self._lock:
            self._q.append([pcm.astype(np.float32) / 32768, on_start, 0])

    def stop(self):
        with self._lock:
            self._q.clear()

    def backlog(self):
        with self._lock:
            left = sum(len(c[0]) - c[2] for c in self._q)
        return left / TTS_RATE + self.stream.latency

    def close(self):
        self.stream.stop()
        self.stream.close()


class Output:
    """DECtalk's output: the virtual cable if installed (or the chosen device), plus an optional monitor on
    the default speakers ("Hear it myself")."""

    def __init__(self, device=""):
        self.main = self.monitor = None
        self.device = ""
        self.set_device(device)

    def set_device(self, name):
        index = _find("output", name) if name else None
        if index is None:
            cable = [i for i, d in _devices("output") if d["name"].startswith(CABLE)]
            index = cable[0] if cable else _wasapi()[1]["default_output_device"]
        old, self.main = self.main, _Player(index)
        self.device = sd.query_devices(index)["name"]
        if old:
            old.close()

    @property
    def hint(self):
        if self.device.startswith(CABLE):
            return f'Pick "{DESCRIPTION}" as the microphone in Discord, OBS or a game.'
        return (f"Playing on {self.device}. To use Moonbase as a microphone, install VB-Audio Virtual Cable "
                "(vb-audio.com/Cable) and pick \"CABLE Input\" as the output.")

    def play(self, pcm, on_start=None):
        self.main.play(pcm, on_start)
        if self.monitor:
            self.monitor.play(pcm)

    def stop(self):
        self.main.stop()
        if self.monitor:
            self.monitor.stop()

    def backlog(self):
        return self.main.backlog()

    def set_monitor(self, on):
        if on and not self.monitor:
            index = _wasapi()[1]["default_output_device"]
            if index >= 0 and sd.query_devices(index)["name"] != self.device:
                self.monitor = _Player(index)
        elif not on and self.monitor:
            self.monitor.close()
            self.monitor = None

    def close(self):
        self.set_monitor(False)
        self.main.close()


class Mic:
    """The microphone as 16 kHz mono float32, delivered in 20 ms blocks."""

    def __init__(self, sink):
        self.sink, self.stream, self.name = sink, None, ""

    def start(self, name):
        self.stop()
        self.name = name
        index = _find("input", name) if name else None
        if index is None:
            return                         # no microphone
        self.stream = sd.InputStream(device=index, samplerate=MIC_RATE, channels=1, dtype="float32",
                                     blocksize=MIC_BLOCK, latency="low", extra_settings=_settings(),
                                     callback=lambda data, frames, t, status: self.sink(data[:, 0].copy()))
        self.stream.start()

    def stop(self):
        if self.stream:
            s, self.stream = self.stream, None
            s.stop()
            s.close()
