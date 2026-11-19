"""Linux audio through PipeWire: the mic is read with pw-record, and DECtalk comes out of a module-pipe-source
virtual microphone, "Moonbase Alpha (DECtalk)". Same interface as audio_windows.py."""
import fcntl
import json
import os
import subprocess
import threading
import time
from collections import deque

import numpy as np

from dectalk import RATE as TTS_RATE

SOURCE = "moonbase_mic"
DESCRIPTION = "Moonbase Alpha (DECtalk)"
HINT = f'Pick "{DESCRIPTION}" as the microphone in Discord, OBS or a game.'
MIC_RATE = 16000
MIC_BLOCK = 320                           # 20 ms reads from pw-record
LEAD = 0.06                               # how far ahead of real time audio is written to the virtual mic


def list_outputs():
    """The output is always the virtual microphone: nothing to choose."""
    return []


def pactl(*args):
    return subprocess.run(["pactl", *args], check=True, capture_output=True, text=True).stdout.strip()


def list_mics():
    out = []
    for s in json.loads(pactl("-f", "json", "list", "sources")):
        if s["name"] == SOURCE or s.get("properties", {}).get("device.class") == "monitor":
            continue
        out.append(dict(name=s["name"], description=s.get("description") or s["name"]))
    return out


def default_mic():
    try:
        name = pactl("get-default-source")
    except subprocess.CalledProcessError:
        name = ""
    mics = list_mics()
    if any(m["name"] == name for m in mics):
        return name
    return mics[0]["name"] if mics else ""


class Output:
    """The virtual microphone: a PipeWire source fed through a FIFO, written in real time; optional loopback
    to the speakers."""

    hint = HINT
    device = DESCRIPTION

    def __init__(self, device=""):        # `device`: Windows picks an output; here it's always this mic
        self._unload_stale()
        run = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
        self.fifo = os.path.join(run, f"{SOURCE}.fifo")
        self.fd = self.loopback = None
        self._io = threading.Lock()       # guards fd: closing, reopening and writing
        self._closed, self._retry_at = False, 0.0
        self._open()
        self._q, self._cv, self._cut = deque(), threading.Condition(), False
        self._end = 0.0                   # monotonic time at which everything written so far has played
        self._left = 0                    # samples of the clause being played not written yet
        threading.Thread(target=self._writer, daemon=True).start()

    def _open(self):
        pactl("load-module", "module-pipe-source", f"source_name={SOURCE}", f"file={self.fifo}",
              "format=s16le", f"rate={TTS_RATE}", "channels=1",
              f'source_properties=device.description="{DESCRIPTION}"')
        # non-blocking with a one-page pipe: when nothing records the virtual mic, audio is dropped
        # instead of piling up and playing late
        self.fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        fcntl.fcntl(self.fd, fcntl.F_SETPIPE_SZ, 4096)

    def _reopen(self):
        """The device went away (PipeWire restarted, or it was unloaded): make it again. Hold self._io."""
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        monitor, self.loopback = self.loopback is not None, None
        try:
            self._unload_stale()          # by name: module ids from before a PipeWire restart mean nothing
            self._open()
            if monitor:
                self.set_monitor(True)
        except (OSError, subprocess.CalledProcessError):
            self._retry_at = time.monotonic() + 1.0     # PipeWire not back yet

    @staticmethod
    def _unload_stale():
        # only pulse modules' lines start "<index>\t<name>\t<args>"; PipeWire's own modules have multi-line args
        for line in pactl("list", "modules", "short").splitlines():
            p = line.split("\t")
            if len(p) >= 3 and p[0].isdigit() and p[1] in ("module-pipe-source", "module-loopback") \
                    and SOURCE in p[2]:
                _unload_quietly(p[0])

    def play(self, pcm, on_start=None):
        with self._cv:
            self._q.append((pcm, on_start))
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._q.clear()
            self._cut = True

    def backlog(self):
        """Seconds of speech still to be heard: queued, the rest of the clause being played, and what is
        written but not yet played."""
        with self._cv:
            queued = (sum(len(p) for p, _ in self._q) + self._left) / TTS_RATE
        return queued + max(0.0, self._end - time.monotonic())

    def _writer(self):
        chunk = TTS_RATE // 100
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait()
                if self._closed:
                    return
                pcm, on_start = self._q.popleft()
                self._cut = False
                self._left = len(pcm)
            for i in range(0, len(pcm), chunk):
                if self._cut or self._closed:
                    break
                now = time.monotonic()
                self._end = max(self._end, now)
                if self._end - now > LEAD:
                    time.sleep(self._end - now - LEAD)
                if i == 0 and on_start:
                    on_start(self._end)   # when this phrase's first sample will be heard
                part = pcm[i:i + chunk]
                with self._io:
                    if self._closed:
                        return
                    if self.fd is None and time.monotonic() >= self._retry_at:
                        self._reopen()
                    if self.fd is not None:
                        try:
                            os.write(self.fd, part.tobytes())   # 220 bytes: below PIPE_BUF, all or nothing
                        except BlockingIOError:
                            pass
                        except OSError:
                            self._reopen()
                self._end += len(part) / TTS_RATE
                self._left = max(0, len(pcm) - i - chunk)
            self._left = 0

    def set_monitor(self, on):
        if on and not self.loopback:
            self.loopback = pactl("load-module", "module-loopback", f"source={SOURCE}", "latency_msec=30",
                                  "source_dont_move=true")
        elif not on and self.loopback:
            _unload_quietly(self.loopback)
            self.loopback = None

    def close(self):
        with self._io:                    # after this the writer can neither write nor reopen
            self._closed = True
            if self.fd is not None:
                try:
                    os.close(self.fd)
                except OSError:
                    pass
                self.fd = None
        with self._cv:
            self._cv.notify()
        self.loopback = None
        self._unload_stale()


def _unload_quietly(module):
    if module:
        try:
            pactl("unload-module", module)
        except subprocess.CalledProcessError:
            pass                          # already gone


class Mic:
    """pw-record from a PipeWire source as 16 kHz mono int16, delivered in 40 ms blocks."""

    def __init__(self, sink):
        self.sink, self.proc, self.name = sink, None, ""

    def start(self, name):
        self.stop()
        self.name = name
        # stdbuf -o0: pw-record writes stdout through a 4 KB stdio buffer, which delivered the mic in
        # 128 ms bursts; unbuffered, a block arrives every 20 ms
        cmd = ["stdbuf", "-o0", "pw-record", "--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16",
               "--latency", "20ms", "-P", "{ node.name = moonbase-listen }", "-"]
        if name:
            cmd[3:3] = ["--target", name]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        n = MIC_BLOCK * 2
        while True:
            b = proc.stdout.read(n)
            if len(b) < n or proc is not self.proc:
                return
            self.sink(np.frombuffer(b, np.int16).astype(np.float32) / 32768)

    def stop(self):
        if self.proc:
            p, self.proc = self.proc, None
            p.terminate()
            p.wait(timeout=2)

