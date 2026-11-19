"""In-process DECtalk (github.com/dectalk/dectalk, built from source: by setup.sh on Linux, by the Windows build
workflow on Windows) via ctypes, speech-to-memory.

`DECtalk().synth(text)` returns int16 mono samples at 11025 Hz plus the sample positions of any
`[:index mark N]` commands in the text.
"""
import ctypes as C
import os
import sys
import threading

import numpy as np

RATE = 11025
_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.platform == "win32":
    # DECtalk.dll and dtalk_us.dic: next to Moonbase.exe when packaged
    _ROOT = (os.path.join(os.path.dirname(sys.executable), "dectalk") if getattr(sys, "frozen", False)
             else os.path.join(_HERE, "vendor", "dectalk-win"))
    _LIB = "DECtalk.dll"
    # the Windows build announces a filled buffer with a registered window message, not TTS_MSG_BUFFER
    _MSG_BUFFER = C.windll.user32.RegisterWindowMessageA(b"DECtalkBufferMessage")
else:
    _ROOT = os.path.join(_HERE, "vendor", "src", "dectalk", "dist")
    _LIB = os.path.join("lib", "libtts.so")                  # loads lib/libtts_us.so
    _MSG_BUFFER = 9                                           # TTS_MSG_BUFFER

_WAVE_FORMAT_1M16 = 0x4          # 11025 Hz, mono, 16-bit PCM
_DO_NOT_USE_AUDIO_DEVICE = 0x80000000
_TTS_FORCE = 1
_NBUF, _BUFLEN, _NIDX = 8, 16384, 64


class _Index(C.Structure):
    _fields_ = [("value", C.c_uint), ("sample", C.c_uint), ("res", C.c_uint)]


class _Buffer(C.Structure):
    _fields_ = [("data", C.c_void_p), ("phonemes", C.c_void_p), ("index", C.POINTER(_Index)),
                ("maxlen", C.c_uint), ("maxph", C.c_uint), ("maxidx", C.c_uint),
                ("len", C.c_uint), ("nph", C.c_uint), ("nidx", C.c_uint), ("res", C.c_uint)]


_CALLBACK = C.CFUNCTYPE(None, C.c_long, C.c_long, C.c_uint, C.c_uint)


class DECtalk:
    """One engine instance. `synth` is serialized; call it from any thread."""

    def __init__(self, root=_ROOT):
        # The engine reads DECtalk.conf and its dictionary relative to the working directory at startup.
        cwd = os.getcwd()
        os.chdir(root)
        try:
            self._lib = lib = C.CDLL(os.path.join(root, _LIB))
            for name, args in [
                ("TextToSpeechStartupEx", [C.c_void_p, C.c_uint, C.c_uint, _CALLBACK, C.c_long]),
                ("TextToSpeechSpeak", [C.c_void_p, C.c_char_p, C.c_uint]),
                ("TextToSpeechSync", [C.c_void_p]),
                ("TextToSpeechShutdown", [C.c_void_p]),
                ("TextToSpeechOpenInMemory", [C.c_void_p, C.c_uint]),
                ("TextToSpeechCloseInMemory", [C.c_void_p]),
                ("TextToSpeechAddBuffer", [C.c_void_p, C.POINTER(_Buffer)]),
            ]:
                fn = getattr(lib, name)
                fn.argtypes, fn.restype = args, C.c_uint
            self._cb = _CALLBACK(self._on_msg)          # keep a reference: the engine calls it later
            self._handle = C.c_void_p()
            r = lib.TextToSpeechStartupEx(C.byref(self._handle), 0, _DO_NOT_USE_AUDIO_DEVICE, self._cb, 0)
        finally:
            os.chdir(cwd)
        if r:
            raise RuntimeError(f"DECtalk startup failed ({r}); is {root} complete?")
        self._h = self._handle
        self._lock = threading.Lock()
        self._chunks, self._marks, self._emitted = [], [], 0
        # Speech-to-memory stays open for the engine's lifetime: reopening it per utterance leaks the
        # engine's buffer slots and it goes silent after a few utterances.
        if lib.TextToSpeechOpenInMemory(self._h, _WAVE_FORMAT_1M16):
            raise RuntimeError("DECtalk: OpenInMemory failed")
        self._data = [C.create_string_buffer(_BUFLEN) for _ in range(_NBUF)]
        self._idx = [(_Index * _NIDX)() for _ in range(_NBUF)]
        self._bufs = [_Buffer(data=C.cast(d, C.c_void_p), index=C.cast(i, C.POINTER(_Index)),
                              maxlen=_BUFLEN, maxidx=_NIDX) for d, i in zip(self._data, self._idx)]
        for b in self._bufs:
            lib.TextToSpeechAddBuffer(self._h, C.pointer(b))
        # the callback identifies a buffer by the low 32 bits of its address: on 64-bit Windows the engine
        # passes the pointer through a 32-bit `long` and the top half is lost
        self._by_addr = {C.addressof(b) & 0xFFFFFFFF: b for b in self._bufs}

    def _on_msg(self, _l1, l2, _user, msg):
        if msg != _MSG_BUFFER:
            return
        b = self._by_addr.get(l2 & 0xFFFFFFFF)
        if b is None:
            return
        self._chunks.append(C.string_at(b.data, b.len))
        self._marks.extend((b.index[i].value, b.index[i].sample) for i in range(b.nidx))
        self._emitted += b.len // 2
        b.len = b.nidx = b.nph = 0
        self._lib.TextToSpeechAddBuffer(self._h, C.pointer(b))

    def synth(self, text):
        """Return (int16 samples, [(index mark value, sample), ...]) for one utterance.

        In-line commands ([:rate], [:dv], [:phoneme on], ...) persist into later utterances.
        """
        with self._lock:
            start = self._emitted            # index-mark sample numbers count from engine startup
            self._chunks, self._marks = [], []
            self._lib.TextToSpeechSpeak(self._h, text.encode("latin-1", "replace"), _TTS_FORCE)
            self._lib.TextToSpeechSync(self._h)  # with TTS_FORCE this flushes the last partial buffer
            pcm = np.frombuffer(b"".join(self._chunks), np.int16).copy()
            marks = sorted(((v, max(0, s - start)) for v, s in self._marks), key=lambda m: m[1])
            return pcm, marks

    def close(self):
        with self._lock:
            self._lib.TextToSpeechCloseInMemory(self._h)
            self._lib.TextToSpeechShutdown(self._h)
