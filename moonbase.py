"""Moonbase engine: talk, and DECtalk (the Moonbase Alpha voice) says it out of a virtual microphone.

Your mic -> NVIDIA Parakeet TDT 0.6B (on the GPU, or the CPU when there is no NVIDIA card), re-transcribing
what you are saying; a word is spoken as soon as several transcriptions in a row agree on it -> DECtalk says
it with your pitch, loudness and pace -> a microphone that Discord, OBS, games etc. can pick: "Moonbase
Alpha (DECtalk)" on Linux, VB-Audio Virtual Cable's "CABLE Output" on Windows.

The window is app.py; this module only reports events through the `publish` callback.
"""
import ctypes
import difflib
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

import prosody as P
from dectalk import RATE as TTS_RATE, DECtalk

WINDOWS = sys.platform == "win32"
if WINDOWS:
    import audio_windows as audio
else:
    import audio_linux as audio

HERE = os.path.dirname(os.path.abspath(__file__))
# a packaged (PyInstaller) app keeps its models next to the .exe
APP_DIR = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else HERE


def _load_cuda_libs():
    """Make NVIDIA's libraries findable by the GPU build of ONNX Runtime, and say whether the GPU can be used.

    Linux: cuDNN and cuBLAS come as NVIDIA's Python packages (nvidia-cudnn-cu13); load them up front, retrying
    until each one's dependencies are in. Windows: they come in the optional GPU pack, unzipped into
    <app>/cuda; the GPU is used only if that is there and an NVIDIA driver is installed."""
    if WINDOWS:
        cuda = os.path.join(APP_DIR, "cuda")
        if not os.path.isdir(cuda):
            return False
        try:
            ctypes.WinDLL("nvcuda.dll")       # the NVIDIA driver
        except OSError:
            return False
        os.add_dll_directory(cuda)
        os.environ["PATH"] = cuda + os.pathsep + os.environ.get("PATH", "")
        libs, load = sorted(glob.glob(os.path.join(cuda, "*.dll"))), ctypes.WinDLL
    else:
        try:
            import nvidia
        except ImportError:
            return False
        libs = sorted(p for d in nvidia.__path__ for p in glob.glob(os.path.join(d, "*", "lib", "lib*.so*")))
        load = lambda p: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)     # noqa: E731
    while libs:
        left = []
        for p in libs:
            try:
                load(p)
            except OSError:
                left.append(p)
        if len(left) == len(libs):
            break
        libs = left
    return True


GPU = _load_cuda_libs()
import sherpa_onnx  # noqa: E402  (after the CUDA libraries)

if WINDOWS:
    SETTINGS_FILE = os.path.join(os.environ.get("APPDATA") or APP_DIR, "Moonbase", "settings.json")
    CABLE_PACK = os.path.join(APP_DIR, "vbcable", audio.CABLE_PACK)    # VB-Audio's package, if bundled
    # the CUDA plugin of ONNX Runtime comes with the GPU pack too
    GPU = GPU and os.path.isfile(os.path.join(os.path.dirname(sherpa_onnx.__file__), "lib",
                                              "onnxruntime_providers_cuda.dll"))
else:
    SETTINGS_FILE = os.path.join(HERE, "settings.json")
MODEL = os.path.join(APP_DIR, "models", "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16")
# voice letter -> (name, average pitch Hz, pitch range %): the speaker definitions compiled into the US
# engine (ph/p_us_vdf*.c, symbols paul, harry, ... in libtts_us.so)
VOICES = {"p": ("Perfect Paul", 122, 100), "h": ("Huge Harry", 89, 80), "f": ("Frank", 155, 90),
          "d": ("Doctor Dennis", 110, 135), "b": ("Beautiful Betty", 208, 240), "u": ("Uppity Ursula", 240, 135),
          "w": ("Whispering Wendy", 200, 175), "r": ("Rough Rita", 106, 80), "k": ("Kit the Kid", 306, 210)}
DEFAULTS = dict(
    listening=True,       # turn speech into DECtalk
    voice="p",
    rate=180,             # DECtalk words per minute at your usual pace
    follow_pitch=0.0,     # 0 = Paul's flat pitch, 1 = move as much as you do, 2 = exaggerate
    follow_range=0.0,
    follow_pace=0.0,
    follow_volume=0.0,
    emphasis=True,        # stress the word you stressed
    voice_level=-50.0,    # dBFS that counts as your voice (raise it above your keystrokes)
    volume=1.0,
    monitor=False,        # also play the virtual mic on your speakers/headphones
    mic="",               # microphone name; "" = system default at startup
    output="",            # Windows: where DECtalk plays; "" = VB-Audio Virtual Cable if installed
    cable_asked=False,    # Windows: the offer to install VB-Audio Virtual Cable was shown
)
LIMITS = dict(rate=(75, 600), follow_pitch=(0, 2), follow_range=(0, 2), follow_pace=(0, 2),
              follow_volume=(0, 2), volume=(0, 2), voice_level=(-70, -10))
MIC_RATE = P.SR
# Transcription. Measured on recorded speech: word start -> spoken 0.44 s median, 3% wrong words (GPU).
STEP = 0.05                               # re-transcribe the utterance this often (a pass takes ~10-30 ms)
AGREE = 3                                 # a word is spoken once this many passes in a row agree on it...
# On the CPU a pass takes ~80-160 ms (8 threads), so: re-transcribe as often as it can keep up with, and
# accept a word after two agreeing passes. Measured: word start -> committed ~0.55-0.6 s, similar accuracy.
CPU_STEP, CPU_AGREE = 0.15, 2
SETTLE = 0.12                             # ...and, if it is the newest word, once you have been quiet this
                                          # long (until then the end of the audio may cut it off)
PAUSE = 0.3                               # silence that ends an utterance: the rest is spoken, start over
MIN_SPEECH = 0.15                         # don't transcribe less speech than this: silence makes it invent words
PREROLL = 0.1                             # audio kept from before the first speech
MAX_UTTERANCE = 8.0                       # past this, drop the audio of words already spoken
SYLLABLE_S = 0.2                          # time per syllable before your pace is known
MAX_FIT = 2.0                             # most words are sped up to keep up with you


def load_settings():
    s = dict(DEFAULTS)
    try:
        with open(SETTINGS_FILE) as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def recognizer():
    """(recognizer, "GPU" or "CPU")"""
    if not os.path.isfile(os.path.join(MODEL, "encoder.fp16.onnx")):
        raise RuntimeError("Speech model missing. Run ./setup.sh in the project folder." if not WINDOWS else
                           f"Speech model missing: {MODEL}. Unzip the whole Moonbase download again.")
    files = dict(encoder=f"{MODEL}/encoder.fp16.onnx", decoder=f"{MODEL}/decoder.fp16.onnx",
                 joiner=f"{MODEL}/joiner.fp16.onnx", tokens=f"{MODEL}/tokens.txt", model_type="nemo_transducer")
    if GPU:
        try:
            rec = sherpa_onnx.OfflineRecognizer.from_transducer(**files, provider="cuda", num_threads=2)
            transcribe(rec, np.zeros(MIC_RATE, np.float32))       # first pass sets up the GPU (~1 s)
            return rec, "GPU"
        except Exception:
            if not WINDOWS:
                raise
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(**files, provider="cpu",
                                                         num_threads=min(8, os.cpu_count() or 4))
    transcribe(rec, np.zeros(MIC_RATE, np.float32))
    return rec, "CPU"


def transcribe(rec, audio):
    """[(word with its punctuation, start in seconds into `audio`)]"""
    s = rec.create_stream()
    s.accept_waveform(MIC_RATE, audio)
    rec.decode_stream(s)
    words = []
    for tok, ts in zip(s.result.tokens, s.result.timestamps):
        if tok.startswith((" ", "▁")) or not words:
            words.append([tok.strip(" ▁"), ts])
        else:
            words[-1][0] += tok
    return [(w, ts) for w, ts in words if norm(w)]


def norm(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def spoken_end(spoken, words):
    """Index in `words` (a new transcription) just after the words already spoken. Matched by content:
    a new transcription can gain or lose a word in front of them."""
    blocks = [b for b in difflib.SequenceMatcher(a=spoken, b=words, autojunk=False).get_matching_blocks()
              if b.size]
    return blocks[-1].b + blocks[-1].size if blocks else min(len(spoken), len(words))


def save_mp3(pcm, path):
    """Write DECtalk samples as an MP3 at 44.1 kHz (every player and chat app takes it): on Linux with the
    system ffmpeg, on Windows with lameenc (LAME), bundled with the app. Returns an error message or None."""
    if not len(pcm):
        return "DECtalk said nothing"
    if WINDOWS:
        try:
            import lameenc
            enc = lameenc.Encoder()
            enc.set_bit_rate(160)
            enc.set_in_sample_rate(44100)
            enc.set_channels(1)
            enc.set_quality(2)
            x = np.interp(np.arange(len(pcm) * 4) / 4, np.arange(len(pcm)), pcm).astype(np.int16)
            with open(path, "wb") as f:
                f.write(enc.encode(x.tobytes()) + enc.flush())
            return None
        except (OSError, RuntimeError) as e:
            return str(e)
    try:
        p = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "s16le", "-ar", str(TTS_RATE),
                            "-ac", "1", "-i", "-", "-ar", "44100", "-codec:a", "libmp3lame", "-q:a", "2",
                            path], input=pcm.tobytes(), capture_output=True, timeout=60)
        return p.stderr.decode(errors="replace").strip() or "ffmpeg failed" if p.returncode else None
    except (OSError, subprocess.TimeoutExpired) as e:
        return str(e)


def selftest(report):
    """Check a build without audio devices (the Windows build machine has none): DECtalk speaks with index
    marks, the speech recognizer hears what it said, and an MP3 gets written. Writes `report`; returns 0 if
    everything worked, else 1."""
    import traceback
    lines, ok = [], False
    try:
        rec, device = recognizer()
        lines.append(f"recognizer: Parakeet on the {device}")
        tts = DECtalk()
        pcm, marks = tts.synth("[:np] [:index mark 1] John Madden. [:index mark 2] Moonbase alpha.")
        lines.append(f"dectalk: {len(pcm) / TTS_RATE:.2f} s of speech, index marks {marks}")
        x = pcm.astype(np.float32) / 32768
        x = np.interp(np.arange(0, len(x), TTS_RATE / MIC_RATE), np.arange(len(x)), x).astype(np.float32)
        heard = " ".join(w for w, _ in transcribe(rec, x))
        lines.append(f"heard: {heard}")
        mp3 = os.path.splitext(report)[0] + ".mp3"
        error = save_mp3(pcm, mp3)
        lines.append(f"mp3: {error or f'{os.path.getsize(mp3)} bytes'}")
        tts.close()
        ok = len(marks) == 2 and "madden" in heard.lower() and not error
    except Exception:
        lines.append(traceback.format_exc())
    lines.append("PASS" if ok else "FAIL")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0 if ok else 1


class Moonbase:
    """The whole pipeline. `publish(event dict)` is called from worker threads."""

    def __init__(self, publish):
        self.publish = publish
        self.s = load_settings()
        self.rec, self.device = recognizer()
        # on the CPU a pass is slower: transcribe less often and accept words sooner (see CPU_STEP)
        self.step, self.agree = (STEP, AGREE) if self.device == "GPU" else (CPU_STEP, CPU_AGREE)
        self.tts = DECtalk()
        self.out = audio.Output(self.s["output"])
        self.out.set_monitor(self.s["monitor"])
        self.lock = threading.Lock()      # changes of audio devices
        self.frames, self.vad, self.base = P.Frames(), P.Vad(self.s["voice_level"]), P.Baseline()
        self.audio, self.jobs = queue.Queue(), queue.Queue()
        self.phrase = []                  # words of the current utterance already spoken
        self.said = None                  # what the current utterance sent to DECtalk so far
        self.mic = audio.Mic(self.audio.put)
        mic = self.s["mic"]
        self.mic.start(mic if any(m["name"] == mic for m in audio.list_mics()) else audio.default_mic())
        threading.Thread(target=self._listen, daemon=True).start()
        threading.Thread(target=self._speak, daemon=True).start()

    # ---- settings ----
    def state(self):
        return dict(t="state", settings=self.s, voices={k: v[0] for k, v in VOICES.items()},
                    mics=audio.list_mics(), mic=self.mic.name, outputs=audio.list_outputs(),
                    output=self.out.device, hint=self.out.hint, device=self.device,
                    cable_offer=WINDOWS and not audio.cable_installed() and os.path.isfile(CABLE_PACK),
                    baseline=self.base.as_dict())

    def update(self, changes):
        for k, v in changes.items():
            if k not in DEFAULTS:
                continue
            kind = type(DEFAULTS[k])
            v = kind(v)
            if k in LIMITS:
                v = kind(min(max(v, LIMITS[k][0]), LIMITS[k][1]))
            if k == "voice" and v not in VOICES:
                continue
            self.s[k] = v
            with self.lock:
                if k == "voice_level":
                    self.vad.level = v
                elif k == "monitor":
                    self.out.set_monitor(v)
                elif k == "mic":
                    self.mic.start(v or audio.default_mic())
                elif k == "output" and WINDOWS:
                    self.out.set_device(v)
                    self.out.set_monitor(False)
                    self.out.set_monitor(self.s["monitor"])
        self._save()
        self.publish(self.state())

    def _save(self):
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
        with open(SETTINGS_FILE, "w") as f:
            json.dump(self.s, f, indent=1)

    def install_cable(self):
        """Windows: install VB-Audio Virtual Cable from the bundled package and switch DECtalk to it. Blocks
        until VB-Audio's installer closes (call it off the UI thread). Publishes t="cable" with the result:
        "ok", "restart" (installed, but Windows only shows the device after a restart), "declined" or an
        error message."""
        try:
            if not audio.install_cable(CABLE_PACK):
                result = "declined"
            else:
                # PortAudio lists devices once: it must restart to see the cable, with every stream closed
                with self.lock:
                    self.mic.stop()
                    self.out.close()
                    audio.rescan()
                    self.s["output"] = ""     # the cable, now that there is one
                    self.out = audio.Output()
                    self.out.set_monitor(self.s["monitor"])
                    self.mic.start(self.mic.name or audio.default_mic())
                self._save()
                result = "ok" if audio.cable_installed() else "restart"
        except Exception as e:
            result = f"Could not install VB-Audio Virtual Cable: {e}"
        self.publish(self.state())
        self.publish(dict(t="cable", result=result))

    def say(self, text, save_to=None):
        """Say typed text (raw DECtalk input works). With `save_to`, also write it to that MP3 file."""
        self.jobs.put(("typed", (text, save_to), None))

    def stop(self):
        while not self.jobs.empty():
            try:
                self.jobs.get_nowait()
            except queue.Empty:
                break
        self.out.stop()

    def close(self):
        self.mic.stop()
        self.out.close()
        self.tts.close()

    # ---- mic -> words ----
    def _listen(self):
        t = 0.0                           # seconds of mic audio so far
        seg, seg_t0 = [], 0.0             # audio of the current utterance, and where it starts (mic seconds)
        passes, spoken = [], []           # recent transcriptions; words already spoken (normalized)
        due, heard, tick = 0.0, "", 0
        while True:
            x = self.audio.get()
            t += len(x) / MIC_RATE
            n = self.frames.push(x)
            db = self.frames.db[len(self.frames.db) - n:]
            if len(db):
                self.vad.push(db)
            tick += 1
            if tick % 5 == 0 and len(db):
                f0 = self.frames.f0[-n:]
                self.publish(dict(t="level", db=round(float(db.max()), 1), thr=round(self.vad.threshold(), 1),
                                  f0=round(float(f0[f0 > 0].mean())) if (f0 > 0).any() else 0,
                                  speaking=self.vad.silence < 0.1, backlog=round(self.out.backlog(), 1)))
            if not self.s["listening"]:
                if seg:                    # muted mid-utterance: forget it (and learn from what was said)
                    self._end_utterance([], t)
                    seg, passes, spoken = [], [], []
                continue
            seg.append(x)
            if not spoken and not passes:
                # nothing heard yet: in silence keep only a moment before the speech starts (not during a
                # short dip inside a word, like the closure in "okay")
                while sum(map(len, seg)) > PREROLL * MIC_RATE + len(x) and self.vad.silence >= 0.25:
                    seg.pop(0)
            seg_t0 = t - sum(map(len, seg)) / MIC_RATE
            if t < due:
                continue
            if self.device == "CPU" and self.audio.qsize():
                continue                  # a slow CPU is behind the mic: catch up before the next pass
            due = t + self.step
            sdb, _ = self.frames.span(seg_t0, t)
            pause = self.vad.silence >= PAUSE
            if (sdb > self.vad.threshold()).sum() / P.FPS < MIN_SPEECH:
                if pause and not spoken:
                    seg, passes = [], []
                continue

            hyp = transcribe(self.rec, np.concatenate(seg))
            words = [norm(w) for w, _ in hyp]
            passes = (passes + [words])[-self.agree:]
            agree = 0
            if len(passes) == self.agree:
                while agree < min(map(len, passes)) and all(p[agree] == words[agree] for p in passes):
                    agree += 1
            # the newest word may still be cut off by the end of the audio: wait for a later word, or a
            # moment of quiet. At a pause, everything goes.
            if pause:
                upto = len(words)
            else:
                upto = agree if self.vad.silence >= SETTLE else min(agree, len(words) - 1)
            p = spoken_end(spoken, words)
            if upto > p:                   # all words new in this pass go to DECtalk as one clause
                self._words([P.Word(w, seg_t0 + ts) for w, ts in hyp[p:upto]], t,
                            final=pause and upto == len(words))
                spoken += words[p:upto]
            text = " ".join(w for w, _ in hyp)

            if pause:
                self._end_utterance([P.Word(w, seg_t0 + ts) for w, ts in hyp], t - self.vad.silence)
                seg, passes, spoken, text = [], [], [], ""
            elif t - seg_t0 > MAX_UTTERANCE:
                # long run without a pause: drop the audio of the words already spoken
                k = spoken_end(spoken, words)
                cut = hyp[k][1] - 0.05 if k < len(hyp) else t - seg_t0 - 1.0
                audio = np.concatenate(seg)[max(0, int(cut * MIC_RATE)):]
                seg, passes, spoken = [audio], [], []
            if text != heard:
                heard = text
                self.publish(dict(t="heard", text=text))

    def _end_utterance(self, words, end):
        """The utterance is over: learn from all of it (with the model's word times) and report it."""
        if self.phrase:
            d = P.measure(self.frames, words or self.phrase, end, self.vad.threshold(), self.base)
            self.phrase = []
            self.jobs.put(("close", d, None))

    def _words(self, words, now, final):
        """Speak these words now, shaped by themselves and the utterance so far (the baseline isn't changed)."""
        self.phrase.extend(words)
        d = P.measure(self.frames, self.phrase, now - 0.05, self.vad.threshold(), self.base, learn=False)
        per = 1 / self.base.pace if self.base.pace else SYLLABLE_S
        slot = sum(max(1, P.syllables(w.text)) for w in words) * per    # the time you take for words like these
        started = [time.monotonic() - (now - w.start) for w in words]    # when you started each one
        self.jobs.put(("words", (d, len(words), final, slot), started))

    # ---- words -> DECtalk -> virtual mic ----
    def _speak(self):
        while True:
            kind, item, spoke_at = self.jobs.get()
            s = dict(self.s)
            _, ap0, pr0 = VOICES[s["voice"]]
            if kind == "close":
                self._report_phrase(item)
                continue
            if kind == "typed":
                text, save_to = item
                rate = round(min(max(s["rate"], 75), 600))
                sc = P.Script([(0.0, f"[:n{s['voice']}][:phoneme on][:rate {rate}] {text}")], 0.0, [], rate, ap0, pr0)
                pcm, _ = self._synth(sc, dict(s, volume=1.0))
                if save_to:                    # the file at full volume, whatever the output volume
                    self._save_mp3(pcm, save_to)
                pcm = np.clip(pcm * s["volume"], -32768, 32767).astype(np.int16)
                self.publish(dict(t="said", kind=kind, id=time.monotonic_ns(), dectalk=sc.text, rate=sc.rate,
                                  ap=sc.ap, pr=sc.pr, gain=0.0, seconds=round(len(pcm) / TTS_RATE, 2),
                                  words=[text], emphasis=[], pauses=[]))
                if len(pcm):
                    self.out.play(pcm)
                continue

            d, n, final, slot = item
            # words said on their own come out longer than inside a sentence: fit them into the time you
            # took, minus what is still playing
            budget = max(0.5 * slot, slot - self.out.backlog())
            cap = s["rate"] * MAX_FIT
            sc = P.words_script(d, n, s, ap0, pr0, final)
            pcm, offsets = self._synth(sc, s)
            for _ in range(2):
                over = len(pcm) / TTS_RATE / max(budget, 0.1)
                if over <= 1.05 or s["rate"] >= cap:
                    break
                s["rate"] = min(cap, s["rate"] * over * 1.05)
                sc = P.words_script(d, n, s, ap0, pr0, final)
                pcm, offsets = self._synth(sc, s)

            if self.said is None:
                self.said = dict(words=[], emphasis=[], texts=[], rates=[], aps=[], prs=[], seconds=0.0,
                                 latency=[], question=False)
            acc = self.said
            first = len(d.words) - n
            acc["emphasis"] += [len(acc["words"]) + k for k in range(n) if first + k in d.emphasis]
            acc["words"] += d.words[first:]
            acc["question"] = final and d.question
            for k, v in (("texts", sc.text), ("rates", sc.rate), ("aps", sc.ap), ("prs", sc.pr)):
                acc[k].append(v)
            acc["seconds"] += len(pcm) / TTS_RATE
            if len(pcm):
                # how long after you started each word DECtalk starts it (its index mark in the clause)
                self.out.play(pcm, lambda at, sp=spoke_at, off=offsets, lat=acc["latency"]:
                              lat.extend(at + off.get(k, 0.0) - sp[k] for k in range(len(sp))))

    def _save_mp3(self, pcm, path):
        self.publish(dict(t="saved", path=path, error=save_mp3(pcm, path)))

    def _synth(self, sc, s):
        """(samples, {index mark: seconds into the samples})"""
        parts, offsets, pos = [], {}, 0
        for gap, text in sc.segments:
            if gap:
                parts.append(np.zeros(round(gap * TTS_RATE), np.int16))
                pos += len(parts[-1])
            seg, marks = self.tts.synth(text)
            x, cut = P.render(seg, marks, sc, s["volume"], TTS_RATE)
            offsets.update((v, (pos + max(0, smp - cut)) / TTS_RATE) for v, smp in marks)
            parts.append(x)
            pos += len(x)
        return np.concatenate(parts), offsets

    def _report_phrase(self, d):
        """One log entry per utterance, with what was learned from all of it. `ms`: the median time from
        you starting a word to DECtalk starting it."""
        acc, self.said = self.said, None
        if not acc:
            return
        lat = sorted(acc["latency"])
        self.publish(dict(
            t="said", kind="speech", id=time.monotonic_ns(), words=acc["words"], emphasis=acc["emphasis"],
            question=acc["question"], pauses=[], pitch_st=round(d.pitch_st, 1), range_x=round(d.range_x, 2),
            pace_x=round(d.pace_x, 2), loud_db=round(d.loud_db, 1), baseline=self.base.as_dict(),
            dectalk=" | ".join(acc["texts"]), rate=round(float(np.mean(acc["rates"]))),
            ap=round(float(np.mean(acc["aps"]))), pr=round(float(np.mean(acc["prs"]))),
            gain=0.0, seconds=round(acc["seconds"], 2),
            ms=round(lat[len(lat) // 2] * 1000) if lat else None))
