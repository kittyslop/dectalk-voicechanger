"""How you said something, and how to make DECtalk say it the same way.

Frames      per-10 ms loudness and pitch of the mic stream
measure()   one phrase -> Delivery: pitch, pitch movement, pace and loudness relative to your usual,
            plus per-word loudness, pauses, emphasis and a rising (question) ending
Baseline    your usual, learned as you talk
script()    Delivery -> DECtalk text with in-line commands, and the gain to apply per word
render()    apply those gains to the synthesized samples and trim the trailing silence
"""
import re
from dataclasses import dataclass, field

import numpy as np

SR = 16000
HOP = 160                 # 10 ms frames
WIN = 640                 # 40 ms analysis window (two periods of the lowest pitch)
FPS = SR // HOP
FMIN, FMAX = 65, 400
_LO, _HI = SR // FMAX, SR // FMIN

_W = WIN - _HI - 2        # YIN integration window: what is left of the frame after the longest lag
_NFFT = 2048

STOPWORDS = set("""a an the and or but so if of to in on at by for with from up as is am are was were be been
it its it's i i'm me my we us our you your he him his she her they them their this that these those
do does did have has had will would can could should not no yes just than then there here what""".split())


def semitones(hz):
    return 12 * np.log2(np.asarray(hz, np.float64) / 100.0)


class Frames:
    """Loudness (dBFS) and pitch (Hz, 0 when unvoiced) of a continuous 16 kHz stream, per 10 ms."""

    def __init__(self, keep_s=60):
        self._tail = np.zeros(0, np.float32)
        self._keep = keep_s * FPS
        self.db = np.zeros(0, np.float32)
        self.f0 = np.zeros(0, np.float32)
        self.first = 0                      # absolute index of db[0]

    @property
    def end(self):
        return self.first + len(self.db)

    def push(self, x):
        buf = np.concatenate([self._tail, x])
        n = (len(buf) - WIN) // HOP + 1 if len(buf) >= WIN else 0
        if n <= 0:
            self._tail = buf
            return 0
        fr = np.lib.stride_tricks.sliding_window_view(buf, WIN)[::HOP][:n]
        self._tail = buf[n * HOP:]
        fr = fr - fr.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(fr * fr, axis=1))
        db = 20 * np.log10(rms + 1e-7)
        f0 = self._yin(fr, db)
        self.db = np.concatenate([self.db, db.astype(np.float32)])
        self.f0 = np.concatenate([self.f0, f0])
        if len(self.db) > self._keep:
            cut = len(self.db) - self._keep
            self.db, self.f0, self.first = self.db[cut:], self.f0[cut:], self.first + cut
        return n

    def span(self, t0, t1):
        """(db, f0) for stream seconds [t0, t1)."""
        a = max(int(t0 * FPS), self.first) - self.first
        b = max(int(t1 * FPS), self.first) - self.first
        return self.db[a:b], self.f0[a:b]

    @staticmethod
    def _yin(fr, db):
        """YIN (de Cheveigné & Kawahara 2002): the first dip of the cumulative-mean-normalized difference
        function, which unlike plain autocorrelation rarely jumps an octave."""
        taus = np.arange(_HI + 2)
        # d(tau) = sum_j (x_j - x_{j+tau})^2 over j < _W, from energies and an FFT cross-correlation
        r = np.fft.irfft(np.conj(np.fft.rfft(fr[:, :_W], _NFFT)) * np.fft.rfft(fr, _NFFT))[:, :_HI + 2]
        c = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(fr * fr, axis=1)], axis=1)
        e0 = c[:, _W:_W + 1]
        et = c[:, taus + _W] - c[:, taus]
        d = np.maximum(e0 + et - 2 * r, 0)
        cm = np.cumsum(d[:, 1:], axis=1)
        dn = np.ones_like(d)
        dn[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cm, 1e-12)
        f0 = np.zeros(len(fr), np.float32)
        for i in range(len(fr)):
            if db[i] < -60:
                continue
            a = dn[i]
            below = np.flatnonzero(a[_LO:_HI] < 0.15)
            if len(below):
                tau = _LO + int(below[0])
                while tau + 1 < _HI and a[tau + 1] < a[tau]:
                    tau += 1
            else:
                tau = _LO + int(np.argmin(a[_LO:_HI]))
            if a[tau] > 0.3:
                continue                    # aperiodic: unvoiced
            l, m, h = a[tau - 1], a[tau], a[tau + 1]
            den = l - 2 * m + h
            shift = 0.5 * (l - h) / den if den > 0 else 0.0
            f0[i] = SR / (tau + shift)
        return f0


class Vad:
    """Speech/silence: louder than `level` dBFS (per 10 ms frame) counts as your voice. An absolute level,
    not one relative to the room's noise: some USB mics send pure digital silence when
    you're quiet, so there is no noise floor to measure against."""

    def __init__(self, level=-50.0):
        self.level = level
        self.silence = 0.0                  # seconds of silence at the end of the stream

    def threshold(self):
        return self.level

    def push(self, db):
        speech = db > self.level
        if speech.any():
            self.silence = (len(db) - 1 - int(np.flatnonzero(speech)[-1])) / FPS
        else:
            self.silence += len(db) / FPS
        return speech


@dataclass
class Word:
    text: str
    start: float                            # stream seconds


@dataclass
class Delivery:
    words: list
    pitch_st: float = 0.0                   # semitones above your usual
    range_x: float = 1.0                    # pitch movement relative to your usual
    pace_x: float = 1.0                     # syllables per second of speech relative to your usual
    loud_db: float = 0.0                    # loudness relative to your usual
    word_db: list = field(default_factory=list)    # each word's loudness relative to the phrase
    word_st: list = field(default_factory=list)    # each word's median pitch, semitones above your usual
    pauses: list = field(default_factory=list)     # silence (s) before each word
    emphasis: list = field(default_factory=list)   # indexes of stressed words
    question: bool = False


def syllables(word):
    w = word.lower()
    digits = sum(c.isdigit() for c in w)
    w = re.sub(r"[^a-z]", "", w)
    n = len(re.findall(r"[aeiouy]+", w))
    if n > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(1 if w else 0, n) + digits


class Baseline:
    """Your usual pitch (semitones re 100 Hz), pitch movement, pace and loudness."""

    def __init__(self):
        self.n = 0
        self.level = self.spread = self.pace = self.loud = None

    def update(self, level, spread, pace, loud):
        self.n += 1
        a = max(0.08, 1.0 / self.n)       # average of everything at first, then a slow moving average
        for k, v in (("level", level), ("spread", spread), ("pace", pace), ("loud", loud)):
            if v is not None:
                old = getattr(self, k)
                setattr(self, k, v if old is None else (1 - a) * old + a * v)

    def as_dict(self):
        hz = None if self.level is None else 100 * 2 ** (self.level / 12)
        return dict(pitch_hz=hz and round(hz), spread_st=self.spread and round(self.spread, 2),
                    pace_sps=self.pace and round(self.pace, 2), loud_db=self.loud and round(self.loud, 1))


def measure(frames, words, end, vad_threshold, base, learn=True):
    """Describe a phrase (words with start times, ending at stream second `end`) relative to `base`;
    with `learn`, fold it into `base` afterwards."""
    n = len(words)
    bounds = [w.start for w in words] + [max(end, words[-1].start + 0.15)]
    t0 = bounds[0] - 0.05
    db, f0 = frames.span(t0, bounds[-1])
    speech = db > vad_threshold
    voiced = (f0 > 0) & speech
    idx = np.flatnonzero(voiced)
    st = semitones(f0[idx])
    if len(st) >= 5:            # 5-frame median: removes single-frame octave errors
        st = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(st, 2, mode="edge"), 5), axis=1)

    level = float(np.median(st)) if len(st) >= 5 else None
    # interquartile range: how much the pitch moves, ignoring the final fall and tracking outliers
    spread = float(np.percentile(st, 75) - np.percentile(st, 25)) if len(st) >= 10 else None
    speech_s = speech.sum() / FPS
    syl = sum(syllables(w.text) for w in words)
    pace = syl / speech_s if syl >= 3 and speech_s >= 0.4 else None
    loud = float(np.mean(db[speech])) if speech.any() else None

    d = Delivery(words=[w.text for w in words])
    usual = base.level
    if base.n:
        if level is not None and base.level is not None:
            d.pitch_st = level - base.level
        if spread is not None and base.spread:
            d.range_x = spread / max(base.spread, 0.3)
        if pace is not None and base.pace:
            d.pace_x = pace / base.pace
        if loud is not None and base.loud is not None:
            d.loud_db = loud - base.loud
    if learn:
        base.update(level, spread, pace, loud)

    # per word: loudness, and pitch peak (a stress is a short rise, so the peak, not the median);
    # a stressed word stands out from the other words of the phrase
    phrase_db = loud if loud is not None else 0.0
    peaks, ok = [], []
    for i, w in enumerate(words):
        wdb, wf0 = frames.span(bounds[i], bounds[i + 1])
        sp = wdb > vad_threshold
        d.word_db.append(float(np.mean(wdb[sp])) - phrase_db if sp.any() else 0.0)
        v = (wf0 > 0) & sp
        peaks.append(float(np.percentile(semitones(wf0[v]), 90)) if v.sum() >= 3 else np.nan)
        d.word_st.append(float(np.median(semitones(wf0[v]))) - usual
                         if v.sum() >= 3 and usual is not None else np.nan)
        core = re.sub(r"[^a-z']", "", w.text.lower())
        ok.append(len(core) >= 3 and core not in STOPWORDS)
    if n >= 3 and not np.all(np.isnan(peaks)):
        unit = max(spread or 1.5, 1.5)
        peaks = np.array(peaks)
        # pitch drifts down over a phrase (declination): judge each peak against the trend line
        have = ~np.isnan(peaks)
        trend = np.polyval(np.polyfit(np.flatnonzero(have), peaks[have], 1), np.arange(n)) \
            if have.sum() >= 3 else np.full(n, np.nanmedian(peaks))
        rise = np.nan_to_num(peaks - trend) / unit
        score = np.where(ok, rise + (np.array(d.word_db) - np.median(d.word_db)) / 4, -9)
        score[0] -= 0.8           # the first content word carries a pitch accent anyway
        for i in np.argsort(score)[::-1][: 1 + n // 10]:
            if score[i] >= 1.8:
                d.emphasis.append(int(i))

    # pauses: silences between speech inside the phrase, each given to the nearest word start
    # (word times are only good to ~0.1 s, so a pause can't be found from word spans alone)
    d.pauses = [0.0] * n
    sp_idx = np.flatnonzero(speech)
    if n > 1 and len(sp_idx):
        starts = np.array([w.start for w in words[1:]])
        for a, b in _runs(~speech[sp_idx[0]:sp_idx[-1] + 1]):
            if (b - a) / FPS >= 0.15:
                k = 1 + int(np.argmin(np.abs(starts - (t0 + (sp_idx[0] + b) / FPS))))
                d.pauses[k] = max(d.pauses[k], (b - a) / FPS)

    # rising ending: the last 300 ms of voicing sits well above the phrase and is still climbing
    if len(idx) >= 10 and level is not None:
        sel = idx >= idx[-1] - 30
        if sel.sum() >= 6:
            tail, tst = idx[sel], st[sel]
            slope = np.polyfit(tail / FPS, tst, 1)[0]
            d.question = float(np.median(tst[len(tst) // 2:])) - level >= 2.5 and slope > 4
    return d


def _runs(mask):
    """[(start, end)) of each run of True."""
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(m))
    return list(zip(edges[::2], edges[1::2]))


@dataclass
class Script:
    segments: list            # [(silence before it in s, DECtalk text)], each synthesized separately
    gain_db: float
    word_gain_db: list
    rate: int
    ap: int
    pr: int

    @property
    def text(self):
        return " ‖ ".join(t for _, t in self.segments)


_CLEAN = re.compile(r"[^A-Za-z0-9'.,?!%$&-]")


def script(d, s, voice_ap, voice_pr):
    """DECtalk text for a Delivery. `s` holds the voice and how strongly to follow each feature.

    A pause splits the phrase into segments: DECtalk's own silence phoneme displaces index marks, so the
    measured silence is inserted between separately synthesized segments instead."""
    fp, fr, fs, fv = s["follow_pitch"], s["follow_range"], s["follow_pace"], s["follow_volume"]
    ap = round(np.clip(voice_ap * 2 ** (fp * np.clip(d.pitch_st, -8, 8) / 12), 50, 350))
    pr = round(np.clip(voice_pr * np.clip(d.range_x, 0.3, 3.0) ** fr, 0, 250))
    rate = round(np.clip(s["rate"] * np.clip(d.pace_x, 0.5, 2.0) ** fs, 75, 600))
    gain = float(np.clip(fv * d.loud_db, -12, 8))
    wgain = [float(np.clip(0.6 * fv * g, -6, 4)) for g in d.word_db]

    head = f"[:n{s['voice']}][:phoneme on][:rate {rate}][:dv ap {ap} pr {pr}]"
    words = [_CLEAN.sub("", w) for w in d.words]
    if d.question and words and not words[-1].endswith("?"):
        words[-1] = words[-1].rstrip(".,!") + "?"
    segs, parts, before = [], [], 0.0
    for i, w in enumerate(words):
        gap = d.pauses[i] if i < len(d.pauses) else 0.0
        if parts and gap >= 0.15:
            if not parts[-1].endswith((".", ",", "?", "!", ";", ":")):
                parts[-1] += ","              # continuing intonation into the pause
            segs.append((before, f"{head} {' '.join(parts)}"))
            parts, before = [], min(gap, 1.5)
        if s["emphasis"] and i in d.emphasis:
            parts.append('["]')               # before the mark: a mark right before a bracket is dropped
        parts.append(f"[:index mark {i}] {w}")
    segs.append((before, f"{head} {' '.join(parts)}"))
    return Script(segs, gain, wgain, rate, ap, pr)


def words_script(d, n, s, voice_ap, voice_pr, final):
    """DECtalk text for the newest `n` words of an utterance in progress, said as one clause as soon as they
    are heard (one clause is ~40% shorter than the same words said one by one).

    The group follows its own pitch, and each word its own loudness and stress; pitch movement and pace come
    from the utterance so far. Punctuation inside the clause is dropped (DECtalk would pause there); if more
    is coming, the last word gets a comma (continuing intonation; the comma pause is trimmed off by render);
    at the end of the utterance it gets the falling or rising ending."""
    idx = list(range(len(d.words) - n, len(d.words)))
    words = [d.words[i].rstrip(".,;:!?") for i in idx[:-1]] + [d.words[idx[-1]]]
    if not words[-1].endswith((".", ",", "?", "!", ";", ":")):
        words[-1] += "," if not final else "?" if d.question else "."
    st = [d.word_st[i] for i in idx if d.word_st and not np.isnan(d.word_st[i])]
    db = [d.word_db[i] for i in idx] if d.word_db else [0.0] * n
    mean_db = float(np.mean(db))
    group = Delivery(words=words, pitch_st=float(np.median(st)) if st else d.pitch_st, range_x=d.range_x,
                     pace_x=d.pace_x, loud_db=d.loud_db + mean_db, word_db=[x - mean_db for x in db],
                     pauses=[0.0] * n, emphasis=[k for k, i in enumerate(idx) if i in d.emphasis],
                     question=final and d.question)
    return script(group, s, voice_ap, voice_pr)


def render(pcm, marks, sc, volume, rate):
    """Apply phrase and per-word gain (ramped over 10 ms) and cut DECtalk's leading and trailing silence.
    Returns (samples, how many leading samples were cut)."""
    if not len(pcm):                        # e.g. text that was only DECtalk commands
        return pcm.astype(np.int16), 0
    x = pcm.astype(np.float32)
    g = np.full(len(x), sc.gain_db, np.float32)
    for (i, start), nxt in zip(marks, marks[1:] + [(None, len(x))]):
        if i < len(sc.word_gain_db):
            g[start:nxt[1]] += sc.word_gain_db[i]
    k = max(1, rate // 100)
    g = np.convolve(np.pad(g, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    x *= volume * 10 ** (g / 20)
    cut = 0
    loud = np.flatnonzero(np.abs(x) > 150)
    if len(loud):
        cut = max(0, loud[0] - rate // 200)
        x = x[cut: min(len(x), loud[-1] + rate // 25)]
    return np.clip(x, -32768, 32767).astype(np.int16), cut
