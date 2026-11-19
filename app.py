#!/usr/bin/env python3
"""Moonbase desktop app: a Windows XP styled window around the engine in moonbase.py.

Started from the app launcher (Linux: setup.sh installs the entry) or Moonbase.exe (Windows). Closing the
window stops everything (on Linux it also removes the "Moonbase Alpha (DECtalk)" input device).

`Moonbase.exe --selftest REPORT` checks a packaged build without a window or audio devices.
"""
import os
import queue
import re
import signal
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WINDOWS = sys.platform == "win32"

FACE, WHITE, EDGE, GROUP, LEGEND = "#ece9d8", "#ffffff", "#7f9db9", "#d0d0bf", "#0046d5"
DIM = "#5a5a5a"
UI = "Tahoma" if WINDOWS else "DejaVu Sans"   # Tahoma: Windows XP's own UI font
FONT = (UI, 9)
SMALL = (UI, 8)
# Luna title bar, top to bottom
LUNA = [(0.0, "#0997ff"), (0.08, "#0053ee"), (0.40, "#0050ee"), (0.88, "#0066ff"), (0.95, "#005bff"),
        (0.96, "#003dd7"), (1.0, "#003dd7")]

SLIDERS = [
    ("voice_level", "Voice detection level, dB (raise above keystrokes)", -70, -10, 1),
    ("rate", "Speed (words/min at your usual pace)", 75, 400, 5),
    ("follow_pitch", "Follow my pitch", 0, 2, 0.05),
    ("follow_range", "Follow my expressiveness", 0, 2, 0.05),
    ("follow_pace", "Follow my pace", 0, 2, 0.05),
    ("follow_volume", "Follow my loudness", 0, 2, 0.05),
    ("volume", "Volume", 0, 2, 0.05),
]
CHECKS = [("emphasis", "Stress the words I stress"), ("monitor", "Hear it myself (use headphones)")]
PRESETS = [
    ("aeiou", "aeiou"),
    ("John Madden", "John Madden. John Madden. John Madden."),
    ("uuuuuuuuuuuu", "[uw<2000,22>]"),
    ("Happy birthday", "[hxae<300,10>piy<300,10> brr<600,12>th<100>dey<600,10>tuw<600,15>yu<1200,14>]"),
]


def mix(a, b, f):
    pa, pb = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in zip(pa, pb))


def gradient(stops, f):
    for (f0, c0), (f1, c1) in zip(stops, stops[1:]):
        if f <= f1:
            return mix(c0, c1, (f - f0) / (f1 - f0) if f1 > f0 else 0)
    return stops[-1][1]


def xp_style(root):
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=FACE, foreground="#000", font=FONT, bordercolor=GROUP,
                troughcolor=WHITE, focuscolor="#e8a33d")
    s.configure("TButton", background="#f5f4ef", bordercolor="#003c74", lightcolor=WHITE,
                darkcolor="#d6d0c5", padding=(10, 2), relief="raised")
    s.map("TButton", background=[("pressed", "#e3e3db"), ("active", "#fdf6e3")],
          lightcolor=[("active", "#fff0cf")], darkcolor=[("active", "#f8b330")])
    s.configure("Small.TButton", font=SMALL, padding=(6, 0))
    s.configure("On.TButton", background="#e3e3db", darkcolor="#8fb7e8")
    s.configure("TLabelframe", background=FACE, bordercolor=GROUP, lightcolor=FACE, darkcolor=FACE, relief="solid")
    s.configure("TLabelframe.Label", background=FACE, foreground=LEGEND)
    s.configure("TLabel", background=FACE)
    s.configure("Dim.TLabel", foreground=DIM)
    s.configure("TCheckbutton", background=FACE, indicatorbackground=WHITE, indicatorforeground="#21a121",
                indicatormargin=(0, 0, 4, 0))
    s.map("TCheckbutton", background=[("active", FACE)])
    for w in ("TCombobox", "TEntry"):
        s.configure(w, fieldbackground=WHITE, bordercolor=EDGE, lightcolor=WHITE, darkcolor=WHITE,
                    selectbackground="#316ac5", selectforeground=WHITE)
    s.configure("TCombobox", background="#3c7ff0", arrowcolor=WHITE, arrowsize=12)
    s.map("TCombobox", fieldbackground=[("readonly", WHITE)], background=[("active", "#5f97f6")])
    s.configure("Horizontal.TScale", background="#f5f4ef", bordercolor="#9d9c99", troughcolor="#e6e3d6",
                lightcolor=WHITE, darkcolor="#c3bfae", sliderlength=11)
    s.configure("Vertical.TScrollbar", background="#c1d3fb", troughcolor="#f3f1ec", bordercolor="#b5c9f5",
                arrowcolor="#4d6185", lightcolor="#e3ebfd", darkcolor="#9fb5e8")
    root.option_add("*TCombobox*Listbox.font", FONT)
    root.option_add("*TCombobox*Listbox.selectBackground", "#316ac5")


class App:
    def __init__(self, root):
        self.root, self.mb, self.events = root, None, queue.Queue()
        self.loading, self.closing, self.dead, self._applying = True, False, False, False
        self._cable_busy = self._cable_offered = False
        self.voices, self.mics, self.settings = {}, [], {}
        self.vars, self.values = {}, {}
        downloads = os.path.expanduser("~/Downloads")
        self.save_dir = downloads if os.path.isdir(downloads) else os.path.expanduser("~")
        root.title("Moonbase Alpha - DECtalk")
        root.configure(background=FACE)
        root.geometry("1060x790")
        root.minsize(860, 740)
        icon = os.path.join(HERE, "icon.png")
        if os.path.exists(icon):
            self._icon = tk.PhotoImage(file=icon)
            root.iconphoto(True, self._icon)
        xp_style(root)
        self._build()
        root.protocol("WM_DELETE_WINDOW", self.quit)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: root.after(0, self.quit))
        self.status("Starting: loading the speech model...")
        threading.Thread(target=self._load, daemon=True).start()
        root.after(40, self._poll)

    # ---- layout ----
    def _build(self):
        r = self.root
        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)

        self.banner = tk.Canvas(r, height=34, highlightthickness=0, bd=0)
        self.banner.grid(row=0, column=0, sticky="ew")
        self.banner.bind("<Configure>", lambda e: self._draw_banner())
        self.listen_btn = ttk.Button(self.banner, text="Listening", command=self._toggle_listen)

        body = ttk.Frame(r, padding=8)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        right = ttk.Frame(body, width=300)
        right.grid(row=0, column=1, sticky="ns")

        # microphone
        mic = ttk.LabelFrame(left, text="Microphone", padding=(8, 4, 8, 8))
        mic.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        mic.columnconfigure(0, weight=1)
        self.meter = tk.Canvas(mic, height=18, background=WHITE, highlightthickness=1,
                               highlightbackground="#686868", bd=0, cursor="sb_h_double_arrow")
        self.meter.grid(row=0, column=0, sticky="ew")
        # click or drag on the meter to set the voice detection level (the red line)
        self.meter.bind("<Button-1>", self._meter_drag)
        self.meter.bind("<B1-Motion>", self._meter_drag)
        self.meter.bind("<ButtonRelease-1>", lambda e: self._set(voice_level=self._level()))
        self._meter_now = (-70.0, False)
        read = ttk.Frame(mic)
        read.grid(row=1, column=0, sticky="w", pady=(4, 4))
        self.pitch_lbl = ttk.Label(read, text="Pitch: -", style="Dim.TLabel")
        self.pitch_lbl.pack(side="left", padx=(0, 16))
        self.behind_lbl = ttk.Label(read, text="Behind: 0.0 s", style="Dim.TLabel")
        self.behind_lbl.pack(side="left")
        heard = tk.Frame(mic, background=WHITE, highlightthickness=1, highlightbackground=EDGE)
        heard.grid(row=2, column=0, sticky="ew")
        self.heard = tk.Label(heard, text="", anchor="w", justify="left", background=WHITE,
                              foreground="#0b2e8c", font=(UI, 12), padx=6, pady=4, height=2)
        self.heard.pack(fill="x")
        heard.bind("<Configure>", lambda e: self.heard.configure(wraplength=e.width - 16))

        # typing
        typ = ttk.LabelFrame(left, text="Type to Paul", padding=(8, 4, 8, 8))
        typ.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        typ.columnconfigure(0, weight=1)
        self.entry = ttk.Entry(typ)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.entry.bind("<Return>", self._say)
        ttk.Button(typ, text="Speak + save MP3", command=self._say_save).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(typ, text="Stop", command=lambda: self.mb and self.mb.stop()).grid(row=0, column=2)
        presets = ttk.Frame(typ)
        presets.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        for name, text in PRESETS:
            ttk.Button(presets, text=name, style="Small.TButton",
                       command=lambda t=text: self.mb and self.mb.say(t)).pack(side="left", padx=(0, 4))

        # said
        said = ttk.LabelFrame(left, text="Said", padding=(8, 4, 8, 8))
        said.grid(row=2, column=0, sticky="nsew")
        said.columnconfigure(0, weight=1)
        said.rowconfigure(0, weight=1)
        self.log = tk.Text(said, wrap="word", background=WHITE, relief="flat", highlightthickness=1,
                           highlightbackground=EDGE, padx=6, pady=4, font=(UI, 11), cursor="arrow")
        self.log.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(said, orient="vertical", command=self.log.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=sb.set, state="disabled")
        self.log.tag_configure("emph", font=(UI, 11, "bold"), foreground="#cc3333")
        self.log.tag_configure("gap", font=SMALL, foreground="#888888")
        self.log.tag_configure("chip", font=SMALL, foreground="#333333", background="#ecebe6")
        self.log.tag_configure("up", font=SMALL, foreground="#7a4a00", background="#ffe3b3")
        self.log.tag_configure("down", font=SMALL, foreground="#1f3f73", background="#d7e6fb")
        self.log.tag_configure("typed", foreground="#444444")
        self.log.tag_configure("sep", font=(UI, 4))

        # voice settings
        vf = ttk.LabelFrame(right, text="Voice", padding=(8, 4, 8, 8))
        vf.pack(fill="x")
        ttk.Label(vf, text="Voice:").pack(anchor="w")
        self.voice_box = ttk.Combobox(vf, state="readonly", width=34)
        self.voice_box.pack(fill="x", pady=(1, 6))
        self.voice_box.bind("<<ComboboxSelected>>", lambda e: self._set(
            voice=next(k for k, v in self.voices.items() if v == self.voice_box.get())))
        ttk.Label(vf, text="Microphone:").pack(anchor="w")
        self.mic_box = ttk.Combobox(vf, state="readonly", width=34)
        self.mic_box.pack(fill="x", pady=(1, 8))
        self.mic_box.bind("<<ComboboxSelected>>", lambda e: self._set(
            mic=self.mics[self.mic_box.current()]["name"]))
        # Windows only: where DECtalk plays (VB-Audio Virtual Cable's "CABLE Input" to be a microphone)
        self.out_label = ttk.Label(vf, text="Output (pick CABLE Input):")
        self.out_box = ttk.Combobox(vf, state="readonly", width=34)
        self.out_box.bind("<<ComboboxSelected>>", lambda e: self._set(
            output=self.outputs[self.out_box.current()]["name"]))
        # Windows, without VB-Audio Virtual Cable: installs the copy that comes with Moonbase
        self.cable_btn = ttk.Button(vf, text="Install virtual mic...", command=self._install_cable)
        self._before_sliders = ttk.Frame(vf)
        self._before_sliders.pack(fill="x")
        for key, label, lo, hi, step in SLIDERS:
            row = ttk.Frame(vf)
            row.pack(fill="x")
            ttk.Label(row, text=label).pack(side="left")
            self.values[key] = ttk.Label(row, text="", style="Dim.TLabel")
            self.values[key].pack(side="right")
            var = tk.DoubleVar()
            sc = ttk.Scale(vf, from_=lo, to=hi, variable=var,
                           command=lambda v, k=key, st=step: self._slide(k, st))
            sc.pack(fill="x", pady=(0, 5))
            sc.bind("<ButtonRelease-1>", lambda e, k=key, st=step: self._slide(k, st, commit=True))
            sc.bind("<KeyRelease>", lambda e, k=key, st=step: self._slide(k, st, commit=True))
            self.vars[key] = var
        for key, label in CHECKS:
            var = tk.BooleanVar()
            ttk.Checkbutton(vf, text=label, variable=var,
                            command=lambda k=key: self._set(**{k: self.vars[k].get()})).pack(anchor="w", pady=1)
            self.vars[key] = var

        bf = ttk.LabelFrame(right, text="Your usual voice (learned)", padding=(8, 4, 8, 8))
        bf.pack(fill="x", pady=(8, 0))
        self.base_lbls = {}
        for i, k in enumerate(["Pitch", "Movement", "Pace", "Loudness"]):
            ttk.Label(bf, text=k, style="Dim.TLabel").grid(row=i, column=0, sticky="w", padx=(0, 12))
            self.base_lbls[k] = ttk.Label(bf, text="-")
            self.base_lbls[k].grid(row=i, column=1, sticky="w")
        ttk.Label(bf, text="Each phrase is compared with this, so talk\nnormally for a few sentences first.",
                  style="Dim.TLabel", font=SMALL).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # status bar
        bar = tk.Frame(r, background=FACE, highlightthickness=0)
        bar.grid(row=2, column=0, sticky="ew")
        self.status_lbl = tk.Label(bar, text="", anchor="w", background=FACE, font=SMALL, relief="sunken",
                                   bd=1, padx=4)
        self.status_lbl.pack(side="left", fill="x", expand=True, padx=(2, 1), pady=2)
        self.device_lbl = tk.Label(bar, text="Output device: -", background=FACE,
                                   font=SMALL, relief="sunken", bd=1, padx=6)
        self.device_lbl.pack(side="right", padx=(1, 2), pady=2)

    def _draw_banner(self):
        c = self.banner
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        for y in range(h):
            c.create_line(0, y, w, y, fill=gradient(LUNA, y / max(h - 1, 1)))
        c.create_text(13, h // 2 + 1, text="Moonbase Alpha - DECtalk", anchor="w", fill="#0f1089",
                      font=(UI, 12, "bold"))
        c.create_text(12, h // 2, text="Moonbase Alpha - DECtalk", anchor="w", fill=WHITE,
                      font=(UI, 12, "bold"))
        c.create_window(w - 8, h // 2, window=self.listen_btn, anchor="e")

    def _level(self):
        return float(round(self.vars["voice_level"].get()))

    def _draw_meter(self, db=None, speaking=None):
        """Input level as XP progress blocks: grey below the voice detection level (ignored), green above
        it (orange while you are speaking); the red line is the level."""
        if db is not None:
            self._meter_now = (db, speaking)
        db, speaking = self._meter_now
        c = self.meter
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        pct = lambda d: max(0.0, min(1.0, (d + 70) / 70))
        thr = self._level()
        tx = 2 + int((w - 4) * pct(thr))
        fill = int((w - 4) * pct(db))
        on = ("#acedad", "#2ed330") if not speaking else ("#ffe8b0", "#f0a830")
        for x in range(2, 2 + fill - 7, 10):       # XP progress bar blocks
            hi, mid = ("#e4e4e4", "#b8b8b8") if x + 4 < tx else on
            c.create_rectangle(x, 2, x + 8, h - 3, fill=mid, width=0)
            c.create_rectangle(x, 2, x + 8, 2 + (h - 5) // 3, fill=hi, width=0)
        c.create_line(tx, 0, tx, h, fill="#cc0000", width=2)

    def _meter_drag(self, e):
        w = self.meter.winfo_width()
        v = max(-70.0, min(-10.0, round(-70 + 70 * (e.x - 2) / max(w - 4, 1))))
        self.vars["voice_level"].set(v)
        self.values["voice_level"].configure(text=self._fmt(v, 1))
        self._draw_meter()

    # ---- engine ----
    def _load(self):
        try:
            import moonbase
            mb = moonbase.Moonbase(self.events.put)
        except Exception as e:     # shown in the window; nothing else can report it
            self.events.put(dict(t="fatal", text=str(e)))
            return
        self.events.put(dict(t="ready", mb=mb))

    def _poll(self):
        level = None
        try:
            while True:
                m = self.events.get_nowait()
                if m["t"] == "level":
                    level = m         # only the latest one matters
                else:
                    self._handle(m)
        except queue.Empty:
            pass
        if level:
            self._handle(level)
        if self.closing and not self.loading:
            self._shutdown()
            return
        if self.dead:
            return
        self.root.after(40, self._poll)

    def _handle(self, m):
        t = m["t"]
        if t == "ready":
            self.mb, self.loading = m["mb"], False
            self._apply_state(self.mb.state())
        elif t == "fatal":
            self.loading, self.dead = False, True
            self.root.withdraw()
            messagebox.showerror("Moonbase", f"Could not start:\n\n{m['text']}")
            self.root.destroy()
        elif t == "state":
            self._apply_state(m)
        elif t == "level":
            self._draw_meter(m["db"], m["speaking"])
            self.pitch_lbl.configure(text=f"Pitch: {m['f0']} Hz" if m["f0"] else "Pitch: -")
            self.behind_lbl.configure(text=f"Behind: {m['backlog']:.1f} s")
        elif t == "heard":
            self.heard.configure(text=m["text"])
        elif t == "said":
            self._add_said(m)
        elif t == "latency":
            mark = f"lat{m['id']}"
            if mark in self.log.mark_names():
                self.log.configure(state="normal")
                self.log.insert(mark, f" {m['ms']} ms after you ", "chip")
                self.log.configure(state="disabled")
        elif t == "saved":
            if m["error"]:
                messagebox.showerror("Moonbase", f"Could not save the MP3:\n\n{m['error']}")
            else:
                self.status(f"Saved {m['path']}")
        elif t == "cable":
            self._cable_busy = False
            self.cable_btn.state(["!disabled"])
            if m["result"] == "ok":
                self.status(self._ready)
                messagebox.showinfo("Virtual microphone", "Done. In Discord, OBS or a game, pick "
                                    '"CABLE Output (VB-Audio Virtual Cable)" as the microphone.', parent=self.root)
            elif m["result"] == "restart":
                self.status("Restart the PC to finish installing VB-Audio Virtual Cable.")
                messagebox.showinfo("Virtual microphone", "VB-Audio Virtual Cable is installed, but Windows "
                                    "needs a restart before it shows up. Restart the PC, then open Moonbase "
                                    "again.", parent=self.root)
            else:
                self.status(self._ready)
                if m["result"] != "declined":
                    messagebox.showerror("Moonbase", m["result"], parent=self.root)

    def _apply_state(self, m):
        self._applying = True
        self.settings, self.voices, self.mics = m["settings"], m["voices"], m["mics"]
        self.outputs = m["outputs"]
        s = self.settings
        for key, _, _, _, step in SLIDERS:
            self.vars[key].set(s[key])
            self.values[key].configure(text=self._fmt(s[key], step))
        for key, _ in CHECKS:
            self.vars[key].set(s[key])
        self.voice_box.configure(values=list(self.voices.values()))
        self.voice_box.set(self.voices[s["voice"]])
        self.mic_box.configure(values=[x["description"] for x in self.mics])
        names = [x["name"] for x in self.mics]
        if m["mic"] in names:
            self.mic_box.current(names.index(m["mic"]))
        if self.outputs:
            if not self.out_box.winfo_manager():
                self.out_label.pack(anchor="w", before=self._before_sliders)
                self.out_box.pack(fill="x", pady=(1, 8), before=self._before_sliders)
            self.out_box.configure(values=[x["description"] for x in self.outputs])
            outs = [x["name"] for x in self.outputs]
            if m["output"] in outs:
                self.out_box.current(outs.index(m["output"]))
        if m.get("cable_offer"):
            if not self.cable_btn.winfo_manager():
                self.cable_btn.pack(anchor="w", pady=(0, 8), before=self._before_sliders)
            if not s["cable_asked"] and not self._cable_offered:
                self._cable_offered = True
                self.root.after(500, self._offer_cable)
        elif self.cable_btn.winfo_manager():
            self.cable_btn.pack_forget()
        self.device_lbl.configure(text=f"Output device: {m['output']}")
        ready = f"Ready ({m['device']}). {m['hint']}"
        if ready != getattr(self, "_ready", None):
            self._ready = ready
            self.status(ready)
        self.listen_btn.configure(text="Listening" if s["listening"] else "Muted",
                                  style="On.TButton" if s["listening"] else "TButton")
        self._show_base(m.get("baseline"))
        self._draw_meter()
        self._applying = False

    def _show_base(self, b):
        if not b:
            return
        self.base_lbls["Pitch"].configure(text=f"{b['pitch_hz']} Hz" if b["pitch_hz"] else "listening...")
        self.base_lbls["Movement"].configure(text=f"{b['spread_st']} semitones" if b["spread_st"] else "-")
        self.base_lbls["Pace"].configure(text=f"{b['pace_sps']} syllables/s" if b["pace_sps"] else "-")
        self.base_lbls["Loudness"].configure(text=f"{b['loud_db']} dBFS" if b["loud_db"] is not None else "-")

    def _add_said(self, m):
        lg = self.log
        lg.configure(state="normal")
        lg.mark_set("ins", "1.0")
        lg.mark_gravity("ins", "right")

        def put(text, *tags):
            lg.insert("ins", text, tags)

        plain = ("typed",) if m["kind"] == "typed" else ()
        for i, w in enumerate(m["words"]):
            if i < len(m["pauses"]) and m["pauses"][i] >= 0.15:
                put(f"|{m['pauses'][i]:.1f}s| ", "gap")
            put(w, *(("emph",) if i in m["emphasis"] else plain))
            put(" ")
        if m.get("question"):
            put("?")
        put("\n")
        if m["kind"] == "speech":
            sgn = lambda v, unit, d: f"{'+' if v > 0 else ''}{v:.{d}f}{unit}"
            for text, up, down in [(f"pitch {sgn(m['pitch_st'], ' st', 1)}", m["pitch_st"] > 1, m["pitch_st"] < -1),
                                   (f"movement x{m['range_x']:.2f}", m["range_x"] > 1.3, m["range_x"] < 0.75),
                                   (f"pace x{m['pace_x']:.2f}", m["pace_x"] > 1.15, m["pace_x"] < 0.87),
                                   (f"loud {sgn(m['loud_db'], ' dB', 1)}", m["loud_db"] > 2, m["loud_db"] < -2)]:
                put(f" {text} ", "up" if up else "down" if down else "chip")
                put(" ")
        put(f" {m['rate']} wpm, {m['ap']} Hz, range {m['pr']}% ", "chip")
        put(" ")
        put(f" {m['seconds']} s ", "chip")
        if m.get("ms") is not None:
            put(" ")
            put(f" {m['ms']} ms after you ", "chip")
        lg.mark_set(f"lat{m['id']}", "ins")
        lg.mark_gravity(f"lat{m['id']}", "left")
        put("\n\n", "sep")
        if int(lg.index("end-1c").split(".")[0]) > 400:
            lg.delete("400.0", "end")
        lg.configure(state="disabled")
        lg.yview_moveto(0)
        if m.get("baseline"):
            self._show_base(m["baseline"])

    # ---- actions ----
    @staticmethod
    def _fmt(v, step):
        return f"{v:.0f}" if step >= 1 else f"{v:.2f}".rstrip("0").rstrip(".")

    def _slide(self, key, step, commit=False):
        if self._applying:
            return
        v = round(self.vars[key].get() / step) * step
        self.values[key].configure(text=self._fmt(v, step))
        if key == "voice_level":
            self._draw_meter()             # the red line on the meter follows the slider
        if commit:
            self._set(**{key: v})

    def _set(self, **changes):
        if self._applying or not self.mb:
            return
        # monitor/mic changes call pactl / restart pw-record: keep the window responsive
        threading.Thread(target=self.mb.update, args=(changes,), daemon=True).start()

    def _toggle_listen(self):
        if self.settings:
            self._set(listening=not self.settings["listening"])

    def _say(self, _e=None):
        text = self.entry.get().strip()
        if text and self.mb:
            self.mb.say(text)
            self.entry.delete(0, "end")

    def _say_save(self):
        """Say the typed text and save it as an MP3, to share."""
        text = self.entry.get().strip()
        if not self.mb:
            return
        if not text:
            self.status("Type what Paul should say first, then press Speak + save MP3.")
            return
        spoken = re.sub(r"\[[^\]]*\]", " ", text)         # DECtalk commands don't belong in the file name
        name = "-".join(re.findall(r"[a-z0-9']+", spoken.lower())[:6]).replace("'", "") or "moonbase"
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save as MP3", initialdir=self.save_dir, initialfile=f"{name}.mp3",
            defaultextension=".mp3", filetypes=[("MP3 audio", "*.mp3")])
        if not path:
            return
        self.save_dir = os.path.dirname(path)
        self.mb.say(text, save_to=path)
        self.entry.delete(0, "end")
        self.status(f"Saving {path}...")

    def _offer_cable(self):
        """Asked once, on the first start without the cable."""
        self._set(cable_asked=True)
        self._install_cable()

    def _install_cable(self):
        """Windows: install VB-Audio Virtual Cable, so other programs can hear Paul as a microphone."""
        if not self.mb or self._cable_busy:
            return
        if not messagebox.askyesno("Virtual microphone", (
                "Install VB-Audio Virtual Cable, so Discord, OBS and games can hear Paul as a microphone?\n\n"
                'Windows asks for permission first. Then click "Install Driver" in VB-Audio\'s installer.\n\n'
                "VB-CABLE is made by VB-Audio (www.vb-cable.com). It is donationware: free to use, and "
                "donations to its makers are welcome."), parent=self.root):
            return
        self._cable_busy = True
        self.cable_btn.state(["disabled"])
        self.status('Installing VB-Audio Virtual Cable: allow it, then click "Install Driver"...')
        threading.Thread(target=self.mb.install_cable, daemon=True).start()

    def status(self, text):
        self.status_lbl.configure(text=text)

    def quit(self):
        if self.closing:
            return
        self.closing = True
        self.root.withdraw()
        if not self.loading:
            self._shutdown()

    def _shutdown(self):
        try:
            if self.mb:
                self.mb.close()    # removes the virtual mic
        finally:
            self.root.destroy()


def _single_instance():
    """True if no other Moonbase is running. The lock is held until the process exits."""
    global _lock
    if WINDOWS:
        import ctypes
        _lock = ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\MoonbaseAlphaDECtalk")
        return ctypes.windll.kernel32.GetLastError() != 183      # ERROR_ALREADY_EXISTS
    import fcntl
    _lock = open(os.path.join(os.environ.get("XDG_RUNTIME_DIR") or "/tmp", "moonbase.lock"), "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        import moonbase
        sys.exit(moonbase.selftest(sys.argv[2]))
    root = tk.Tk(className="Moonbase")
    if not _single_instance():
        root.withdraw()
        messagebox.showinfo("Moonbase", "Moonbase is already running.")
        return
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
