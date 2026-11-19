# DECtalk Voice Changer

Talk into your mic and DECtalk (Perfect Paul, the Moonbase Alpha voice) says it, live, as a microphone
that Discord, OBS and games can use. Speech recognition runs on your own PC, and nothing is sent online.
The app is called **Moonbase**, and it runs on Windows and Linux.

![The Moonbase window: mic level meter, what it heard, what Paul said, and the voice settings](docs/screenshot.png)

## Windows

1. Download `Moonbase-windows.zip` from the [latest release](https://github.com/kittyslop/dectalk-voicechanger/releases/latest)
   and unzip it anywhere. Keep the folder together.
2. If you have an NVIDIA graphics card, also download `Moonbase-windows-GPU-pack.zip` and unzip it into the
   same place, so its `Moonbase` folder merges with the first one. Speech recognition then runs on the
   graphics card and is faster. Without the pack it runs on the processor, which works on any PC.
3. Start `Moonbase.exe`. If Windows warns that the publisher is unknown, click "More info", then "Run anyway".
4. The first time, Moonbase offers to install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/), which
   is what makes Paul a microphone. Click Yes, allow it when Windows asks, then click "Install Driver".
5. In Discord, OBS or a game, pick **CABLE Output (VB-Audio Virtual Cable)** as the microphone.

The zip includes a guide, `README.txt`. Settings are kept in `%APPDATA%\Moonbase`.

## Linux

You need PipeWire (`pactl`, `pw-record`), an NVIDIA GPU with the CUDA 13 runtime (`pacman -S cuda`), the
system Python with Tk (`pacman -S tk`), `uv`, `curl`, `git`, and a C toolchain (`gcc`, `make`, `autoconf`,
`automake`) to build DECtalk.

```
./setup.sh   # once: builds DECtalk -> vendor/, Parakeet speech model (~1.1 GB) -> models/, deps -> .venv/,
             # and adds "Moonbase Alpha" to the app launcher
```

Then open **Moonbase Alpha** from the app launcher, and pick **Moonbase Alpha (DECtalk)** as the microphone
in other apps. Closing the window stops it and removes the virtual mic. Only one copy runs at a time.
Everything lives in this folder, and `settings.json` holds the window's settings. To remove the launcher
entry, run `rm ~/.local/share/applications/moonbase.desktop`.

## What Paul does

Each word is spoken as soon as you've said it. By default Paul keeps his own flat delivery, like in the
game, except for stressed words and questions. If you raise the "Follow my ..." sliders (0 by default), he
also follows how you said it: your voice is compared with your usual one, which Moonbase learns as you
talk, and DECtalk is told to match:

| You | DECtalk |
|-----|---------|
| higher / lower pitch on this word | average pitch `[:dv ap]` |
| animated / monotone | pitch range `[:dv pr]` |
| faster / slower | `[:rate]`, and each word is fitted into the time you took |
| louder / quieter | gain on the samples |
| a word you stress (pitch peak + loudness) | emphatic stress `["]` |
| rising ending | question intonation `?` |

## Tips

- If keystrokes or other noise count as speech, raise "Voice detection level", or click/drag on the mic
  meter, until the red line sits above the noise but below your voice. Grey blocks are ignored and green
  ones count as speech.
- "Hear it myself" also plays Paul on your speakers or headphones. Use headphones, or the mic hears Paul.
- The typing box takes raw DECtalk input, so Moonbase Alpha songs work (`[:phoneme on]` is already set).
- "Speak + save MP3" says the typed text and saves it as an MP3 to share (44.1 kHz, at full volume whatever
  the Volume slider says). It asks where to save, starting in Downloads.

## How it works

- `app.py`: the window (Tk, Windows XP styled), launched by the `.desktop` entry or `Moonbase.exe`.
- `audio_linux.py` / `audio_windows.py`: mic in and DECtalk out (a PipeWire virtual mic on Linux; WASAPI
  and the virtual cable on Windows).
- `moonbase.py`: the mic, in unbuffered 20 ms blocks, goes to NVIDIA Parakeet TDT 0.6B v2 (sherpa-onnx). On
  the GPU it re-transcribes the current utterance every 50 ms, and one pass takes ~10-30 ms. A word is
  spoken once three passes in a row agree on it and a later word exists, or once you've been quiet for a
  moment. Words agreed on together go to DECtalk as one clause, and DECtalk speeds up to 2× your speed
  setting to keep up. At a pause (0.3 s) the rest is spoken, with the sentence ending on the last word.
  The utterance and the model's word times are then used to learn your usual voice. On the CPU it
  re-transcribes every 150 ms and needs two agreeing passes.
- `prosody.py`: per-10 ms loudness and pitch (YIN), measurement against your baseline, the DECtalk text
  with in-line commands, and per-word gain.
- `dectalk.py`: the DECtalk engine in-process through ctypes (speech-to-memory, index marks).
  The engine is [dectalk/dectalk](https://github.com/dectalk/dectalk), built from source at a pinned
  commit. No prebuilt engine binaries are downloaded.

Measured on recorded speech (two LibriSpeech clips, 66 words, so a rough figure), on the GPU: you hear a
word about 1.2 s after you start saying it, and 4-6 of the 66 words come out wrong. Now and then the model
invents a few words at the end of what it has heard so far ("which is the only thing that"), and they get
spoken.

On Linux, cuDNN and cuBLAS come from NVIDIA's Python packages inside `.venv/`. The sherpa-onnx CUDA wheel is
the maintainer's build from Hugging Face (`csukuangfj2/sherpa-onnx-wheels`), pinned by URL and hash in
`uv.lock`.

### Building the Windows version

`.github/workflows/windows.yml` builds it on a GitHub Windows runner. Run it from the Actions tab, or push a
`v*` tag. The workflow:
- builds DECtalk with Visual Studio from the same pinned commit
- packages the app with PyInstaller
- runs `Moonbase.exe --selftest`, which checks DECtalk, recognition and MP3 saving (the runner has no audio
  devices)
- publishes a release with two zips

The two zips are needed because a release file can be at most 2 GiB:
- `Moonbase-windows.zip`: the app, DECtalk, the speech model and VB-Audio's cable package.
- `Moonbase-windows-GPU-pack.zip`: NVIDIA's CUDA/cuDNN DLLs and ONNX Runtime's CUDA plugin.

Windows can't create a microphone by itself, so DECtalk is played into VB-Audio Virtual Cable. VB-Audio's
package ships unmodified in `vbcable\`. Their licence allows this, as long as the app and guide say where
it comes from and that it's donationware. When the cable is missing, Moonbase offers to run VB-Audio's
installer, then rescans the audio devices and switches to the cable. The guide shipped in the zip is
`packaging/windows/README.txt`. Everything the build fetches is pinned by hash.

## Credits

- [DECtalk](https://github.com/dectalk/dectalk), the speech synthesizer behind Moonbase Alpha's voice.
- [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (CC-BY-4.0) for speech
  recognition, run with [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
- [VB-CABLE](https://vb-cable.com) by VB-Audio, the virtual microphone on Windows. It is donationware: free
  to use, and donations to its makers are welcome.
