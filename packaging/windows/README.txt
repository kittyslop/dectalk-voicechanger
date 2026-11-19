MOONBASE ALPHA - DECtalk
========================

Talk into your microphone and DECtalk (Perfect Paul, the Moonbase Alpha voice) says it. Other programs
(Discord, OBS, games) hear Paul as a microphone.

SETUP
-----
1. Unzip Moonbase-windows.zip anywhere (for example Documents). Keep the whole folder together.

2. NVIDIA graphics card? Also unzip Moonbase-windows-GPU-pack.zip into the SAME place, so its "Moonbase"
   folder merges with the first one. Speech recognition then runs on the graphics card (faster).
   Without it, Moonbase uses the processor, which works on any PC.

3. Start Moonbase.exe. Windows may warn that the app is from an unknown publisher: click
   "More info" and "Run anyway".

4. The first time, Moonbase offers to install VB-Audio Virtual Cable: that's what lets Discord, OBS and
   games hear Paul as a microphone. Click Yes, allow it when Windows asks, then click "Install Driver"
   in VB-Audio's installer. If Moonbase then says so, restart the PC.
   Said no? The "Install virtual mic..." button in Moonbase does it later. Without the cable, Paul plays
   on your speakers only (and you can still save MP3s).

USING IT
--------
- In Discord, OBS or the game, pick "CABLE Output (VB-Audio Virtual Cable)" as the microphone.
- In Moonbase, "Output" should say "CABLE Input". "Microphone" is your real microphone.
- Paul talks flat, like in the game. To make him follow your pitch, expressiveness, pace or loudness, raise
  the "Follow my ..." sliders (talk normally for a few sentences first: Moonbase learns your usual voice).
- Keys or other noise counting as speech? Raise "Voice detection level", or click/drag on the meter at
  the top: the red line should sit above the noise but below your voice.
- "Hear it myself" also plays Paul on your speakers or headphones (use headphones).
- Type in "Type to Paul" to make him say anything. "Speak + save MP3" saves it as an MP3 to share.

The status bar at the bottom says whether speech recognition runs on the GPU or the CPU.
Settings are kept in %APPDATA%\Moonbase.

CREDITS
-------
- VB-CABLE (VB-Audio Virtual Cable) is made by VB-Audio, www.vb-cable.com, and included unmodified in
  the "vbcable" folder. It is donationware: free to use, and donations to its makers are welcome.
  To remove it, run VBCABLE_Setup_x64.exe from that zip and click "Remove Driver".
- Speech recognition: NVIDIA Parakeet TDT 0.6B v2 (CC-BY-4.0), huggingface.co/nvidia/parakeet-tdt-0.6b-v2,
  run with sherpa-onnx (github.com/k2-fsa/sherpa-onnx).
- DECtalk, built from github.com/dectalk/dectalk.
