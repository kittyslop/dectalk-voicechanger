MOONBASE ALPHA - DECtalk
========================

Talk into your microphone and DECtalk (Perfect Paul, the Moonbase Alpha voice) says it, matching your
pitch, loudness and pace. Other programs (Discord, OBS, games) hear Paul as a microphone.

SETUP
-----
1. Unzip Moonbase-windows.zip anywhere (for example Documents). Keep the whole folder together.

2. NVIDIA graphics card? Also unzip Moonbase-windows-GPU-pack.zip into the SAME place, so its "Moonbase"
   folder merges with the first one. Speech recognition then runs on the graphics card (faster).
   Without it, Moonbase uses the processor, which works on any PC.

3. To use Paul as a microphone, install VB-Audio Virtual Cable (free): https://vb-audio.com/Cable/
   Run its installer as administrator, then restart the PC.
   Without it, Moonbase plays Paul on your speakers only (and can still save MP3s).

4. Start Moonbase.exe. Windows may warn that the app is from an unknown publisher: click
   "More info" and "Run anyway".

USING IT
--------
- In Discord, OBS or the game, pick "CABLE Output (VB-Audio Virtual Cable)" as the microphone.
- In Moonbase, "Output" should say "CABLE Input". "Microphone" is your real microphone.
- Talk normally for a few sentences first: Moonbase learns your usual voice and follows how you talk.
- Keys or other noise counting as speech? Raise "Voice detection level", or click/drag on the meter at
  the top: the red line should sit above the noise but below your voice.
- "Hear it myself" also plays Paul on your speakers or headphones (use headphones).
- Type in "Type to Paul" to make him say anything. "Speak + save MP3" saves it as an MP3 to share.

The status bar at the bottom says whether speech recognition runs on the GPU or the CPU.
Settings are kept in %APPDATA%\Moonbase.
