#!/usr/bin/env bash
# One-time setup: DECtalk built from source into vendor/, speech model into models/, Python deps into .venv/.
# Everything stays inside this folder. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

for tool in uv curl sha256sum tar stdbuf git autoreconf automake make gcc pactl pw-record nvidia-smi ffmpeg; do
  command -v "$tool" >/dev/null || { echo "missing: $tool"; exit 1; }
done
/usr/bin/python3 -c "import tkinter" 2>/dev/null || { echo "missing: Tk for the system Python (pacman -S tk)"; exit 1; }
ldconfig -p | grep -q "libcudart.so.13 " || { echo "missing: CUDA 13 runtime (pacman -S cuda)"; exit 1; }

fetch() {   # url sha256 file
  if [ -f "$3" ] && echo "$2  $3" | sha256sum -c --quiet 2>/dev/null; then return; fi
  echo "downloading $(basename "$3")"
  curl -fL --retry 3 -o "$3.part" "$1"
  echo "$2  $3.part" | sha256sum -c --quiet || { echo "checksum mismatch: $3"; rm -f "$3.part"; exit 1; }
  mv "$3.part" "$3"
}

# DECtalk from source: github.com/dectalk/dectalk, pinned commit. Only speech-to-memory is used, so system
# audio output is left out of the build.
DT_COMMIT=986ea12ec8dac85881394d7bd7daf1f1d5098b9a
if [ ! -f vendor/src/dectalk/dist/lib/libtts_us.so ]; then
  rm -rf vendor/src/dectalk
  mkdir -p vendor/src
  git init -q vendor/src/dectalk
  git -C vendor/src/dectalk fetch -q --depth 1 https://github.com/dectalk/dectalk.git "$DT_COMMIT"
  git -C vendor/src/dectalk checkout -q FETCH_HEAD
  [ "$(git -C vendor/src/dectalk rev-parse HEAD)" = "$DT_COMMIT" ] || { echo "dectalk commit mismatch"; exit 1; }
  echo "building DECtalk (~1 min)"
  (cd vendor/src/dectalk/src && ./autogen.sh >/dev/null 2>&1 && ./configure --disable-audio >/dev/null &&
    make -j"$(nproc)" >/dev/null 2>&1) || { echo "DECtalk build failed"; exit 1; }
fi

# NVIDIA Parakeet TDT 0.6B v2 (English), fp16 for the GPU, sherpa-onnx export. No checksum is published for
# this file; this is the hash of the copy that was checked when it was added.
M=sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-fp16
mkdir -p models
if [ ! -f "models/$M/encoder.fp16.onnx" ]; then
  fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$M.tar.bz2" \
    37f67a1a6c942dae27d345ee395fbd19e25ee48996faf70fca25779026054cf0 "models/$M.tar.bz2"
  tar -xjf "models/$M.tar.bz2" -C models
  rm "models/$M.tar.bz2"
fi

uv sync

# App launcher entry
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
cat > "$APPS/moonbase.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Moonbase Alpha
GenericName=DECtalk voice
Comment=Talk and DECtalk says it through a virtual microphone
Exec="$PWD/.venv/bin/python" "$PWD/app.py"
Icon=$PWD/icon.png
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass=Moonbase
EOF
command -v update-desktop-database >/dev/null && update-desktop-database -q "$APPS" || true
echo "ready: open \"Moonbase Alpha\" from the app launcher"
