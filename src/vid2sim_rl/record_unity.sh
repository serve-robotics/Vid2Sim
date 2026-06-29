#!/bin/bash
# Record the HIGH-QUALITY Unity window — run this on the HOST, not in Docker.
#
# demo.py only records the agent's policy-input camera, which the Unity build
# renders at a fixed 128x72 (baked into the binary's camera sensor). This script
# instead screen-captures the actual Unity render window (e.g. 1280x720), which
# is the good-looking view you see on screen.
#
# One-time host setup:
#     sudo apt-get install -y ffmpeg xdotool
#
# Usage (launch Unity first, then run this in another host terminal):
#     ./record_unity.sh [output.mp4] [window-name]
#     # defaults: unity_capture.mp4, window name "Video2Sim"
# Press Ctrl-C to stop recording.
set -e

OUT="${1:-unity_capture.mp4}"
WIN_NAME="${2:-Video2Sim}"

for tool in ffmpeg xdotool; do
  command -v "$tool" >/dev/null || { echo "Missing '$tool'. Run: sudo apt-get install -y ffmpeg xdotool"; exit 1; }
done

WIN=$(xdotool search --name "$WIN_NAME" | head -1)
if [ -z "$WIN" ]; then
  echo "No window matching '$WIN_NAME' found. Launch the Unity env first."; exit 1
fi

xdotool windowactivate "$WIN" 2>/dev/null || true
eval "$(xdotool getwindowgeometry --shell "$WIN")"   # sets X, Y, WIDTH, HEIGHT

echo "Recording window $WIN (${WIDTH}x${HEIGHT} at +${X},${Y}) -> $OUT"
echo "Press Ctrl-C to stop."
# crf 18 = visually lossless; yuv420p for broad player compatibility.
ffmpeg -y -f x11grab -framerate 30 -video_size "${WIDTH}x${HEIGHT}" \
  -i "${DISPLAY}+${X},${Y}" \
  -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p "$OUT"
