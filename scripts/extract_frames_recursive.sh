#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <input_dir> <output_dir>"
  echo "Example: $0 /data/videos /data/frames"
}

if [[ $# -ne 2 ]]; then
  usage
  exit 1
fi

input_dir="$1"
output_dir="$2"

if [[ ! -d "$input_dir" ]]; then
  echo "Error: input directory does not exist: $input_dir" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is not installed or not in PATH." >&2
  exit 1
fi

# Normalize to absolute paths for consistent relative path handling.
input_dir="$(cd "$input_dir" && pwd)"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

# Common video extensions (case-insensitive).
while IFS= read -r -d '' video_file; do
  rel_path="${video_file#"$input_dir"/}"
  rel_dir="$(dirname "$rel_path")"
  video_filename="$(basename "$video_file")"
  video_stem="${video_filename%.*}"

  dest_dir="$output_dir/$rel_dir/$video_stem"
  mkdir -p "$dest_dir"

  echo "Extracting: $video_file"
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$video_file" "$dest_dir/%04d.jpg"
  echo "Saved frames to: $dest_dir"
done < <(
  find "$input_dir" -type f \( \
    -iname "*.mp4" -o \
    -iname "*.mov" -o \
    -iname "*.avi" -o \
    -iname "*.mkv" -o \
    -iname "*.webm" -o \
    -iname "*.m4v" -o \
    -iname "*.mpg" -o \
    -iname "*.mpeg" -o \
    -iname "*.wmv" -o \
    -iname "*.flv" \
  \) -print0
)

echo "Done."
