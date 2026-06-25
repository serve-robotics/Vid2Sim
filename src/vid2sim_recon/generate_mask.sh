# Heuristic dynamic masks category
MASK_PROMPT="person.pedestrian.cyclist.child.adult.bag.backpack.handbag.suitcase.hat.shoes.cloth.wheelchair"

set -euo pipefail

if [[ $# -lt 1 ]]; then
	echo "Usage: $0 <sequence_dir>"
	exit 1
fi

seq_path="$(readlink -f "$1")"
deva_script="submodules/vid2sim-deva-segmentation/generate_mask.sh"

if [[ ! -d "$seq_path/images" ]]; then
	echo "Expected images directory not found: $seq_path/images"
	exit 1
fi

has_flat_images="$(find "$seq_path/images" -maxdepth 1 -type f \
	\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 1)"

# Generate dynamic masks from SAM2 (recommended)
# python tools/generate_mask_sam2.py \
# --text_prompt $MASK_PROMPT \
# --video_dir $1/images \
# --output_dir $1/tmp \
# --result_dir $1/masks

# Generate dynamic masks from DEVA
if [[ -n "$has_flat_images" ]]; then
	cd submodules/vid2sim-deva-segmentation
	bash generate_mask.sh "$seq_path" "$MASK_PROMPT"
	exit 0
fi

mapfile -t image_subdirs < <(find "$seq_path/images" -mindepth 1 -maxdepth 1 -type d | sort)
if [[ ${#image_subdirs[@]} -eq 0 ]]; then
	echo "No image files found in $seq_path/images and no subdirectories to process."
	exit 1
fi

mkdir -p "$seq_path/masks"
echo "Detected nested image folders under $seq_path/images; processing each subfolder."

max_seqs="${VID2SIM_MASK_MAX_SEQS:-0}"
processed=0

for image_dir in "${image_subdirs[@]}"; do
	sequence_name="$(basename "$image_dir")"
	first_image="$(find "$image_dir" -maxdepth 1 -type f \
		\( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | head -n 1)"

	if [[ -z "$first_image" ]]; then
		echo "Skipping $sequence_name (no images found)."
		continue
	fi

	echo "Generating masks for $sequence_name"

	temp_seq_dir="$(mktemp -d)"
	ln -s "$image_dir" "$temp_seq_dir/images"

	(
		cd submodules/vid2sim-deva-segmentation
		bash generate_mask.sh "$temp_seq_dir" "$MASK_PROMPT"
	)

	mkdir -p "$seq_path/masks/$sequence_name"
	cp -a "$temp_seq_dir/masks/." "$seq_path/masks/$sequence_name/"
	rm -rf "$temp_seq_dir"

	processed=$((processed + 1))
	if [[ "$max_seqs" -gt 0 && "$processed" -ge "$max_seqs" ]]; then
		echo "Reached VID2SIM_MASK_MAX_SEQS=$max_seqs, stopping early."
		break
	fi
done

if [[ "$processed" -eq 0 ]]; then
	echo "No valid image subfolders found under $seq_path/images."
	exit 1
fi