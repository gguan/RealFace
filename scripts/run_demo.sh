#!/bin/bash
set -e
echo "=== FaceForge Demo ==="
if [ -z "$1" ]; then
    echo "Usage: $0 <image_path> [subject_id]"
    echo "Example: $0 input/photo.jpg alice"
    exit 1
fi
IMAGE=$1
SUBJECT=${2:-"demo"}
python -m faceforge.cli \
    --input "$IMAGE" \
    --output output/ \
    --subject "$SUBJECT" \
    --config config/mac_mps.yaml
echo "Done! Check output/ for results."
