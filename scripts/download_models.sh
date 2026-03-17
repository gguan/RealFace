#!/usr/bin/env bash
# download_models.sh — Download / install all pretrained model weights for FaceForge
# Run from the repository root: bash scripts/download_models.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRETRAINED="$ROOT/data/pretrained"

echo "=== FaceForge model downloader ==="
echo "Target directory: $PRETRAINED"
mkdir -p "$PRETRAINED/FLAME2020"

# ---------------------------------------------------------------------------
# 1. FLAME 2020
# ---------------------------------------------------------------------------
# FLAME 2020 requires accepting a licence agreement.
# 1. Visit https://flame.is.tue.mpg.de/ and create a free account.
# 2. Download "FLAME 2020" (generic_model.pkl) and "FLAME_masks.pkl".
# 3. Place the files at:
#      data/pretrained/FLAME2020/generic_model.pkl
#      data/pretrained/FLAME2020/FLAME_masks.pkl
echo ""
echo "[FLAME 2020] Manual download required."
echo "  -> https://flame.is.tue.mpg.de/"
echo "  -> Place files in: $PRETRAINED/FLAME2020/"

# ---------------------------------------------------------------------------
# 2. MICA weights
# ---------------------------------------------------------------------------
# MICA requires accepting a separate licence.
# 1. Visit https://github.com/Zielon/MICA and follow the instructions.
# 2. Download the pretrained checkpoint (mica.tar).
# 3. Place it at: data/pretrained/mica.tar
echo ""
echo "[MICA] Manual download required."
echo "  -> https://github.com/Zielon/MICA"
echo "  -> Place file at: $PRETRAINED/mica.tar"

# ---------------------------------------------------------------------------
# 3. InsightFace buffalo_l (auto-downloads on first use)
# ---------------------------------------------------------------------------
echo ""
echo "[InsightFace buffalo_l] Will auto-download on first run of FaceForge."
echo "  (Requires internet access during first inference.)"

# ---------------------------------------------------------------------------
# 4. PyTorch3D (source install — binary wheels not always available)
# ---------------------------------------------------------------------------
echo ""
echo "[PyTorch3D] Install from source:"
echo "  pip install 'git+https://github.com/facebookresearch/pytorch3d.git'"
echo "  OR follow: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"

# ---------------------------------------------------------------------------
# 5. chumpy (legacy FLAME dependency)
# ---------------------------------------------------------------------------
echo ""
echo "[chumpy] Installing from GitHub..."
pip install "git+https://github.com/mattloper/chumpy" --quiet && echo "  chumpy installed." || echo "  chumpy install failed — install manually."

echo ""
echo "=== Done. Check messages above for any required manual steps. ==="
