# Reference Projects & Papers

3D face reconstruction references used in the FaceForge pipeline.

---

## MICA — Towards Metrical Reconstruction of Human Faces
**ECCV 2022** · [GitHub](https://github.com/Zielon/MICA) · [Project Page](https://zielon.github.io/mica/)

Single-image metrical 3D face reconstruction invariant to expression, pose, and illumination.

**Key techniques:**
- ArcFace identity encoder (pretrained on Glint360K) → 512-dim embedding
- MappingNetwork: identity embedding → 300-dim FLAME shape coefficients
- Metric shape prediction (absolute scale, not relative)
- Supervised on paired 2D image + 3D scan datasets (~2315 subjects)

---

## DECA — Detailed Expression Capture and Animation
**SIGGRAPH 2021** · [GitHub](https://github.com/YadiraF/DECA) · [Paper](https://arxiv.org/abs/2012.04012)

Reconstruct an animatable detailed 3D face from a single in-the-wild image.

**Key techniques:**
- Coarse shape + fine UV displacement map decomposition
- Detail-consistency loss to disentangle person-specific details from expression wrinkles
- Differentiable renderer for end-to-end photometric training
- Regresses shape, albedo, expression, pose, illumination from single image

---

## HiFace — High-Fidelity 3D Face Reconstruction
**ICCV 2023** · [Project Page](https://project-hiface.github.io) · [Paper](https://arxiv.org/abs/2303.11225)

High-fidelity reconstruction by explicitly modeling static (identity) and dynamic (expression) geometric details.

**Key techniques:**
- Static detail: linear combination of a learned displacement basis
- Dynamic detail: linear interpolation of two displacement maps at polarized expressions
- Vertex tension mechanism to blend static + dynamic details
- Differentiable renderer for end-to-end training on synthetic and real images
- Region-weighted loss for facial sub-regions (eyes, nose, mouth, etc.)

---

## HRN — Hierarchical Representation Network
**CVPR 2023** · [GitHub](https://github.com/youngLBW/HRN) · [Project Page](https://younglbw.github.io/HRN-homepage/) · [Paper](https://arxiv.org/abs/2302.14434)

Accurate and detailed face reconstruction from a single in-the-wild image via hierarchical geometry disentanglement.

**Key techniques:**
- Hierarchical representation: coarse 3DMM shape + fine geometric details
- De-retouching module for decoupling geometry and appearance
- Valid mask to handle occlusion interference
- Contour-aware differentiable rendering
- FaceHD-100: high-quality 3D face dataset introduced alongside the method

---

## Pixel3DMM — Versatile Screen-Space Priors
**arXiv 2025** · [GitHub](https://github.com/SimonGiebenhain/pixel3dmm) · [Project Page](https://simongiebenhain.github.io/pixel3dmm/) · [Paper](https://arxiv.org/abs/2505.00615)

Per-pixel geometric priors from a vision foundation model to constrain 3DMM optimization for single-image face reconstruction.

**Key techniques:**
- DINO foundation model as backbone for per-pixel feature extraction
- Per-pixel surface normal and UV coordinate prediction heads
- Dense constraints replace sparse landmarks → 15%+ geometric accuracy improvement
- Trained on 1000+ identities / 976K images registered to FLAME topology
- New diverse benchmark covering varied expressions, angles, and ethnicities

---

## FLAME Head Tracker
[GitHub](https://github.com/PeizhiYan/flame-head-tracker)

FLAME-based tracker for single-image reconstruction and monocular video frame-by-frame fitting.

**Key techniques:**
- Optimizes expression coefficients, jaw pose, eye pose, lighting, and camera pose per frame
- Offline fitting (not real-time); saves per-frame results as `.npz`
- Supports FLAME model with full parameter set (shape + expression + pose)
- Compatible with downstream Gaussian avatar pipelines (e.g., Gaussian Déjà-vu)
