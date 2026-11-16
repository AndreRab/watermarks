# Tree-Ring Watermarks in Google Colab

Open `tree_ring_colab.ipynb` in [Google Colab](https://colab.research.google.com/) using **File → Upload notebook**. Select **Runtime → Change runtime type → GPU**, then choose **Run all** in a fresh runtime.

The notebook includes:

- SD 2.1-base from a public mirror at a fixed commit, downloading only FP16 safetensors.
- Tree-Ring key construction, initial-noise spectrum visualization, and paired image generation.
- Image-based detection through empty-prompt DDIM inversion, with the paper's L1 metric and model-based p-value.
- Unmarked-image and wrong-key controls.
- JPEG, Gaussian blur, crop+resize, Gaussian noise, and rotations of 75° and 180°, testing both branches.
- Plots, an image gallery, intermediate CSV files, and a final ZIP with settings and library versions.
- An optional attack strength sweep and a small ROC/AUC benchmark, disabled by default.

The target budget is a GPU with 12–16 GB VRAM and about 12 GB system RAM. CPU offload is enabled by default; UNet and VAE use FP16, CFG is evaluated sequentially, and images are processed one at a time. Actual memory usage is reported during execution; a full run on this configuration has not yet been measured. Weights download into the Colab virtual machine and are not needed on your Mac.

For the first run, keep the defaults: 512×512, 50 generation and detection steps, guidance 7.5, radius 10, and channel 3. Fewer steps reduce execution time but barely reduce peak VRAM. The first run downloads several GB and may take tens of minutes. The remaining attack time is estimated after the first inversions.

If installation updates packages you have already imported, restart the runtime and continue with section 2. Each run saves its results in a separate folder with a UTC timestamp. Download the ZIP through the **Files** sidebar before the VM disconnects, or set `DOWNLOAD_ZIP=True` to request a download automatically.

## Validation before delivery

Checks covered the `.ipynb` schema, compilation of every code cell, equivalence of the adapted mask/key/embedding/L1 metric to the authors' functions, DDIM algebra, and execution of all computational cells using tiny randomly initialized CPU models. The latter checks library calls, attacks, plots, optional branches, and ZIP export, but **does not verify SD 2.1 detection accuracy or measure GPU memory**. The pinned ML libraries were installed in a separate temporary environment. Full SD loading and execution in Colab have not been tested. After translation, the notebook schema, code compilation, and unchanged executable structure were checked again.

This is a teaching adaptation of [Tree-Ring Watermarks](https://arxiv.org/abs/2305.20030), not a full reproduction of the paper's tables. The notebook documents the differences and includes the license for fragments adapted from the [authors' implementation](https://github.com/YuxinWenRick/tree-ring-watermark). `ALPHA=0.01` sets the model-based test threshold; it does not establish an actual 1% FPR on a small sample.
