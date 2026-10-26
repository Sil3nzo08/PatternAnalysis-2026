# 2D CAN for HipMRI Prostate Segmentation 

## 1. Overview
The problem I am investigating is that manual contouring is extremely time-consuming and has high stake consequences.
The imaging specialist has to precisely outline where the prostate is on HipMRI images, so the machine knows exactly
where to aim. Under-contouring will risk tumour recurrence, and over-contouring can cause severe radiation toxicity.
We want to avoid harming the OARs, such as the bladder, in the process. This job is also very difficult since the 
boundaries are very visually ambiguous on the HipMRI scans, and the prostate makes up a small percentage of the
pelvic scan.

Consequently, a model has been created to serve as an *annotation assistant* that can save clinical contouring time by 
proposing anatomical contours of the HipMRI scans. It proposes a draft segmentation of each HipMRI slice for the
specialist to review and correct. The underlying structure of the model is a 2D Context Aggregation Network (2D CAN).
It uses dilated convolutions to expand the receptive field exponentially across layers, aggregating multi-scale 
contextual information without downsampling feature maps or rescaling the input (REFERENCE). This is a
completely different approach to the U-Net, which relies on successive pooling and subsampling layers to gain 
context, and then recovers resolution through its decoder and skip connections. 

![Model architecture]

## 2. Feasibility Review
Your one-page review from the midpoint check-off, as its own section:
user need and acceptance criteria, model choice, preliminary evidence, risks and fallback.

## 3. Repository Structure and Usage
- `modules.py`, `dataset.py`, `train.py`, `predict.py`: one line each.
- Commands to train and predict, e.g. `python train.py --epochs ...`
- Example input and output (a figure of the four-panel result).

## 4. Dependencies and Reproducibility
Exact versions (Python, PyTorch, nibabel, numpy, matplotlib), random seeds, hardware used, and any Slurm script or command.

## 5. Data and Pre-processing
- Dataset and label order (0 background, 1 body, 2 bone, 3 bladder, 4 rectum, 5 prostate).
- Normalisation, one-hot encoding, augmentation.
- **Split justification:** patient-level split, sizes of train/val/test, and why it prevents leakage.

## 6. Investigation of the Open Research Dilemma
### 6.1 Baseline vs CAN
Table of per-class Dice and IoU on the same test split, for both models.

### 6.2 Resource Profiling
| Model | Parameters | Peak VRAM | Inference time/slice | Training time |
|---|---|---|---|---|

### 6.3 Dice vs Slice Position
The apex-to-base plot for both models, with a short interpretation.

### 6.4 Failure Case Autopsies
One subsection per case (4 of them): the overlay figure with a colour legend, a Dice / FP / FN table, the root-cause diagnosis, and the CAN vs U-Net comparison.

### 6.5 Recommendation to Project Manager
Does CAN's extra cost justify its gain? Which slices need human review? Back it with the numbers above.

## 7. Artificial Intelligence Usage Disclosure
Which tools you used, what tasks they helped with, and how you verified the output (tests, checking against the labels, etc.).

## 8. References
[1] ... [2] ... Also cite any code you adapted, including the Appendix B Nifti loader.
