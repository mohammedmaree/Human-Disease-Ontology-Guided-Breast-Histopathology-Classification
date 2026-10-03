Human Disease Ontology-Guided Breast Histopathology Classification
Research code for:
> **Human Disease Ontology-Guided Hierarchical Regularization for Breast Histopathology Classification**
This project investigates whether structured disease relationships from the Human Disease Ontology (DO) can provide useful
> training-time hierarchical supervision for eight-class breast histopathology classification on BreaKHis v1.
The implementation uses It uses a conventional ImageNet-pretrained ResNet-50 and adds a fixed ontology-derived
> auxiliary objective:
[
L = L_{CE} + \lambda_{DO}L_{DO}.
]
The ontology is used during training only. Inference remains image-only.
---
Study at a glance
Dataset	BreaKHis v1
Images	7,909
Independent cases	82
Classes	8 histopathology subtypes
Magnifications	40×, 100×, 200×, 400×
Visual backbone	ResNet-50, ImageNet pretrained
Input size	224 × 224
Training	End-to-end fine-tuning
Optimizer	AdamW
Initial learning rate	3e-4
Weight decay	1e-4
Batch size	32
Epochs	20 fixed epochs
LR schedule	Cosine annealing
Ontology neighbor mass	0.15
Ontology loss weight	0.20
Outer CV	3-fold, case-grouped
Random seed	42
Negative control	Shuffled DO hierarchy, shuffle seed 123
Primary inference unit	Case
Bootstrap	5,000 paired subtype-stratified case replicates
BreaKHis classes
`adenosis`
`ductal_carcinoma`
`fibroadenoma`
`lobular_carcinoma`
`mucinous_carcinoma`
`papillary_carcinoma`
`phyllodes_tumor`
`tubular_adenoma`
Main experimental design
E0 — Image-only baseline
Standard ResNet-50 classification with the conventional supervised objective.
[
L_{E0}=L_{CE}.
]
E-DO — Correct ontology
The BreaKHis subtype labels are mapped to concepts in the supplied Human Disease Ontology resource. A fixed class-relation matrix is constructed from supported ontology relations and used to create an ontology-aware auxiliary target.
[
L_{E-DO}=L_{CE}+\lambda_{DO}L_{DO}.
]
E-DO-S — Shuffled ontology control
The same ontology-loss mechanism and relation-matrix structure are retained, but class identities are deterministically permuted before training.
This control asks whether any observed effect depends on correct semantic correspondence, rather than merely adding an auxiliary loss.
---
Important methodological choices
Case-disjoint evaluation
BreaKHis contains many images from the same case. The project therefore assigns complete case directories to folds before training.
No case is allowed to appear in both training and testing for an outer fold.
The final protocol uses:
3 outer folds;
82 independent cases;
every subtype represented in every fold;
the held-out fold is untouched until final evaluation.
Image-level and case-level evaluation
Image predictions are produced first.
For case-level inference, the class probabilities of all images belonging to the same case are averaged:
[
\bar p_{case}
\frac{1}{n_c}
\sum_{i=1}^{n_c}p_i.
]
The final case prediction is:
[
\hat y_{case}=\arg\max_j \bar p_{case,j}.
]
Because the images within a case are correlated, case-level results are the primary inferential results.
Ontology supervision
The relation matrix is fixed before model optimization.
Supported relations are assigned:
`1.0` for directly supported parent/child relations;
`0.5` for classes sharing a supported ontology ancestor;
`0.0` otherwise.
The diagonal self-relations are removed before constructing ontology neighbor supervision.
For mapped classes with supported neighbors, a small mass α=0.15 is distributed over those neighbors.
For a class with no supported ontology neighbors, no neighbor mass is introduced. Under the supplied ontology, this applies to tubular adenoma.
---
Validated BreaKHis-to-DO mapping
The experiments use the supplied `HumanDiseaseOntology.owl` and `configs/breakhis_doid_map.yaml`.
BreaKHis subtype	DOID	Mapping status
Adenosis	DOID:5998	Partial
Ductal carcinoma	DOID:3007	Exact
Fibroadenoma	DOID:1618	Exact
Lobular carcinoma	DOID:0050938	Exact
Mucinous carcinoma	DOID:3610	Exact
Papillary carcinoma	DOID:5592	Exact
Phyllodes tumor	DOID:1631	Exact
Tubular adenoma	—	Unavailable
The repository should treat the mapping YAML and supplied OWL ontology as the authoritative experimental resources.
---
Project structure
```text
Human Disease Ontology/
├── artifacts/
├── configs/
│   └── breakhis_doid_map.yaml
├── data/
├── figures/
├── knowledge/
│   └── HumanDiseaseOntology.owl
├── outputs/
│   ├── baseline_resnet50_outer_cv/
│   ├── do_hierarchy_corrected/
│   ├── do_hierarchy_shuffled_corrected/
│   └── final_analysis/
├── paper/
├── scripts/
│   ├── build_manifest.py
│   ├── make_folds.py
│   └── analyze_final_results.py
├── src/
│   ├── train_baseline_cv.py
│   ├── train_joint_cv.py
│   ├── train_hierarchy_cv.py
│   ├── knowledge/
│   │   ├── do_semantics.py
│   │   └── do_hierarchy.py
│   ├── models/
│   │   └── joint_semantic.py
│   └── utils.py
├── README.md
└── requirements.txt
```
> The raw BreaKHis images are not assumed to be committed to GitHub. Keep the large local dataset outside the repository and pass the local path to the scripts.
---
Environment
The code requires Python 3.10+ and a working PyTorch installation.
The repository contains a `requirements.txt`. For GPU execution, install a PyTorch build compatible with your NVIDIA driver/CUDA environment before installing the remaining dependencies.
Example Windows setup:
```powershell
cd "<PATH_TO_REPOSITORY>"

py -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```
Verify PyTorch and CUDA:
```powershell
python -c "import sys, torch; print('Python:', sys.executable); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE')"
```
For the conference experiments, CUDA is recommended. The scripts support `--require-cuda` to fail explicitly if a GPU is unavailable.
Dataset setup
The project expects the BreaKHis breast histology directory containing the subtype/case hierarchy.
Set the path to the local BreaKHis breast histology directory and build the image manifest:
```powershell
python scripts\build_manifest.py "<PATH_TO_BREAKHIS_BREAST_DIRECTORY>"
```
A successful run should report:
```text
wrote 7909 images, 82 case directories
```
The manifest is written to:
```text
artifacts\breakhis_manifest.csv
```
---
Build case-grouped folds
Create the final three-fold case-disjoint split:
```powershell
python scripts\make_folds.py artifacts\breakhis_manifest.csv --n_splits 3
```
The script writes:
```text
artifacts\breakhis_manifest_folds.csv
artifacts\breakhis_case_folds.csv
```
Expected properties:
```text
Images: 7909
Cases: 82
Folds: 3
Case leakage: 0
All subtypes present in every fold: YES
```
---
Build semantic records
The ontology parser is implemented in:
```text
src\knowledge\do_semantics.py
```
The hierarchy/relation construction is implemented in:
```text
src\knowledge\do_hierarchy.py
```
Build the compact semantic records:
```powershell
python -c "from pathlib import Path; from src.knowledge.do_semantics import build_semantic_records, save_semantic_records; r=build_semantic_records(Path(r'knowledge\HumanDiseaseOntology.owl'), Path(r'configs\breakhis_doid_map.yaml')); save_semantic_records(r, Path(r'artifacts\breakhis_do_semantics.json')); [print(f'\n[{k}] {v.status} | {v.doid}\n{v.semantic_text}') for k,v in sorted(r.items())]"
```
---
Run the image-only baseline
The final baseline uses the same three outer folds and fixed 20-epoch protocol as the ontology experiments.
```powershell
python -m src.train_baseline_cv `
  --manifest artifacts\breakhis_manifest_folds.csv `
  --out outputs\baseline_resnet50_outer_cv `
  --n-splits 3 `
  --epochs 20 `
  --batch-size 32 `
  --lr 3e-4 `
  --weight-decay 1e-4 `
  --seed 42 `
  --image-size 224 `
  --workers 4 `
  --require-cuda
```
Main outputs:
```text
outputs\baseline_resnet50_outer_cv\
├── fold_0\
├── fold_1\
├── fold_2\
├── fold_metrics.csv
└── aggregate.json
```
---
Run the correct-ontology model
Final E-DO configuration:
`neighbor_mass = 0.15`
`ontology_weight = 0.20`
`20` fixed epochs
same visual model/training settings as E0.
```powershell
python -m src.train_hierarchy_cv `
  --manifest artifacts\breakhis_manifest_folds.csv `
  --out outputs\do_hierarchy_corrected `
  --ontology knowledge\HumanDiseaseOntology.owl `
  --mapping configs\breakhis_doid_map.yaml `
  --mode do_hierarchy `
  --n-splits 3 `
  --epochs 20 `
  --batch-size 32 `
  --lr 3e-4 `
  --weight-decay 1e-4 `
  --neighbor-mass 0.15 `
  --ontology-weight 0.20 `
  --image-size 224 `
  --workers 4 `
  --seed 42 `
  --require-cuda
```
The final objective is:
[
L=L_{CE}+0.20L_{DO}.
]
The ontology matrix is fixed before model optimization and is not derived from the held-out fold.
Run the shuffled-ontology control
The shuffled control keeps the relation structure but permutes class identities with `shuffle_seed=123`.
```powershell
python -m src.train_hierarchy_cv `
  --manifest artifacts\breakhis_manifest_folds.csv `
  --out outputs\do_hierarchy_shuffled_corrected `
  --ontology knowledge\HumanDiseaseOntology.owl `
  --mapping configs\breakhis_doid_map.yaml `
  --mode shuffled_do_hierarchy `
  --n-splits 3 `
  --epochs 20 `
  --batch-size 32 `
  --lr 3e-4 `
  --weight-decay 1e-4 `
  --neighbor-mass 0.15 `
  --ontology-weight 0.20 `
  --image-size 224 `
  --workers 4 `
  --seed 42 `
  --shuffle-seed 123 `
  --require-cuda
```
---
Final matched statistical analysis
After E0, E-DO, and E-DO-S are complete:
```powershell
python scripts\analyze_final_results.py `
  --baseline outputs\baseline_resnet50_outer_cv `
  --do outputs\do_hierarchy_corrected `
  --shuffled outputs\do_hierarchy_shuffled_corrected `
  --out outputs\final_analysis `
  --n-splits 3 `
  --bootstrap 5000 `
  --seed 42
```
The analysis script:
verifies identical held-out images across models;
computes image-level metrics;
aggregates probabilities within cases;
computes case-level metrics;
performs paired subtype-stratified case bootstrap;
reports percentile 95% CIs for paired metric differences.
Key outputs:
```text
outputs\final_analysis\
├── image_level_fold_metrics.csv
├── case_level_fold_metrics.csv
├── image_level_aggregate.csv
├── case_level_aggregate.csv
├── pooled_out_of_fold_metrics.csv
├── case_stratified_paired_bootstrap.csv
└── analysis_metadata.json
```
---
Core source modules
`src/train_baseline_cv.py`
Image-only ResNet-50 baseline:
BreaKHis loading;
case-grouped evaluation;
class weighting;
augmentation;
ResNet-50 fine-tuning;
out-of-fold predictions;
standard metrics.
`src/train_hierarchy_cv.py`
E-DO and E-DO-S implementation:
DO hierarchy parsing;
fixed relation matrix construction;
ontology-aware auxiliary targets;
objective `CE + ontology_weight * DO_hierarchy_loss`;
shuffled semantic control;
fold predictions and metrics.
`src/train_joint_cv.py`
Additional semantic-prototype research module.
This module is retained for future experimentation and is not required for the final conference E0/E-DO/E-DO-S results.
`src/knowledge/do_semantics.py`
Parses the supplied OWL ontology and extracts:
DO identifiers;
labels;
synonyms;
definitions;
parents;
ancestors;
verified subtype mappings.
`src/knowledge/do_hierarchy.py`
Constructs the task-level ontology relation matrix and computes ontology-aware hierarchical supervision.
`scripts/analyze_final_results.py`
Performs the final matched image/case analysis and paired subtype-stratified bootstrap.
---
Reproducibility
The reproducibility configuration is:
```text
seed = 42
shuffle_seed = 123
n_splits = 3
epochs = 20
batch_size = 32
image_size = 224
lr = 3e-4
weight_decay = 1e-4
neighbor_mass = 0.15
ontology_weight = 0.20
```
The test fold is never used for model selection.
The final inferential analysis treats the 82 cases, rather than the 7,909 correlated images, as the independent evaluation units.
Experimental results
The pooled case-level results are:
Model	Accuracy	Balanced Acc.	Macro-F1	Weighted-F1	Macro-AUC
E0	0.6220	0.4385	0.4486	0.6105	0.7465
E-DO	0.6220	0.4385	0.4452	0.6049	0.7814
E-DO-S	0.6098	0.4206	0.4307	0.5971	0.7359
Observed paired case-level macro-AUC differences:
```text
E-DO vs E0
Difference: +0.0349
95% CI:      [-0.0051, 0.0807]

E-DO vs E-DO-S
Difference: +0.0455
95% CI:      [-0.0029, 0.0977]
```
Both confidence intervals include zero.
GitHub repository policy
Do not commit:
```text
.venv/
__pycache__/
*.py[cod]
raw BreaKHis image files
large model checkpoints
temporary logs
machine-specific absolute paths
private tokens or credentials
```
Suggested `.gitignore`:
```gitignore
.venv/
__pycache__/
*.py[cod]
*.pyo
*.log
.ipynb_checkpoints/
.vscode/
.idea/
.env
.env.*
outputs/
*.pt
*.pth
*.ckpt
```
Large experimental outputs should be handled with Git LFS, GitHub Releases, or an archival repository rather than normal Git commits.
Data and ontology resources
BreaKHis
BreaKHis is the public breast histopathology benchmark used in this study. The raw dataset should be downloaded from its official/public distribution and kept outside the Git repository if size/licensing considerations require it.
Human Disease Ontology
The project uses the supplied:
```text
knowledge/HumanDiseaseOntology.owl
```
Redistribution of ontology resources should follow their applicable license terms.
Research scope
This repository contains the implementation used in the associated study.
The current design intentionally avoids:
graph neural networks;
a second trainable semantic encoder;
ontology input at inference time;
multi-magnification fusion;
large multimodal architectures.
These components are outside the scope of the current implementation.
The central research question is:
> **Can a fixed, publicly available disease hierarchy provide useful auxiliary supervision to a conventional histopathology classifier when evaluation is case-disjoint and a structure-preserving shuffled hierarchy is used as a negative control?**
---
Citation
When using or extending this code, please cite the associated conference paper and the original BreaKHis and Human Disease Ontology publications.
---
