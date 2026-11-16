# Liver tumor differentiation with generalist CT foundation models

Code for evaluating two generalist CT foundation models, **RADAR** and **Merlin**, on the differential diagnosis of
liver tumors at multiphase contrast-enhanced CT (public MCT-LTDiag dataset; HCC, ICC, BCLM, CRLM, HH; 517 cases):

1. **Zero-shot** evaluation using the existing model outputs (RADAR finding scores, Merlin phecode probabilities).
2. **Linear probing**: L2-regularized logistic regression on frozen image features
   (stratified fivefold cross-validation repeated five times, nested selection of the regularization strength).

No data, model weights, or results are included in this repository.

## External resources

| Resource | Source |
|---|---|
| MCT-LTDiag dataset (CC0) | Harvard Dataverse, doi:10.7910/DVN/S3RW15 |
| RADAR code and weights | https://github.com/alibaba-damo-academy/damo-radar (pretrained checkpoint `checkpoint_radar_pretrain.pth`) |
| Merlin | `pip install merlin-vlm` (weights: `stanfordmimi/Merlin` on Hugging Face) |
| Merlin phecode list | `documentation/phenotypes.csv` in https://github.com/StanfordMIMI/Merlin |

## Layout

```
scripts/
  preprocess/
    download_mctltdiag.sh    # download all case tars from Dataverse (resumable); needs tar_ids.txt ("<fileId> <name>.tar")
    prep_mctlt.py            # extract 4 phases from case tars and reorient to LAS (RADAR input)
    reorient_dir.py          # parallel LAS reorientation of already-extracted NIfTIs
  inference/
    radar_feat.pbs           # RADAR inference (4 phases) saving scores + liver organ features (PBS job)
    miyabi_infer.pbs         # RADAR inference without feature saving (scores only)
    merlin_infer.py/.pbs     # Merlin inference: 2048-d / 512-d image features + 1692 phecode probabilities
  analysis/
    eval_zeroshot.py         # zero-shot: RADAR 4-class and Merlin 3-class (HH excluded) benchmark-format metrics (bootstrap 95% CI)
    linear_probe.py          # linear probing (Merlin or RADAR features), out-of-fold predictions
    eval_benchmark4.py       # linear-probe results in the dataset benchmark format (bootstrap 95% CI)
    fig_zs_vs_lp.py          # Figure 1: zero-shot vs linear-probe sensitivity/accuracy (grouped bars)
    fig_confusion.py         # confusion-matrix figure
patches/
  radar_inference_save_organ_features.patch   # minimal change to RADAR's inference_demo.py
```

## Environment

- RADAR: Python 3.10, `transformers==4.25.1`, `timm==0.4.12`, `fairscale==0.4.4`, `monai`, `SimpleITK`, `nibabel`
  (requirements of the RADAR repository; `decord` is not needed for inference). Tested with PyTorch 2.14.1 (cu126) on NVIDIA GH200 (aarch64).
- Merlin: Python 3.10, `merlin-vlm==0.0.7`. The package expects the checkpoint under `merlin/models/checkpoints/`.
- Analysis: `numpy`, `pandas`, `scikit-learn` (1.7.2 tested), `matplotlib`, `torch` (to read RADAR feature files).

The PBS scripts were written for a PBS Pro GPU cluster; set `WORK_DIR` and `YOUR_GROUP` (and module names) for your site.

## Steps

```bash
export WORK_DIR=/path/to/workdir          # contains mctlt/, damo-radar/, scripts/ (copy of this repo's scripts)

# 1. Data
bash scripts/preprocess/download_mctltdiag.sh                 # -> $WORK_DIR/mctlt/tars/*.tar
grep -v '^ID,' mctlt/meta_info_patient.tab | cut -d, -f1,2 > mctlt/labels_all.csv   # case id, tumor type
#    raw NIfTI per phase (used by Merlin): mctlt/nii_raw/{nc,art,pvp,delay}/<case>.nii.gz  (see merlin_infer.pbs)

# 2. RADAR (scores + liver features)
cd damo-radar/RADAR_inference && cp inference_demo.py inference_feat.py \
  && patch inference_feat.py < ../../patches/radar_inference_save_organ_features.patch && cd -
qsub scripts/inference/radar_feat.pbs       # reorients to LAS, writes results_radar_feat/

# 3. Merlin (features + phecode probabilities)
qsub scripts/inference/merlin_infer.pbs     # writes results_merlin/merlin_{phase}.npz

# 4. Analysis (run from WORK_DIR; outputs go to results/)
python scripts/analysis/eval_zeroshot.py mctlt/labels_all.csv results_radar_feat results_merlin phenotypes.csv out/
mkdir -p results/figs
export MERLIN_DIR=results_merlin RADAR_FEAT_DIR=results_radar_feat
python scripts/analysis/linear_probe.py merlin
python scripts/analysis/linear_probe.py radar
python scripts/analysis/eval_benchmark4.py
python scripts/analysis/fig_zs_vs_lp.py
python scripts/analysis/fig_confusion.py
```

## Notes

- RADAR input: volumes are reoriented from LPS (dataset) to LAS to match the official demonstration case.
- Each model accepts a single phase; phases are processed independently, and "4-phase" features are concatenations.
- Tumor masks provided with the dataset are not used.
- The RADAR code is licensed CC BY-NC-SA 4.0 by its authors; the patch in `patches/` modifies that code and is subject to the same license.

## Citation

If you use this code, please cite our preprint:

> [Authors]. How Well Do Generalist CT Foundation Models Differentiate Liver Tumors? medRxiv [year]. doi:[to be added]

```bibtex
@article{liver_tumor_ddx_ct_fm,
  title   = {How Well Do Generalist CT Foundation Models Differentiate Liver Tumors?},
  author  = {[Authors]},
  journal = {medRxiv},
  year    = {[year]},
  doi     = {[to be added]}
}
```
