"""Merlin (stanfordmimi/Merlin) inference on MCT-LTDiag NIfTIs; saves raw outputs for later analysis.
Usage: python merlin_infer.py <nii_root> <out_dir>
  nii_root/<phase>/<case>.nii.gz for phase in nc/art/pvp/delay
Saves per phase: <out_dir>/merlin_<phase>.npz with
  ids (N,), emb2048 (N,2048) pooled image features, emb512 (N,512) contrastive image embedding (unnormalized),
  pheno_prob (N,1692) sigmoid of phenotype logits (row order = documentation/phenotypes.csv)
"""
import glob
import os
import sys

import numpy as np
import torch
from monai.data import DataLoader, Dataset

from merlin import Merlin
from merlin.data.monai_transforms import ImageTransforms


def main():
    nii_root, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    model = Merlin()  # default: contrastive + EHR phenotype heads
    model.eval().cuda()
    i3 = model.model.encode_image.i3_resnet
    pooled = {}
    i3.avgpool.register_forward_hook(lambda m, i, o: pooled.__setitem__("x", o))

    for phase in ["nc", "art", "pvp", "delay"]:
        files = sorted(glob.glob(os.path.join(nii_root, phase, "*.nii.gz")))
        out = os.path.join(out_dir, f"merlin_{phase}.npz")
        if not files or os.path.exists(out):
            continue
        ds = Dataset([{"image": f} for f in files], transform=ImageTransforms)
        dl = DataLoader(ds, batch_size=1, num_workers=16, shuffle=False)
        ids, e2048, e512, prob = [], [], [], []
        with torch.no_grad():
            for f, batch in zip(files, dl):
                x = batch["image"].cuda()
                contrastive, ehr = model.model.encode_image(x)
                e2048.append(pooled["x"].flatten(1).float().cpu().numpy())
                e512.append(contrastive.reshape(1, -1).float().cpu().numpy())
                prob.append(torch.sigmoid(ehr).reshape(1, -1).float().cpu().numpy())
                ids.append(os.path.basename(f).replace(".nii.gz", ""))
        np.savez(out, ids=np.array(ids), emb2048=np.concatenate(e2048), emb512=np.concatenate(e512),
                 pheno_prob=np.concatenate(prob))
        print(phase, len(ids), "saved", out, flush=True)


if __name__ == "__main__":
    main()
