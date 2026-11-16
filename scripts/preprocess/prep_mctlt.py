"""Extract 4 phases from MCT-LTDiag tars and reorient to LAS (RADAR demo orientation).
Usage: python prep_mctlt.py <tar_dir> <out_dir> [ids_csv]
Writes <out_dir>/<phase>/<case_id>.nii.gz for phase in nc/art/pvp/delay.
"""
import gzip
import os
import sys
import tarfile

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

PHASES = ["nc", "art", "pvp", "delay"]
TARGET = axcodes2ornt(("L", "A", "S"))

tar_dir, out_dir = sys.argv[1], sys.argv[2]
if len(sys.argv) > 3:
    ids = [l.strip().split(",")[0] for l in open(sys.argv[3]) if l.strip()]
else:
    ids = sorted(f[:-4] for f in os.listdir(tar_dir) if f.endswith(".tar"))

for p in PHASES:
    os.makedirs(os.path.join(out_dir, p), exist_ok=True)

for cid in ids:
    with tarfile.open(os.path.join(tar_dir, f"{cid}.tar")) as tf:
        for p in PHASES:
            dst = os.path.join(out_dir, p, f"{cid}.nii.gz")
            if os.path.exists(dst):
                continue
            try:
                raw = tf.extractfile(f"./NIFTI/{p}.nii.gz").read()
            except KeyError:
                print(f"MISSING {cid} {p}")
                continue
            im = nib.Nifti1Image.from_bytes(gzip.decompress(raw))
            im = im.as_reoriented(ornt_transform(io_orientation(im.affine), TARGET))
            nib.save(im, dst)
    print(cid, "ok", flush=True)
