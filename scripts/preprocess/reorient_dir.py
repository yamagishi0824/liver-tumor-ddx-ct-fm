"""Reorient all NIfTIs under <src>/<phase>/ to LAS into <dst>/<phase>/ (parallel, skips existing).
Usage: python reorient_dir.py <src_root> <dst_root> [n_proc]
"""
import glob
import os
import sys
from multiprocessing import Pool

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform

TARGET = axcodes2ornt(("L", "A", "S"))


def work(args):
    src, dst = args
    if os.path.exists(dst):
        return
    im = nib.load(src)
    im = im.as_reoriented(ornt_transform(io_orientation(im.affine), TARGET))
    nib.save(im, dst + ".tmp.nii.gz")
    os.replace(dst + ".tmp.nii.gz", dst)


if __name__ == "__main__":
    src_root, dst_root = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    jobs = []
    for p in ["nc", "art", "pvp", "delay"]:
        os.makedirs(os.path.join(dst_root, p), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(src_root, p, "*.nii.gz"))):
            jobs.append((f, os.path.join(dst_root, p, os.path.basename(f))))
    with Pool(n) as pool:
        pool.map(work, jobs, chunksize=4)
    print("reoriented", len(jobs))
