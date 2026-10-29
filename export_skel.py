"""Export the SKEL skeleton as one mesh per bone, for the Bone Finder AR overlay.

Writes two files next to each other:
  skeleton.glb        24 named meshes (pelvis, femur_r, ... hand_l), rest pose, metres, y-up
  skeleton_rig.json   rest joint positions, joint tree and a few landmarks, same coordinates

The skeleton is shaped to one person in one of three ways:
  --smpl-fit FILE   SMPL parameters (.npz/.pkl with betas, poses, trans) fitted to a photo of
                    the person by an image-to-SMPL regressor (4D-Humans / HMR2.0, SMPLify-X,
                    ...). SKEL's own aligner then turns them into SKEL shape parameters.
  --betas FILE      SKEL betas you already have (.npy/.npz/.pkl/.json, 10 numbers; for .npz/.pkl
                    the key is "betas"), e.g. from SKEL's examples/align_to_SMPL_frame.py.
  (neither)         the average body.

Usage (from a Python 3.12 environment, see README):
  python export_skel.py --skel-dir path/to/SKEL --gender male --smpl-fit me_smpl.npz --out .

--skel-dir is a clone of https://github.com/MarilynKeller/SKEL whose data/skel/ folder holds
skel_male.pkl and skel_female.pkl (download them from https://skel.is.tue.mpg.de).
SKEL is licensed for non-commercial research use only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh


def load_betas(path):
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        data = data["betas"] if isinstance(data, dict) else data
    elif path.suffix == ".npy":
        data = np.load(path)
    else:
        import pickle
        raw = np.load(path, allow_pickle=True) if path.suffix == ".npz" else pickle.load(open(path, "rb"))
        data = raw["betas"]
    return torch.as_tensor(np.asarray(data, dtype=np.float32).reshape(-1, 10)[:1])


def fit_betas(path, gender, device):
    """SMPL parameters from a photo -> SKEL shape parameters, using SKEL's own aligner."""
    from skel.alignment.aligner import SkelFitter
    from skel.alignment.utils import load_smpl_seq
    smpl = load_smpl_seq(path, gender=gender, straighten_hands=False)
    fitter = SkelFitter(smpl["gender"], device=device, export_meshes=False)
    res = fitter.run_fit(smpl["trans"][:1], smpl["betas"][:1], smpl["poses"][:1], batch_size=1)
    return torch.as_tensor(np.asarray(res["betas"], dtype=np.float32).reshape(-1, 10)[:1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skel-dir", required=True, help="clone of the SKEL repository")
    ap.add_argument("--gender", choices=["male", "female"], default="male")
    ap.add_argument("--out", default=".", help="folder for skeleton.glb and skeleton_rig.json")
    ap.add_argument("--smpl-fit", help="SMPL parameters fitted to the person's photo (.npz/.pkl)")
    ap.add_argument("--betas", help="SKEL shape parameters (.npy/.npz/.pkl/.json)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", help="device for the SMPL->SKEL fit")
    args = ap.parse_args()

    skel_dir = Path(args.skel_dir).resolve()
    sys.path.insert(0, str(skel_dir))
    from skel.skel_model import SKEL
    from skel.kin_skel import skel_joints_name

    model = SKEL(gender=args.gender, model_path=str(skel_dir / "data" / "skel")).to("cpu")
    betas = torch.zeros(1, model.num_betas)
    if args.smpl_fit:
        betas = fit_betas(args.smpl_fit, args.gender, args.device)
    elif args.betas:
        betas = load_betas(args.betas)
    print("shape parameters:", np.round(betas[0].numpy(), 3).tolist())
    with torch.no_grad():
        out = model(torch.zeros(1, model.num_q_params), betas, torch.zeros(1, 3))

    verts = out.skel_verts[0].numpy()
    faces = model.skel_f.numpy()
    joints = out.joints[0].numpy()
    weights = model.skel_weights_rigid.to_dense().numpy()          # (vertices, 24), one bone per vertex
    assert weights.shape[1] == len(skel_joints_name) == joints.shape[0], "unexpected bone count"
    vert_bone = weights.argmax(axis=1)
    face_bone = np.array([np.bincount(vert_bone[f], minlength=weights.shape[1]).argmax() for f in faces])

    whole = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    scene = trimesh.Scene()
    for b, name in enumerate(skel_joints_name):
        idx = np.where(face_bone == b)[0]
        if len(idx) == 0:
            print(f"  (no faces for {name}, skipped)")
            continue
        part = whole.submesh([idx], append=True)
        scene.add_geometry(part, node_name=name, geom_name=name)
        print(f"  {name:12s} {len(idx):6d} faces")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.export(out_dir / "skeleton.glb")

    parents = [-1] + model.parent.tolist()                           # model.parent covers joints 1..23
    # Landmarks the browser can't read off the joints alone
    verts_by_bone = {n: np.asarray(whole.vertices)[np.unique(faces[face_bone == b])] for b, n in enumerate(skel_joints_name) if (face_bone == b).any()}
    hv = verts_by_bone["head"]
    bins = np.linspace(hv[:, 1].min(), hv[:, 1].max(), 25)
    widths = [np.ptp(hv[(hv[:, 1] >= lo) & (hv[:, 1] < hi), 0]) if ((hv[:, 1] >= lo) & (hv[:, 1] < hi)).any() else 0 for lo, hi in zip(bins[:-1], bins[1:])]
    ear_y = float((bins[int(np.argmax(widths))] + bins[int(np.argmax(widths)) + 1]) / 2)   # widest slice of the skull ~ ear level
    ear_mid = [float(hv[:, 0].mean()), ear_y, float((hv[:, 2].min() + hv[:, 2].max()) / 2)]
    rig = {
        "gender": args.gender,
        "ear_mid": np.round(ear_mid, 5).tolist(),
        "centroids": {n: np.round(v.mean(axis=0), 5).tolist() for n, v in verts_by_bone.items()},
        "joints": {n: joints[i].round(5).tolist() for i, n in enumerate(skel_joints_name)},
        "parents": {n: (skel_joints_name[p] if p >= 0 else None) for n, p in zip(skel_joints_name, parents)},
    }
    (out_dir / "skeleton_rig.json").write_text(json.dumps(rig, indent=1))
    print(f"wrote {out_dir / 'skeleton.glb'} and {out_dir / 'skeleton_rig.json'}")


if __name__ == "__main__":
    main()
