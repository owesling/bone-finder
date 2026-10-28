#!/usr/bin/env python3
"""
synth_osso.py: synthetic pretraining data for Bone Finder, from SMPL bodies and OSSO skeletons.

Run the stages in order (each one reads what the previous one wrote into --work):

  bodies   Sample varied SMPL body shapes in a relaxed standing pose. Saves each
           body mesh plus the 3D positions where MediaPipe's landmarks would be.
  landmarks  (optional) Recompute the landmark positions after editing
           landmark_config.json, without re-running OSSO.
  osso     Run OSSO on every body and collect its per-part bone meshes.
  parts    List the bone parts OSSO produced and write part_map.json, which maps
           each part to a Bone Finder class. Review it before the next stage.
  rows     Photograph each body with randomized virtual webcams, "touch" visible
           skin points, label each with the bone beneath, and export CSV rows in
           exactly the format train.py reads (same 86 f_ features as the app).
  preview  Draw one body's labeled touch points and landmarks to a PNG, to check
           that labels, sides and landmarks look right.

Requirements: numpy, scipy, matplotlib (preview). The bodies stage also needs
torch + smplx and the SMPL model files. Non-ASCII mesh formats need trimesh.
OSSO, SMPL and their outputs are licensed for non-commercial research only.
"""
import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Must match bone-finder.html exactly (FEAT_LM, FEATURE_NAMES, META_COLS, RAW_COLS)
# ---------------------------------------------------------------------------
FEAT_LM = [("nose", 0), ("eyeL", 2), ("eyeR", 5), ("earL", 7), ("earR", 8), ("mouthL", 9), ("mouthR", 10),
           ("shoulderL", 11), ("shoulderR", 12), ("elbowL", 13), ("elbowR", 14), ("wristL", 15), ("wristR", 16),
           ("hipL", 23), ("hipR", 24), ("kneeL", 25), ("kneeR", 26), ("ankleL", 27), ("ankleR", 28)]
FEATURE_NAMES = (["f_tip_x", "f_tip_y"]
                 + [f"f_rel_{n}_{a}" for n, _ in FEAT_LM for a in ("x", "y")]
                 + [f"f_has_{n}" for n, _ in FEAT_LM]
                 + ["f_ptr_left", "f_dir_x", "f_dir_y", "f_oh_present", "f_oh_x", "f_oh_y"]
                 + [f"f_oh_d{i}" for i in range(21)])
META_COLS = ["participant", "session", "trial_id", "label", "bone", "side", "frame", "t_ms",
             "ptr_side", "gesture", "tip_speed", "notes"]
RAW_COLS = ([f"pose{i}_{k}" for i in range(33) for k in ("x", "y", "z", "v")]
            + [f"{h}{i}_{k}" for h in ("ptr", "oth") for i in range(21) for k in ("x", "y", "z")])

BONE_IDS = {"parietal", "frontal", "temporal", "nasal", "orbit", "zygomatic", "maxilla", "mandible", "cervical",
            "clavicle", "acromion", "manubrium", "sternum", "xiphoid", "trueRibs", "falseRibs", "floatingRibs",
            "abdomen", "ilium", "pubis", "humerus", "elbow", "radiusUlna", "carpals", "metacarpals",
            "proximalPhalanx", "middlePhalanx", "distalPhalanx", "hand", "femur", "patella", "tibiaFibula", "foot"}
SIDED = {"temporal", "orbit", "zygomatic", "clavicle", "acromion", "trueRibs", "falseRibs", "floatingRibs", "ilium",
         "humerus", "elbow", "radiusUlna", "carpals", "metacarpals", "proximalPhalanx", "middlePhalanx",
         "distalPhalanx", "hand", "femur", "patella", "tibiaFibula", "foot"}
ARM_IDS = {"humerus", "elbow", "radiusUlna", "carpals", "metacarpals", "proximalPhalanx", "middlePhalanx",
           "distalPhalanx", "hand"}   # touched with the opposite hand
# Splitters subdivide one large OSSO part (e.g. a whole skull) by where on the body it is touched.
SPLITTERS = {"@skull", "@thorax", "@sternum", "@pelvis", "@spine", "@ignore"}

W, H = 1280, 720
MESH_EXT = {".ply", ".obj", ".stl", ".off", ".glb", ".gltf"}

DEFAULT_LANDMARK_CFG = {
    "_help": "Which SMPL vertices/joints stand in for each MediaPipe landmark. Edit to calibrate against real "
             "MediaPipe output, then re-run 'bodies'. Vertex ids assume SMPL topology (6890 vertices).",
    "vertices": {"nose": 332, "eyeL": 2800, "eyeR": 6260, "earL": 583, "earR": 4071},
    "joints": {"shoulderL": 16, "shoulderR": 17, "elbowL": 18, "elbowR": 19, "wristL": 20, "wristR": 21,
               "hipL": 1, "hipR": 2, "kneeL": 4, "kneeR": 5, "ankleL": 7, "ankleR": 8},
    "mouth_drop": 0.6,          # mouth sits this fraction of the eye-to-nose-tip height below the nose tip
    "mouth_half_width": 0.5,    # mouth corners, as a fraction of the distance between the eyes
    "hip_width_scale": 1.0,     # widen/narrow hip landmarks around their midpoint
    "hip_raise": 0.0,           # move hip landmarks up (meters)
}
LM_INDEX = {"nose": 0, "eyeL": 2, "eyeR": 5, "earL": 7, "earR": 8, "mouthL": 9, "mouthR": 10,
            "shoulderL": 11, "shoulderR": 12, "elbowL": 13, "elbowR": 14, "wristL": 15, "wristR": 16,
            "hipL": 23, "hipR": 24, "kneeL": 25, "kneeR": 26, "ankleL": 27, "ankleR": 28}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def body_dirs(work):
    dirs = sorted(p for p in Path(work).glob("body_*") if p.is_dir())
    if not dirs:
        sys.exit(f"No body_* folders in {work}. Run the 'bodies' stage first.")
    return dirs


def read_mesh(path):
    """Returns (vertices Nx3, triangle faces Mx3). ASCII PLY and OBJ need no extra packages."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".obj":
        return _read_obj(path)
    if ext == ".ply":
        with open(path, "rb") as fh:
            if b"format ascii" in fh.read(400):
                return _read_ply_ascii(path)
    try:
        import trimesh
    except ImportError:
        sys.exit(f"Reading {path.name} needs trimesh:  pip install trimesh")
    m = trimesh.load(path, force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def _triangulate(polys):
    tris = []
    for p in polys:
        tris += [[p[0], p[i], p[i + 1]] for i in range(1, len(p) - 1)]
    return np.asarray(tris, int).reshape(-1, 3)


def _read_obj(path):
    v, polys = [], []
    for line in open(path):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            polys.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return np.asarray(v, float), _triangulate(polys)


def _read_ply_ascii(path):
    lines = open(path).read().splitlines()
    nv = nf = 0
    props, cur, i = [], None, 0
    while lines[i].strip() != "end_header":
        tok = lines[i].split()
        if tok[:2] == ["element", "vertex"]:
            nv, cur = int(tok[2]), "vertex"
        elif tok[:2] == ["element", "face"]:
            nf, cur = int(tok[2]), "face"
        elif tok and tok[0] == "element":
            cur = None
        elif tok and tok[0] == "property" and cur == "vertex":
            props.append(tok[-1])
        i += 1
    ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
    body = lines[i + 1:]
    v = np.array([[float(r.split()[ix]), float(r.split()[iy]), float(r.split()[iz])] for r in body[:nv]])
    polys = [[int(x) for x in r.split()[1:1 + int(r.split()[0])]] for r in body[nv:nv + nf]]
    return v, _triangulate(polys)


def write_ply_ascii(path, v, f):
    with open(path, "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
                 f"property float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
        fh.writelines(f"{a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in v)
        fh.writelines(f"3 {a} {b} {c}\n" for a, b, c in f)


def sample_surface(v, f, n, rng):
    """Area-uniform random points on a triangle mesh, with outward face normals."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cr, axis=1)
    idx = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
    pts = (1 - r1)[:, None] * a[idx] + (r1 * (1 - r2))[:, None] * b[idx] + (r1 * r2)[:, None] * c[idx]
    nrm = cr[idx] / (np.linalg.norm(cr[idx], axis=1, keepdims=True) + 1e-12)
    if np.mean(np.sum(nrm * (pts - v.mean(0)), axis=1)) < 0:   # faces wound inward: flip
        nrm = -nrm
    return pts, nrm, area.sum()


# ---------------------------------------------------------------------------
# Stage 1: bodies
# ---------------------------------------------------------------------------
def smpl_landmarks(verts, joints, cfg):
    """3D stand-ins for the MediaPipe landmarks the app uses, from an SMPL mesh."""
    top = max(cfg["vertices"].values())
    if top >= len(verts):
        sys.exit(f"landmark_config.json uses vertex {top}, but the mesh has only {len(verts)} vertices. "
                 "The vertex ids assume SMPL topology (6890 vertices).")
    lm = {LM_INDEX[k]: verts[i].copy() for k, i in cfg["vertices"].items()}
    lm.update({LM_INDEX[k]: joints[i].copy() for k, i in cfg["joints"].items()})
    left = unit(lm[11] - lm[12])
    for a, b in ((2, 5), (7, 8)):                    # make sure "L" is the body's own left
        if np.dot(lm[a] - lm[b], left) < 0:
            lm[a], lm[b] = lm[b], lm[a]
    up = unit((lm[11] + lm[12]) / 2 - (lm[23] + lm[24]) / 2)
    eye_mid = (lm[2] + lm[5]) / 2
    drop = max(np.dot(eye_mid - lm[0], up), 0.01) * cfg["mouth_drop"]
    mouth = lm[0] - up * drop
    half = np.linalg.norm(lm[2] - lm[5]) * cfg["mouth_half_width"]
    lm[9], lm[10] = mouth + left * half, mouth - left * half
    hip_mid = (lm[23] + lm[24]) / 2
    for i in (23, 24):
        lm[i] = hip_mid + (lm[i] - hip_mid) * cfg["hip_width_scale"] + up * cfg["hip_raise"]
    return lm


def cmd_bodies(a):
    try:
        import torch
        import smplx
    except ImportError:
        sys.exit("The bodies stage needs:  pip install torch smplx")
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    cfg_path = work / "landmark_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(DEFAULT_LANDMARK_CFG, indent=2))
    cfg = json.loads(cfg_path.read_text())
    rng = np.random.default_rng(a.seed)
    models = {g: smplx.create(a.smpl_dir, model_type="smpl", gender=g, num_betas=10) for g in ("female", "male")}

    def run(model, betas, arm, sign):
        bp = torch.zeros(1, 69)
        bp[0, 15 * 3 + 2] = -sign * arm          # left shoulder (SMPL joint 16), about the z axis
        bp[0, 16 * 3 + 2] = sign * arm           # right shoulder (SMPL joint 17)
        out = model(betas=torch.tensor(betas[None], dtype=torch.float32), body_pose=bp,
                    global_orient=torch.zeros(1, 3))
        return out.vertices[0].detach().numpy(), out.joints[0].detach().numpy()

    for i in range(a.n):
        gender = str(rng.choice(["female", "male"]))
        betas = np.clip(rng.normal(0, a.beta_std, 10), -2.5, 2.5)
        arm = math.radians(rng.uniform(55, 75))
        model = models[gender]
        verts, joints = run(model, betas, arm, 1.0)
        if joints[20][1] > joints[16][1] - 0.1:        # arms went up instead of down: flip the rotation
            verts, joints = run(model, betas, arm, -1.0)
        d = work / f"body_{i:04d}"
        d.mkdir(exist_ok=True)
        write_ply_ascii(d / "body.ply", verts, np.asarray(model.faces, int))
        np.save(d / "joints.npy", joints)
        lm = smpl_landmarks(verts, joints, cfg)
        (d / "landmarks3d.json").write_text(json.dumps({"gender": gender, "lm": {str(k): v.tolist() for k, v in lm.items()}}))
        (d / "params.json").write_text(json.dumps({"gender": gender, "betas": betas.tolist(), "arm_deg": math.degrees(arm)}))
        print(f"{d.name}: {gender}, height {np.ptp(verts[:, 1]):.2f} m")


def cmd_landmarks(a):
    cfg = json.loads((Path(a.work) / "landmark_config.json").read_text())
    for d in body_dirs(a.work):
        verts, _ = read_mesh(d / "body.ply")
        lm = smpl_landmarks(verts, np.load(d / "joints.npy"), cfg)
        old = json.loads((d / "landmarks3d.json").read_text())
        old["lm"] = {str(k): v.tolist() for k, v in lm.items()}
        (d / "landmarks3d.json").write_text(json.dumps(old))
    print("Landmarks updated. Re-run 'rows' to regenerate the CSV.")


# ---------------------------------------------------------------------------
# Stage 2: osso
# ---------------------------------------------------------------------------
def _snapshot(folder):
    return {p: p.stat().st_mtime for p in Path(folder).rglob("*") if p.is_file()} if Path(folder).exists() else {}


def cmd_osso(a):
    osso = Path(a.osso_dir).resolve()
    if not (osso / "main.py").exists():
        sys.exit(f"{osso} doesn't contain OSSO's main.py")
    out_dir = osso / a.osso_out
    for d in body_dirs(a.work):
        dest = d / "osso"
        if dest.exists() and any(dest.iterdir()) and not a.force:
            print(f"{d.name}: already done (use --force to redo)")
            continue
        gender = json.loads((d / "params.json").read_text())["gender"]
        before = _snapshot(out_dir)
        cmd = [a.python, "main.py", "--mesh_input", str((d / "body.ply").resolve()), "--gender", gender,
               "--per_part", *shlex.split(a.osso_args)]
        r = subprocess.run(cmd, cwd=osso, capture_output=True, text=True)
        (d / "osso_log.txt").write_text(" ".join(cmd) + "\n\n" + r.stdout + "\n" + r.stderr)
        if r.returncode != 0:
            print(f"{d.name}: OSSO failed, see {d / 'osso_log.txt'}")
            continue
        new = [p for p, t in _snapshot(out_dir).items()
               if before.get(p) != t and p.suffix.lower() in MESH_EXT and "tmp" not in p.relative_to(out_dir).parts]
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        for p in new:
            shutil.copy2(p, dest / "__".join(p.relative_to(out_dir).parts))
        note = "" if len(new) > 1 else "  WARNING: expected several part meshes; check --per_part worked"
        print(f"{d.name}: {len(new)} mesh files{note}")


# ---------------------------------------------------------------------------
# Stage 3: parts
# ---------------------------------------------------------------------------
GUESSES = [("mandib", "mandible"), ("jaw", "mandible"), ("skull", "@skull"), ("cran", "@skull"), ("head", "@skull"),
           ("clavic", "clavicle"), ("scapul", "acromion"), ("humer", "humerus"), ("radius", "radiusUlna"),
           ("ulna", "radiusUlna"), ("forearm", "radiusUlna"), ("hand", "@ignore"), ("carp", "@ignore"),
           ("phalan", "@ignore"), ("finger", "@ignore"), ("stern", "@sternum"), ("rib", "@thorax"),
           ("thorax", "@thorax"), ("cage", "@thorax"), ("cervic", "cervical"), ("neck", "cervical"),
           ("spine", "@spine"), ("vertebr", "@spine"), ("lumbar", "@spine"), ("thoracic", "@spine"),
           ("sacr", "@ignore"), ("coccy", "@ignore"), ("pelv", "@pelvis"), ("ilium", "ilium"), ("hip", "@pelvis"),
           ("femur", "femur"), ("patell", "patella"), ("knee", "patella"), ("tibia", "tibiaFibula"),
           ("fibula", "tibiaFibula"), ("foot", "foot"), ("tars", "foot"), ("calcan", "foot"), ("talus", "foot"),
           ("toe", "foot"), ("skel", "@ignore"), ("body", "@ignore"), ("skin", "@ignore")]


# OSSO's per-part output, one mesh per body segment. Sides are detected from geometry, so Left/Right
# share a class. Spine1/Spine2/Sternum all go to @thorax, which is correct wherever the ribs are stored.
OSSO_PARTS = {"Pelvis": "@pelvis", "Spine": "@spine", "Spine1": "@thorax", "Spine2": "@thorax",
              "Sternum": "@thorax", "Neck": "cervical", "Head": "@skull",
              "LeftClavicle": "clavicle", "RightClavicle": "clavicle",
              "LeftArm1": "humerus", "RightArm1": "humerus", "LeftArm2": "radiusUlna", "RightArm2": "radiusUlna",
              "LeftHand": "@ignore", "RightHand": "@ignore",
              "LeftLeg1": "femur", "RightLeg1": "femur", "LeftLeg2": "tibiaFibula", "RightLeg2": "tibiaFibula",
              "LeftFoot": "foot", "RightFoot": "foot"}


def part_key(path):
    """'body_skel_posed_per_parts__9_LeftArm1.ply' -> 'LeftArm1'; whole-skeleton files keep their full name."""
    stem = Path(path).stem
    if "__" in stem:
        stem = stem.split("__")[-1]
        head, _, rest = stem.partition("_")
        if head.isdigit() and rest:
            stem = rest
    return stem


def cmd_parts(a):
    work = Path(a.work)
    found = {}
    for d in body_dirs(work):
        for p in sorted((d / "osso").glob("*")) if (d / "osso").exists() else []:
            if p.suffix.lower() in MESH_EXT:
                found.setdefault(part_key(p), (d, p))
    if not found:
        sys.exit("No OSSO part meshes found. Run the 'osso' stage first.")
    map_path = work / "part_map.json"
    existing = json.loads(map_path.read_text()) if map_path.exists() else {}
    mapping = {"_help": "Map each OSSO part to a Bone Finder class, a splitter (@skull @thorax @sternum @pelvis "
                        "@spine) or @ignore. Sides (L/R) are detected from geometry, so one class per bone type. "
                        "'width' is the part's side-to-side size in shoulder widths."}
    print(f"{'part':<34}{'guess':<14}{'verts':>7}{'width':>7}   position on body (first body with this part)")
    for key, (d, p) in sorted(found.items()):
        v, _ = read_mesh(p)
        lm = {int(k): np.array(x) for k, x in json.loads((d / "landmarks3d.json").read_text())["lm"].items()}
        fr = Frame(lm)
        c = v.mean(0)
        width = np.ptp((v - fr.O) @ fr.left) / fr.sw
        t, s = fr.torso_ts(c)
        side = "left" if np.dot(c - fr.O, fr.left) > 0.02 else "right" if np.dot(c - fr.O, fr.left) < -0.02 else "midline"
        guess = existing.get(key) or OSSO_PARTS.get(key) or next((g for kw, g in GUESSES if kw in key.lower()), "@ignore")
        mapping[key] = guess
        where = f"height t={t:+.2f} (0 shoulders, 1 hips), {side}"
        print(f"{key:<34}{guess:<14}{len(v):>7}{width:>7.2f}   {where}")
    map_path.write_text(json.dumps(mapping, indent=2))
    print(f"\nWrote {map_path}. Check every line, especially anything marked @ignore, then run 'rows'.")


# ---------------------------------------------------------------------------
# Geometry of the body: a 3D version of the app's torso and head frames
# ---------------------------------------------------------------------------
class Frame:
    def __init__(self, lm):
        self.lm = lm
        self.Ls, self.Rs = lm[11], lm[12]
        self.O = (self.Ls + self.Rs) / 2
        self.sw = float(np.linalg.norm(self.Ls - self.Rs))
        self.left = unit(self.Ls - self.Rs)
        self.hips = (lm[23] + lm[24]) / 2
        up = self.O - self.hips
        self.up = unit(up - np.dot(up, self.left) * self.left)
        fwd = unit(np.cross(self.left, self.up))
        self.forward = fwd if np.dot(lm[0] - self.O, fwd) >= 0 else -fwd

    def plane(self, p):   # frontal plane: x toward the body's left, y down, meters
        r = p - self.O
        return np.array([np.dot(r, self.left), -np.dot(r, self.up)])

    def torso_ts(self, p):
        q, v = self.plane(p), self.plane(self.hips)
        t = float(np.dot(q, v) / (np.dot(v, v) or 1))
        hip_w = np.linalg.norm(self.lm[23] - self.lm[24])
        half = self.sw / 2 + ((hip_w * 1.4) / 2 - self.sw / 2) * min(max(t, 0), 1)
        return t, float((q[0] - v[0] * t) / half)

    def head_xy(self, p):
        q = self.plane(p)
        eye, mouth = (self.plane(self.lm[2]) + self.plane(self.lm[5])) / 2, (self.plane(self.lm[9]) + self.plane(self.lm[10])) / 2
        down = mouth - eye
        e = np.linalg.norm(down) or 1
        el, er = self.plane(self.lm[7]), self.plane(self.lm[8])
        lat, half = unit(el - er), np.linalg.norm(el - er) / 2 or 1
        rel = q - eye
        return float(np.dot(rel, lat) / half), float(np.dot(rel, down / e) / e)


def head_class(x, y):          # same zones as the app's rule-based head classifier
    ax = abs(x)
    if y < -1.4: return "parietal"
    if y < -0.3: return "frontal" if ax < 0.8 else "temporal"
    if ax >= 0.78: return "temporal" if y < 1.0 else "mandible"
    if y < 0.3: return "nasal" if ax < 0.2 else "orbit" if ax < 0.62 else "zygomatic"
    if y < 0.6 and ax < 0.18: return "nasal"
    if y < 0.75: return "maxilla" if ax < 0.38 else "zygomatic"
    if y < 1.0: return "maxilla" if ax < 0.5 else "mandible"
    return "mandible"


def thorax_class(t, s):
    a = abs(s)
    if t < 0.12 and a < 0.22: return "manubrium"
    if t < 0.42 and a < 0.18: return "sternum"
    if t < 0.5 and a < 0.18: return "xiphoid"
    if t < 0.36: return "trueRibs"
    if a >= 0.62: return "falseRibs" if t < 0.5 else "floatingRibs"
    return "falseRibs"


def label_touch(p, bone_pt, depth, target, fr, a):
    """Class id for a skin point p whose nearest bone point is bone_pt, in OSSO part mapped to `target`."""
    if target == "@ignore" or depth > a.max_depth:
        return None
    t, s = fr.torso_ts(p)
    if 0.3 < t < 1.02 and abs(s) < 0.95 and depth > a.abdomen_depth:
        return "abdomen"                                    # front of the torso, bone is far behind
    if target == "@skull": cid = head_class(*fr.head_xy(p))
    elif target == "@thorax": cid = thorax_class(t, s)
    elif target == "@sternum": cid = "manubrium" if t < 0.12 else "sternum" if t < 0.42 else "xiphoid"
    elif target == "@pelvis": cid = "ilium" if abs(s) >= 0.35 else "pubis"
    elif target == "@spine": cid = "cervical" if t < 0.02 else "abdomen"
    else: cid = target
    left = np.dot(p - fr.O, fr.left) > 0
    near = lambda i_l, i_r, r: np.linalg.norm(p - fr.lm[i_l if left else i_r]) < r * fr.sw
    if cid in ("humerus", "radiusUlna") and near(13, 14, a.elbow_radius): cid = "elbow"
    if cid in ("femur", "tibiaFibula") and near(25, 26, a.knee_radius): cid = "patella"
    if cid in ("humerus", "clavicle", "trueRibs") and near(11, 12, a.shoulder_radius): cid = "acromion"
    return cid


# ---------------------------------------------------------------------------
# Virtual webcam
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, fr, rng, a):
        center = fr.O * 0.6 + fr.hips * 0.4
        yaw, pitch, roll = (math.radians(rng.normal(0, a.yaw_deg)), math.radians(rng.normal(0, 4)),
                            math.radians(rng.normal(0, 2)))
        fwd = fr.forward * math.cos(yaw) + np.cross(fr.up, fr.forward) * math.sin(yaw)
        self.pos = center + fwd * rng.uniform(a.dist_min, a.dist_max) + fr.up * rng.normal(0.05, 0.1)
        d = unit(center + fr.up * math.tan(pitch) - self.pos)
        right = unit(np.cross(d, fr.up))                       # image right = the body's own left
        up = np.cross(right, d)
        self.right = right * math.cos(roll) + up * math.sin(roll)
        self.up = np.cross(self.right, d)
        self.d = d
        self.f = (W / 2) / math.tan(math.radians(rng.uniform(55, 80)) / 2)

    def project(self, X):
        v = np.atleast_2d(X) - self.pos
        z = v @ self.d
        uv = np.stack([W / 2 + self.f * (v @ self.right) / z, H / 2 - self.f * (v @ self.up) / z], axis=1)
        return uv, z


def inside(uv):
    return (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)


def visibility(cam, dense_pts, pts, nrm, cell=4, eps=0.02):
    """Point-splat depth buffer: a point is visible if nothing on the body is in front of it."""
    duv, dz = cam.project(dense_pts)
    ok = inside(duv) & (dz > 0)
    gx, gy = (duv[ok, 0] // cell).astype(int), (duv[ok, 1] // cell).astype(int)
    zb = np.full((H // cell + 1, W // cell + 1), np.inf)
    np.minimum.at(zb, (gy, gx), dz[ok])
    uv, z = cam.project(pts)
    vis = inside(uv) & (z > 0) & (np.sum(nrm * (cam.pos - pts), axis=1) > 0)
    iy, ix = (np.clip(uv[:, 1], 0, H - 1) // cell).astype(int), (np.clip(uv[:, 0], 0, W - 1) // cell).astype(int)
    vis &= z <= zb[iy, ix] + eps
    return vis, uv


# ---------------------------------------------------------------------------
# Features: a line-for-line port of computeFeatures() in bone-finder.html
# ---------------------------------------------------------------------------
def compute_features(P, tip, ptr_left, h6, h8):
    Ls, Rs = P[11], P[12]
    O = (Ls + Rs) / 2
    s = np.linalg.norm(Ls - Rs) or 1.0
    ex = (Ls - Rs) / (np.linalg.norm(Ls - Rs) or 1.0)
    ey = np.array([-ex[1], ex[0]])
    B = lambda p: np.array([np.dot(p - O, ex) / s, np.dot(p - O, ey) / s])
    t = B(tip)
    f, has = [t[0], t[1]], []
    for _, i in FEAT_LM:
        if i in P:
            q = B(P[i]); f += [t[0] - q[0], t[1] - q[1]]; has.append(1)
        else:
            f += [0.0, 0.0]; has.append(0)
    f += has
    f.append(1 if ptr_left else 0)
    dvec = B(h8) - B(h6)
    n = np.linalg.norm(dvec) or 1.0
    f += [dvec[0] / n, dvec[1] / n]
    f += [0, 0, 0] + [5] * 21                     # other hand: not simulated
    return f


# ---------------------------------------------------------------------------
# Stage 4: rows
# ---------------------------------------------------------------------------
def load_body(d, mapping):
    v, f = read_mesh(d / "body.ply")
    lm = {int(k): np.array(x) for k, x in json.loads((d / "landmarks3d.json").read_text())["lm"].items()}
    parts = []
    for p in sorted((d / "osso").glob("*")) if (d / "osso").exists() else []:
        if p.suffix.lower() in MESH_EXT and mapping.get(part_key(p), "@ignore") != "@ignore":
            pv, pf = read_mesh(p)
            parts.append((part_key(p), pv, pf))
    return v, f, lm, parts


def body_rows(d, bidx, mapping, rng, a, debug=False):
    from scipy.spatial import cKDTree
    v, f, lm, parts = load_body(d, mapping)
    if not parts:
        print(f"{d.name}: no mapped bone parts, skipped")
        return [], None
    fr = Frame(lm)
    # Bone point cloud, tagged by part
    bpts, btag = [], []
    areas = [sample_surface(pv, pf, 10, rng)[2] for _, pv, pf in parts]
    for k, ((key, pv, pf), ar) in enumerate(zip(parts, areas)):
        n = max(200, int(a.bone_samples * ar / sum(areas)))
        pts, _, _ = sample_surface(pv, pf, n, rng)
        bpts.append(pts); btag += [k] * n
    bpts, btag = np.vstack(bpts), np.array(btag)
    if np.linalg.norm(bpts.mean(0) - v.mean(0)) > 0.25:
        print(f"{d.name}: WARNING skeleton center is {np.linalg.norm(bpts.mean(0) - v.mean(0)):.2f} m from the body "
              "center. OSSO's output may be in a different coordinate frame or scale.")
    btree = cKDTree(bpts)
    dense, _, _ = sample_surface(v, f, a.dense_samples, rng)
    rows, dbg = [], []
    cap = max(1, math.ceil(a.per_class / a.cameras))
    for c in range(a.cameras):
        cam = Camera(fr, rng, a)
        pts, nrm, _ = sample_surface(v, f, a.candidates, rng)
        vis, uv = visibility(cam, dense, pts, nrm)
        lm_uv = {i: cam.project(p)[0][0] for i, p in lm.items()}
        if not (11 in lm_uv and 12 in lm_uv and inside(np.array([lm_uv[11], lm_uv[12]])).all()):
            continue                                    # shoulders out of frame: the app would skip too
        sw_px = np.linalg.norm(lm_uv[11] - lm_uv[12])
        bias = {i: rng.normal(0, a.lm_bias * sw_px, 2) for i in lm_uv}     # per-camera systematic offset
        counts = {}
        depth, nn = btree.query(pts)
        for j in rng.permutation(np.flatnonzero(vis)):
            key = parts[btag[nn[j]]][0]
            cid = label_touch(pts[j], bpts[nn[j]], depth[j], mapping[key], fr, a)
            if cid is None or cid not in BONE_IDS:
                continue
            side = ("L" if np.dot(pts[j] - fr.O, fr.left) > 0 else "R") if cid in SIDED else ""
            label = cid + (f"_{side}" if side else "")
            if counts.get(label, 0) >= cap:
                continue
            counts[label] = counts.get(label, 0) + 1
            rows.append(make_row(d, bidx, c, pts[j], label, cid, side, fr, cam, lm_uv, bias, sw_px, rng, a))
            if debug: dbg.append((uv[j], label))
        # "none": fingertip in the air, clearly off the body in the image
        skin_uv = uv[vis]
        if len(skin_uv) and a.none_per_camera:
            stree, got, tries = cKDTree(skin_uv), 0, 0
            while got < a.none_per_camera and tries < a.none_per_camera * 40:
                tries += 1
                q = (fr.O + fr.left * rng.uniform(-3, 3) * fr.sw - fr.up * rng.uniform(-1.5, 4) * fr.sw
                     + fr.forward * rng.uniform(0.05, 0.5))
                quv, qz = cam.project(q)
                if qz[0] <= 0 or not inside(quv)[0] or stree.query(quv[0])[0] < a.none_margin * sw_px:
                    continue
                got += 1
                rows.append(make_row(d, bidx, c, q, "none", "none", "", fr, cam, lm_uv, bias, sw_px, rng, a, air=True))
                if debug: dbg.append((quv[0], "none"))
    return rows, (dbg, fr, lm_uv if rows else None)


_trial = [0]


def make_row(d, bidx, c, p, label, cid, side, fr, cam, lm_uv, bias, sw_px, rng, a, air=False):
    if cid in ARM_IDS:
        ptr = "R" if side == "L" else "L"
    elif air:
        ptr = "L" if np.dot(p - fr.O, fr.left) > 0 else "R"
        if rng.random() < 0.2: ptr = "L" if ptr == "R" else "R"
    else:
        ptr = str(rng.choice(["L", "R"]))
    sgn = 1 if ptr == "L" else -1
    # the pointing hand arrives from its own side, from below and in front
    approach = unit(fr.left * sgn * rng.uniform(0.2, 0.8) - fr.up * rng.uniform(0.3, 1.0)
                    + fr.forward * rng.uniform(0.2, 0.9))
    tip3 = p + fr.forward * 0.01
    h8 = cam.project(tip3)[0][0] + rng.normal(0, a.tip_noise * sw_px, 2)
    h6 = cam.project(tip3 + approach * 0.045)[0][0] + rng.normal(0, a.tip_noise * sw_px, 2)
    P = {}
    for i, q in lm_uv.items():
        if i in (11, 12) or (0 <= q[0] < W and 0 <= q[1] < H and rng.random() > a.drop_prob):
            P[i] = q + bias[i] + rng.normal(0, a.lm_noise * sw_px, 2)
    feats = compute_features(P, h8, ptr == "L", h6, h8)
    _trial[0] += 1
    row = [f"SYN{bidx:04d}", "synthetic", _trial[0], label, cid, side, 0, 0, ptr, 1, 0.0, f"osso cam{c}"] + feats
    if a.with_raw:
        raw = []
        for i in range(33):
            raw += [P[i][0] / W, P[i][1] / H, 0.0, 1.0] if i in P else ["", "", "", ""]
        for i in range(21):
            q = h6 if i == 6 else h8 if i == 8 else None
            raw += [q[0] / W, q[1] / H, 0.0] if q is not None else ["", "", ""]
        raw += [""] * 63
        row += raw
    return row


def load_mapping(work):
    path = Path(work) / "part_map.json"
    if not path.exists():
        sys.exit("part_map.json not found. Run the 'parts' stage first.")
    m = {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}
    bad = {k: v for k, v in m.items() if v not in BONE_IDS and v not in SPLITTERS}
    if bad:
        sys.exit(f"part_map.json has unknown targets: {bad}\nAllowed: {sorted(BONE_IDS | SPLITTERS)}")
    return m


def cmd_rows(a):
    mapping = load_mapping(a.work)
    rng = np.random.default_rng(a.seed)
    header = META_COLS + FEATURE_NAMES + (RAW_COLS if a.with_raw else [])
    total, counts = 0, {}
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for bidx, d in enumerate(body_dirs(a.work)):
            rows, _ = body_rows(d, bidx, mapping, rng, a)
            for r in rows:
                w.writerow([f"{x:.5f}" if isinstance(x, float) else x for x in r])
                counts[r[3]] = counts.get(r[3], 0) + 1
            total += len(rows)
            print(f"{d.name}: {len(rows)} rows")
    print(f"\nWrote {total} rows to {a.out}")
    for k in sorted(counts):
        print(f"  {k:<22}{counts[k]:>7}")


def cmd_preview(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mapping = load_mapping(a.work)
    d = body_dirs(a.work)[a.body]
    a.cameras = 1
    rows, (dbg, fr, lm_uv) = body_rows(d, a.body, mapping, np.random.default_rng(a.seed), a, debug=True)
    if not dbg:
        sys.exit("Nothing visible from the sampled camera. Try another --seed.")
    labels = sorted({l for _, l in dbg})
    cmap = plt.get_cmap("tab20", len(labels))
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    for k, lab in enumerate(labels):
        pts = np.array([u for u, l in dbg if l == lab])
        ax.scatter(pts[:, 0], pts[:, 1], s=9, color="lightgray" if lab == "none" else cmap(k), label=lab)
        if lab != "none":
            ax.annotate(lab, pts.mean(0), fontsize=7, ha="center")
    for i, q in lm_uv.items():
        ax.plot(q[0], q[1], "kx")
        ax.annotate(str(i), q, fontsize=7, color="k")
    body_uv = np.array([u for u, l in dbg if l != "none"] + list(lm_uv.values()))
    lo, hi = body_uv.min(0), body_uv.max(0)
    pad = 0.15 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(hi[1] + pad, lo[1] - pad); ax.set_aspect("equal")
    ax.set_title(f"{d.name}: labeled touch points as the webcam would see them (image right = body's left)")
    ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=2)
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"Saved {a.out} ({len(dbg)} points)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    work = argparse.ArgumentParser(add_help=False)
    work.add_argument("--work", default="synth_work", help="working folder shared by all stages")
    work.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("bodies", parents=[work])
    b.add_argument("--smpl-dir", required=True, help="folder containing smpl/SMPL_FEMALE.pkl and SMPL_MALE.pkl")
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--beta-std", type=float, default=1.0, help="spread of body shapes (SMPL betas)")

    sub.add_parser("landmarks", parents=[work])

    o = sub.add_parser("osso", parents=[work])
    o.add_argument("--osso-dir", required=True, help="your clone of github.com/MarilynKeller/OSSO")
    o.add_argument("--python", default=sys.executable, help="python of the environment OSSO is installed in")
    o.add_argument("--osso-out", default="out", help="folder (inside the OSSO clone) where OSSO writes results")
    o.add_argument("--osso-args", default="", help="extra arguments passed to OSSO's main.py")
    o.add_argument("--force", action="store_true")

    sub.add_parser("parts", parents=[work])

    for name in ("rows", "preview"):
        r = sub.add_parser(name, parents=[work])
        r.add_argument("--out", default="synthetic.csv" if name == "rows" else "preview.png")
        r.add_argument("--cameras", type=int, default=4, help="virtual webcams per body")
        r.add_argument("--per-class", type=int, default=40, help="max rows per class per body (keeps classes balanced)")
        r.add_argument("--candidates", type=int, default=8000, help="skin points tried per camera")
        r.add_argument("--none-per-camera", type=int, default=8)
        r.add_argument("--bone-samples", type=int, default=40000)
        r.add_argument("--dense-samples", type=int, default=150000, help="points used for the visibility test")
        r.add_argument("--max-depth", type=float, default=0.12, help="ignore touches with no bone within this (m)")
        r.add_argument("--abdomen-depth", type=float, default=0.045, help="front-torso touches deeper than this are 'abdomen'")
        r.add_argument("--elbow-radius", type=float, default=0.14)
        r.add_argument("--knee-radius", type=float, default=0.16)
        r.add_argument("--shoulder-radius", type=float, default=0.12)
        r.add_argument("--dist-min", type=float, default=1.4, help="camera distance range (m)")
        r.add_argument("--dist-max", type=float, default=3.2)
        r.add_argument("--yaw-deg", type=float, default=8.0, help="how far the person turns from the camera")
        r.add_argument("--lm-bias", type=float, default=0.02, help="systematic landmark offset, x shoulder width")
        r.add_argument("--lm-noise", type=float, default=0.01, help="per-frame landmark jitter, x shoulder width")
        r.add_argument("--tip-noise", type=float, default=0.008)
        r.add_argument("--drop-prob", type=float, default=0.03, help="chance a visible landmark goes missing")
        r.add_argument("--none-margin", type=float, default=0.1, help="air points must be this far off the body")
        r.add_argument("--with-raw", action="store_true", help="also write raw landmark columns (for checking)")
        if name == "preview":
            r.add_argument("--body", type=int, default=0)

    a = ap.parse_args()
    {"bodies": cmd_bodies, "landmarks": cmd_landmarks, "osso": cmd_osso, "parts": cmd_parts, "rows": cmd_rows, "preview": cmd_preview}[a.cmd](a)


if __name__ == "__main__":
    main()
