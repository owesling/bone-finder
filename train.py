"""
Train the Bone Finder pointing classifier from CSVs recorded in Collect mode.

    pip install numpy pandas scikit-learn
    python train.py recordings/*.csv --out model.json

Then open Bone Finder, expand "Trained model" and load model.json.
"""
import argparse
import glob
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def feature_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


# ---------- Mirror augmentation --------------------------------------------
# A left-side example, flipped horizontally, is a valid right-side example.
# In the body frame that means: negate every x feature, swap left/right
# landmark columns, flip which hand is pointing, and swap the label's side.
LR = re.compile(r"^(f_(?:rel|has)_[a-z]+)([LR])(.*)$")
X_NEGATE = re.compile(r"^(f_tip_x|f_rel_.+_x|f_dir_x|f_oh_x)$")


def partner(col):
    m = LR.match(col)
    return m.group(1) + ("R" if m.group(2) == "L" else "L") + m.group(3) if m else col


def swap_label(label):
    if label.endswith("_L"):
        return label[:-2] + "_R"
    if label.endswith("_R"):
        return label[:-2] + "_L"
    return label


def mirror(df):
    feats = feature_cols(df)
    m = df.copy()
    m[feats] = df[[partner(c) for c in feats]].to_numpy()
    for c in feats:
        if X_NEGATE.match(c):
            m[c] = -m[c]
    m["f_ptr_left"] = 1 - df["f_ptr_left"]
    m["label"] = df["label"].map(swap_label)
    return m


# ---------- Training ----------------------------------------------------------
def make_mlp(seed=0):
    return MLPClassifier(hidden_layer_sizes=(64, 64), activation="relu", alpha=1e-3,
                         learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                         n_iter_no_change=20, random_state=seed)


class Model:
    """Scaler + network, with class names encoded as integers for training."""
    def __init__(self, df, seed=0):
        feats = feature_cols(df)
        aug = pd.concat([df, mirror(df)], ignore_index=True)   # doubles the data
        self.feats = feats
        self.enc = LabelEncoder().fit(aug["label"])
        self.scaler = StandardScaler().fit(aug[feats])
        self.mlp = make_mlp(seed).fit(self.scaler.transform(aug[feats]), self.enc.transform(aug["label"]))

    def predict(self, df):
        return self.enc.inverse_transform(self.mlp.predict(self.scaler.transform(df[self.feats])))

    def predict_proba(self, df):
        return self.mlp.predict_proba(self.scaler.transform(df[self.feats]))


def top_confusions(y_true, y_pred, labels, n=8):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    pairs = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]), reverse=True)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+", help="CSV files from Collect mode (globs allowed)")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--min-gesture", type=float, default=0.0,
                    help="drop frames where the pointing gesture wasn't detected (set to 1 to enable)")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.csv for p in glob.glob(pattern)})
    if not paths:
        sys.exit("No CSV files found.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).copy()
    df = df[df["gesture"] >= args.min_gesture]
    feats = feature_cols(df)
    df[feats] = df[feats].fillna(0)
    df["label"] = df["label"].astype(str)
    df["recording"] = df["session"].astype(str) + "#" + df["trial_id"].astype(str)

    n_people = df["participant"].nunique()
    print(f"{len(df)} frames, {df['recording'].nunique()} recordings, {n_people} participants, "
          f"{df['label'].nunique()} classes, {len(feats)} features")

    # ---- Honest evaluation: test on people (or recordings) the model never saw
    if n_people >= 3:
        groups, split_by = df["participant"], "participants"
    else:
        groups, split_by = df["recording"], "recordings"
        print("WARNING: fewer than 3 participants, so the test set shares people with the training set. "
              "Accuracy will look better than it will be for new users.")
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(df, groups=groups))
    train, test = df.iloc[tr_idx], df.iloc[te_idx]

    knn_scaler = StandardScaler().fit(train[feats])
    knn = KNeighborsClassifier(n_neighbors=15).fit(knn_scaler.transform(train[feats]), train["label"])
    knn_acc = accuracy_score(test["label"], knn.predict(knn_scaler.transform(test[feats])))

    pred = Model(train).predict(test)
    acc = accuracy_score(test["label"], pred)

    # Per-recording accuracy: majority vote over ~1 s of frames, closer to how the app behaves
    votes = pd.DataFrame({"rec": test["recording"].values, "true": test["label"].values, "pred": pred})
    rec = votes.groupby("rec").agg(true=("true", "first"), pred=("pred", lambda s: s.mode().iloc[0]))
    rec_acc = (rec["true"] == rec["pred"]).mean()

    print(f"\nHeld-out {split_by}: {test['label'].size} frames")
    print(f"  k-nearest-neighbors baseline, per frame: {knn_acc:.1%}")
    print(f"  neural network, per frame:               {acc:.1%}")
    print(f"  neural network, per recording (vote):    {rec_acc:.1%}")
    print("\nPer-class results:")
    print(classification_report(test["label"], pred, zero_division=0))
    labels = sorted(df["label"].unique())
    print("Most common mix-ups (true -> predicted):")
    for n, t, p in top_confusions(test["label"], pred, labels):
        print(f"  {t:>22} -> {p:<22} {n} frames")

    # ---- Final model on all data, exported for the browser
    model = Model(df)
    scaler, mlp = model.scaler, model.mlp
    r = lambda a: np.round(np.asarray(a, dtype=float), 6).tolist()
    out = {
        "format": "bonefinder-mlp-1",
        "feature_names": feats,
        "mean": r(scaler.mean_),
        "scale": r(scaler.scale_),
        "activation": mlp.activation,
        "out_activation": mlp.out_activation_,
        "classes": [str(c) for c in model.enc.inverse_transform(mlp.classes_)],
        "layers": [{"W": r(W), "b": r(b)} for W, b in zip(mlp.coefs_, mlp.intercepts_)],
        "meta": {"n_frames": int(len(df)), "n_participants": int(n_people), "split_by": split_by,
                 "test_accuracy_frames": round(float(acc), 4), "test_accuracy_recordings": round(float(rec_acc), 4)},
    }
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nSaved {args.out}: {len(out['classes'])} classes, layers "
          f"{' -> '.join(str(len(l['b'])) for l in out['layers'])}")


if __name__ == "__main__":
    main()
