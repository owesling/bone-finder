# Bone Finder

A webcam anatomy app that runs in the browser. Point at part of your body and it names the bone underneath, with a short description and a fact about it.

The app uses [MediaPipe](https://developers.google.com/mediapipe) pose and hand tracking. All tracking happens in the browser tab, and no video leaves your computer.

## What's in this folder

| Path | What it is |
| --- | --- |
| `bone-finder.html` | The whole app in one file: tracking, the rule-based classifier, Explore/Quiz/Collect modes, and the code that runs a trained model. |
| `train.py` | Trains a small neural network on CSVs from Collect mode and exports `model.json` for the app. |
| `synth_osso.py` | Builds synthetic training rows from SMPL body meshes and OSSO skeletons, for pretraining. |
| `OSSO/` | A local copy of [OSSO](https://github.com/MarilynKeller/OSSO) (CVPR 2022), which infers a skeleton from a body mesh. Has its own `osso_venv`. |
| `mesh/` | The [MPI-IS mesh](https://github.com/MPI-IS/mesh) package, which OSSO needs. |
| `bonefinder_P01_*.xlsx` | A recorded data session. |
| `bone_classifier_*.mp4` | Screen recordings of the app. |
| `.venv/` | Python environment for `train.py` / `synth_osso.py`. |

## Running the app

1. Serve the folder over HTTP. Browsers block the webcam on `file://` pages.
   ```
   python -m http.server 8000
   ```
2. Open <http://localhost:8000/bone-finder.html> in Chrome or Edge and click **Start camera**.
3. Frame yourself so your head, shoulders and hips are in view. Step back to include knees and feet.
4. Make a pointing hand (index finger straight, other fingers curled), rest the fingertip on your body and hold still until the ring fills.

To find hand bones, hold your other hand up with the palm facing the camera and point at its fingers, knuckles or wrist. Left and right always mean *your* left and right.

You need an internet connection the first time, because MediaPipe (`@mediapipe/tasks-vision@0.10.14`) and its pose and hand models load from a CDN.

### Modes

- **Explore**: point anywhere. Each new bone you find goes on a list.
- **Quiz**: the app names a bone and you find it.
- **Collect data**: records labeled training examples (see below).

**Settings** lets you mirror the camera, show the skeleton overlay, require a pointing gesture, read bone names aloud, and change the hold time.

## Training a model

Out of the box, the app uses hand-written geometric rules. A trained model can be more accurate.

### 1. Collect data

In **Collect data** mode:

1. Enter a participant code (not a name) and notes on lighting, clothing and camera distance.
2. Choose the body regions to include and the number of rounds.
3. Click **Start session**. The app asks the participant to touch specific points and records about one second of tracking for each one. Bone labels stay hidden during collection so they don't influence where people point.
4. Click **Download CSV** when you're done.

Record several different people. The training script tests on people it hasn't seen, so the reported accuracy is only meaningful with 3 or more participants.

### 2. Train

```
pip install numpy pandas scikit-learn
python train.py recordings/*.csv --out model.json
```

The script:

- doubles the data by mirroring it (a left-side example, flipped, is a right-side example),
- holds out 25% of participants for testing (or 25% of recordings, with a warning, if there are fewer than 3 participants),
- prints per-frame and per-recording accuracy, a k-nearest-neighbors baseline, per-class results and the most common mix-ups,
- retrains on all the data and writes `model.json`.

Use `--min-gesture 1` to drop frames where no pointing gesture was detected.

### 3. Load it

In the app, expand **Trained model**, load `model.json` and tick **Use the trained model instead of rules**. **Inspect the model's input and output** shows the live features and predictions.

## Synthetic pretraining data (optional)

`synth_osso.py` builds a large labeled dataset without recording anyone. It samples varied SMPL bodies, runs OSSO to get the skeleton beneath each one, places virtual webcams around them, "touches" visible skin points and labels each touch with the nearest bone. The output CSV has the same 86 `f_` features as the app, so `train.py` reads it directly.

Run the stages in order. They share a working folder (`--work`, default `synth_work`):

```
python synth_osso.py bodies   --smpl-dir path/to/models --n 60
python synth_osso.py osso     --osso-dir OSSO --python OSSO/osso_venv/Scripts/python.exe
python synth_osso.py parts    # writes part_map.json, so check it before continuing
python synth_osso.py preview  --body 0 --out preview.png
python synth_osso.py rows     --out synthetic.csv
```

| Stage | What it does |
| --- | --- |
| `bodies` | Samples SMPL body shapes in a relaxed standing pose and saves each mesh with estimated MediaPipe landmark positions. Needs `torch`, `smplx` and the SMPL model files (`smpl/SMPL_FEMALE.pkl`, `SMPL_MALE.pkl`). |
| `landmarks` | Recomputes landmarks after you edit `landmark_config.json`, without rerunning OSSO. |
| `osso` | Runs OSSO's `main.py --per_part` on each body and collects the bone meshes for each part. |
| `parts` | Lists the parts OSSO produced and writes `part_map.json`, which maps each part to a Bone Finder class or to a splitter (`@skull`, `@thorax`, `@pelvis`, ...) that divides a large part by where it was touched. |
| `preview` | Draws one body's labeled touch points and landmarks to a PNG so you can check the labels and sides. |
| `rows` | Writes the training CSV. Run `python synth_osso.py rows -h` to see options for camera distance, noise, depth limits and class balance. |

Each synthetic body is recorded as its own participant (`SYN0000`, `SYN0001`, ...). Combine the synthetic data with real recordings when training:

```
python train.py synthetic.csv recordings/*.csv --out model.json
```

Other requirements: `numpy`, `scipy`, `matplotlib` (for `preview`), and `trimesh` for binary mesh formats.

### Setting up OSSO

See [OSSO/installation.md](OSSO/installation.md). In short, OSSO needs Python 3.8, the `gloss_skeleton` code, the compiled MPI `mesh` package (which needs Boost) and the STAR model files, which require an account at star.is.tue.mpg.de. OSSO is only tested on Ubuntu, so on Windows, WSL is the easiest way to run it.

## How classification works

For each frame, the app builds a feature vector from the pointing fingertip's position relative to 19 body landmarks (nose, eyes, ears, mouth, shoulders, elbows, wrists, hips, knees, ankles). The vector also records which landmarks are visible, which hand is pointing, the finger's direction, and the position and shape of the other hand. The rule-based classifier maps these to head, torso, arm and leg zones. The trained model is a 64×64 ReLU MLP whose weights `train.py` exports to JSON, and it runs in plain JavaScript in the page.

The feature definitions (`FEAT_LM`, `FEATURE_NAMES`, `META_COLS`, `RAW_COLS`) are duplicated in `bone-finder.html` and `synth_osso.py`. **If you change them in one file, change them in the other**, and retrain.

## Licenses

OSSO, SMPL, STAR and anything generated from them (including synthetic CSVs and models trained on them) are licensed **for non-commercial research only**. See [OSSO/LICENSE.txt](OSSO/LICENSE.txt) and [mesh/LICENSE.txt](mesh/LICENSE.txt).
