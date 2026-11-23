"""Evaluate the project's pretrained EfficientNet-B3 classifier on downloaded external datasets.
Usage: python scripts/external_eval.py <project_root> <data_root>
<data_root> must contain Kaggle downloads: data_masoud/ (masoudnickparvar/brain-tumor-mri-dataset),
data_sartaj/ (sartajbhuvaji/brain-tumor-classification-mri), data_44c/ (fernando2rad/brain-tumor-mri-images-44c)."""
import hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

project, scratch = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(project / "src"))
from eval_pretrained_classifier import build_model, CKPT_CLASSES, MEAN, STD  # project's own code

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
tf = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([MEAN], [STD])])
model = build_model()
model.load_state_dict(torch.load(project / "external_models/brain_tumor_classifier/weights/best_model.pth",
                                 map_location="cpu", weights_only=True), strict=True)
model.eval()
torch.set_num_threads(max(1, torch.get_num_threads()))


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


# Hashes of the data the checkpoint (and the project) trained on: Kaggle masoud Training.
train_hashes = {md5(p) for p in (scratch / "data_masoud/Training").rglob("*") if p.suffix.lower() in EXTS}
local_test = {md5(p) for p in (project / "archive/Testing").rglob("*") if p.suffix.lower() in EXTS}
fresh_test = {md5(p) for p in (scratch / "data_masoud/Testing").rglob("*") if p.suffix.lower() in EXTS}
local_train = {md5(p) for p in (project / "archive/Training").rglob("*") if p.suffix.lower() in EXTS}
print(f"fresh Kaggle Testing == local archive/Testing: {fresh_test == local_test} ({len(fresh_test)} unique)")
print(f"fresh Kaggle Training == local archive/Training: {train_hashes == local_train} ({len(train_hashes)} unique)")


def predict(paths):
    out = []
    for i in range(0, len(paths), 32):
        xs = []
        for p in paths[i:i + 32]:
            with Image.open(p) as im:
                xs.append(tf(im))
        with torch.inference_mode():
            out += torch.softmax(model(torch.stack(xs)), 1).tolist()
    return np.array(out)


def evaluate(name, mapping, root, ood=None):
    """mapping: folder name -> true class. ood: folders with no valid label (report prediction distribution)."""
    res = {"dataset": name}
    paths, y = [], []
    for folder, cls in mapping.items():
        for p in sorted((root / folder).iterdir()):
            if p.suffix.lower() in EXTS:
                paths.append(p); y.append(CKPT_CLASSES.index(cls))
    probs = predict(paths)
    pred = probs.argmax(1)
    y = np.array(y)
    in_train = np.array([md5(p) in train_hashes for p in paths])
    from sklearn.metrics import classification_report, confusion_matrix
    labels = sorted(set(y.tolist()))
    res["n"] = len(y)
    res["accuracy"] = float((pred == y).mean())
    res["exact_duplicates_of_training_images"] = int(in_train.sum())
    if (~in_train).any():
        res["accuracy_excluding_training_duplicates"] = float((pred[~in_train] == y[~in_train]).mean())
        res["n_excluding_training_duplicates"] = int((~in_train).sum())
    res["per_class_recall"] = {CKPT_CLASSES[c]: float((pred[y == c] == c).mean()) for c in labels}
    res["confusion_matrix(rows=true,cols=pred glioma/meningioma/notumor/pituitary)"] = \
        {CKPT_CLASSES[c]: confusion_matrix(y, pred, labels=range(4))[c].tolist() for c in labels}
    res["mean_confidence_correct"] = float(probs.max(1)[pred == y].mean())
    if (pred != y).any():
        res["mean_confidence_wrong"] = float(probs.max(1)[pred != y].mean())
    print(f"\n=== {name} ===")
    print(classification_report(y, pred, labels=labels, target_names=[CKPT_CLASSES[c] for c in labels],
                                digits=4, zero_division=0))
    print(json.dumps({k: v for k, v in res.items() if k != "dataset"}, indent=1))
    if ood:
        dist = {}
        for folder in ood:
            ps = [p for p in sorted((root / folder).iterdir()) if p.suffix.lower() in EXTS]
            if ps:
                pr = predict(ps).argmax(1)
                dist[folder] = dict(Counter(CKPT_CLASSES[i] for i in pr))
        res["out_of_scope_tumour_predictions"] = dist
        print("out-of-scope tumour types (no correct 4-class label):")
        for k, v in dist.items():
            print(f"  {k:28s} {v}")
    return res


results = []
m = scratch / "data_masoud/Testing"
results.append(evaluate("Kaggle masoudnickparvar Testing (fresh download)", {c: c for c in CKPT_CLASSES}, m))

s = scratch / "data_sartaj/Testing"
results.append(evaluate("Kaggle SARTAJ Testing", {"glioma_tumor": "glioma", "meningioma_tumor": "meningioma",
                                                   "no_tumor": "notumor", "pituitary_tumor": "pituitary"}, s))

f = scratch / "data_44c"
mapping = {}
for d in f.iterdir():
    if not d.is_dir():
        continue
    base = d.name.rsplit(" ", 1)[0]
    if base == "Meningioma":
        mapping[d.name] = "meningioma"
    elif base == "_NORMAL":
        mapping[d.name] = "notumor"
    elif base in ("Astrocitoma", "Glioblastoma", "Oligodendroglioma"):
        mapping[d.name] = "glioma"
ood = sorted(d.name for d in f.iterdir() if d.is_dir() and d.name not in mapping)
results.append(evaluate("Kaggle fernando2rad 44-class (independent source; glioma family / meningioma / normal)",
                        mapping, f, ood=ood))

(scratch / "external_eval_results.json").write_text(json.dumps(results, indent=2))
print("\nwritten", scratch / "external_eval_results.json")
