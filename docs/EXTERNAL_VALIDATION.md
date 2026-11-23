# System check and external validation (2026-11-23)

Environment: Windows 11, Python 3.12.7, torch 2.15.0+cpu, torchvision 0.29.2+cpu, CPU only.
Classifier weights restored from `external_models/brain_tumor_classifier_source_main.zip`; SHA-256 `7840ed87...7c4a` matches `external_models/README.md`.

## 1. Does the system run?

| Check | Result |
|---|---|
| `src/check_environment.py` | Pass (CPU) |
| `src/inspect_dataset.py --data-dir archive` | Pass: 5,600 train / 1,600 test, 1,400 / 400 per class |
| `src/data_pipeline.py` | Pass: 0 corrupt, batch (16,1,64,64) in [-1,1] |
| `src/eval_pretrained_classifier.py` | Pass: reproduces the documented **0.9594 acc / 0.9587 macro-F1** exactly |
| `app/backend/tests` (pytest) | 11 / 16 pass. All classify, validation, CORS and error-handling tests pass. The 5 failures (`models_load`, `metrics`, `experiment`, `generate`, `gallery`) all need `outputs/deduplicated_experiment/` (cWGAN-GP checkpoint + saved result files), which is git-ignored and not present in this copy. They are not code bugs; they need the GAN to be retrained or the outputs folder restored. |

## 2. Is it identifying the class correctly? (data downloaded from Kaggle)

`scripts/external_eval.py` (results: `docs/external_eval_results.json`)

| Test set | n | Accuracy | Notes |
|---|---|---|---|
| masoudnickparvar Testing (the dataset the project proposes, fresh download) | 1,600 | **0.959** | Byte-identical to local `archive/Testing`. Glioma recall 0.85, others >= 0.99 |
| SARTAJ Testing (one of masoud's source datasets) | 394 | 0.805 (0.747 excluding 141 exact copies of masoud Training) | Glioma recall only 0.40 (misread as meningioma/no tumour). This test set has known label noise. |
| fernando2rad 44-class (**independent source**, mapped: astrocytoma/glioblastoma/oligodendroglioma -> glioma, meningioma, normal -> no tumour) | 2,403 | **0.425** | Glioma recall 0.14, meningioma 0.45, no tumour 0.93. Most errors are predicted "no tumour". |

Other tumour types in the 44-class set (schwannoma, neurocytoma, etc.) have no correct label; the model still gives one of the four classes (mostly "no tumour") with high confidence. There is no out-of-distribution rejection.

## Conclusion

- On the data it was designed for, the classifier works and the reported numbers are reproducible.
- It does **not** generalise to MRI from other sources: accuracy falls to 43% and many real tumours are called "no tumour". Glioma is the weakest class everywhere.
- This supports the project's own caveats (README Phase 6/7, `docs/WEB_APP_AUDIT.md`): the 95.9% is optimistic and the system is not suitable for clinical use.
