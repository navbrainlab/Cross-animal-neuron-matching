import re
import sys
from pathlib import Path
import numpy as np

dataset = sys.argv[1]

root = Path(
    "/home/ubuntu/klb/nuclr/geotransformer_official/"
    "cv5x3_results"
) / dataset

seeds = [1, 42, 123]
folds = [1, 2, 3, 4, 5]

patterns = {
    "Top1": r"Top-1\s*:\s*([0-9.]+)%",
    "Top5": r"Top-5\s*:\s*([0-9.]+)%",
    "MRR": r"MRR\s*:\s*([0-9.]+)",
    "Hung": r"Hungarian Accuracy\s*:\s*([0-9.]+)%",
    "Coverage": r"GT Candidate Coverage\s*:\s*([0-9.]+)%",
}

all_rows = []
fold_means = []

print(f"\n{'='*95}")
print(f"GeoTransformer (adapted) | {dataset.upper()} | 5-fold x seeds {{1,42,123}}")
print(f"{'='*95}")

for fold in folds:
    fold_rows = []

    print(f"\nFold {fold}")

    for seed in seeds:
        base = root / f"fold_{fold}" / f"seed_{seed}"
        test_file = base / "test.txt"
        best_file = base / "best_checkpoint.txt"

        if not test_file.exists():
            raise FileNotFoundError(test_file)

        txt = test_file.read_text(errors="ignore")

        row = {
            "fold": fold,
            "seed": seed,
        }

        for key, pat in patterns.items():
            m = re.search(pat, txt)
            if not m:
                raise RuntimeError(
                    f"Cannot parse {key}: {test_file}"
                )
            row[key] = float(m.group(1))

        iteration = "?"
        val_top1 = "?"

        if best_file.exists():
            btxt = best_file.read_text()

            mi = re.search(r"iter=(\d+)", btxt)
            mv = re.search(r"val_top1=([0-9.]+)", btxt)

            if mi:
                iteration = mi.group(1)
            if mv:
                val_top1 = f"{float(mv.group(1)):.2f}"

        fold_rows.append(row)
        all_rows.append(row)

        print(
            f"  seed={seed:3d} "
            f"best={iteration:>5} "
            f"ValT1={val_top1:>6}  "
            f"TestT1={row['Top1']:6.2f}%  "
            f"T5={row['Top5']:6.2f}%  "
            f"Hung={row['Hung']:6.2f}%  "
            f"MRR={row['MRR']:.4f}  "
            f"Cov={row['Coverage']:6.2f}%"
        )

    fm = {
        key: np.mean([r[key] for r in fold_rows])
        for key in patterns
    }

    fold_means.append(fm)

    print(
        f"  --> fold mean: "
        f"Top1={fm['Top1']:.2f}%  "
        f"Top5={fm['Top5']:.2f}%  "
        f"Hung={fm['Hung']:.2f}%  "
        f"MRR={fm['MRR']:.4f}"
    )


print(f"\n{'='*95}")
print("FINAL — 3 seeds averaged within each fold; mean ± sample SD across 5 folds")
print(f"{'='*95}")

for key in ["Top1", "Top5", "Hung", "MRR", "Coverage"]:
    vals = np.asarray(
        [x[key] for x in fold_means],
        dtype=float
    )

    mean = vals.mean()
    sd = vals.std(ddof=1)

    if key == "MRR":
        print(f"{key:10s}: {mean:.4f} ± {sd:.4f}")
    else:
        print(f"{key:10s}: {mean:.2f}% ± {sd:.2f}%")


print(f"\n{'='*95}")
print("REFERENCE — pooled 15-run mean ± SD")
print(f"{'='*95}")

for key in ["Top1", "Top5", "Hung", "MRR", "Coverage"]:
    vals = np.asarray(
        [x[key] for x in all_rows],
        dtype=float
    )

    if key == "MRR":
        print(
            f"{key:10s}: "
            f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
        )
    else:
        print(
            f"{key:10s}: "
            f"{vals.mean():.2f}% ± {vals.std(ddof=1):.2f}%"
        )
