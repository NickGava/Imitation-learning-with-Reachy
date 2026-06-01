(MIGLIORARE LA STEREO POSE ESTIMATION ricalibrando)             (✓)

shapeDBA vs DBA         ✓

togliere human demo dai grafici         ✓

inserire peak wrist distance error e smoothness nella heatmap        ✓
spider chart per la heatmap, define boundariess                      ✓

rifare tutta la data acquisition di stereo ✓ e mix ✓

*fare più run di addestramento e calcolare media e varianza, se varianza è bassa è buono:*

py run_compute_canonical.py --n-demos 10        ✓
py run_compute_canonical.py --n-demos 25        ✓
py run_compute_canonical.py --n-demos 55        ✓

ricarica il dataset su kaggle       ✓

!python run_bc_approach.py --exercise a --start 1 --end 5 --training-runs 5 --n-demos 10 and save       ✓
!python run_bc_approach.py --exercise a --start 1 --end 5 --training-runs 5 --n-demos 25 and save       ✓
!python run_bc_approach.py --exercise a --start 1 --end 5 --training-runs 5 --n-demos 55 and save       ✓

!python run_bc_approach.py --exercise a --start 11 --end 15 --training-runs 5 --n-demos 10 and save     ✓
!python run_bc_approach.py --exercise a --start 11 --end 15 --training-runs 5 --n-demos 25 and save     ✓
!python run_bc_approach.py --exercise a --start 11 --end 15 --training-runs 5 --n-demos 55 and save     ✓

!python run_bc_approach.py --exercise a --start 21 --end 25 --training-runs 5 --n-demos 10 and save     ✓
!python run_bc_approach.py --exercise a --start 21 --end 25 --training-runs 5 --n-demos 25 and save     ✓
!python run_bc_approach.py --exercise a --start 21 --end 25 --training-runs 5 --n-demos 55 and save     ✓


py -m evaluation_and_comparison.evaluate --all --n-demos 10     ✓
py -m evaluation_and_comparison.evaluate --all --n-demos 25     ✓
py -m evaluation_and_comparison.evaluate --all --n-demos 55     ✓

py -m evaluation_and_comparison.evaluate_demos --all            ✓

py -m evaluation_and_comparison.evaluate_modality --n-demos 55    ✓  

mettere ordine nella repo       ✓

implementare la sicurezza - check

## PER PROVARE ESERCIZIO --------------------------------------------------------------------------------------------------------------------
# 1. Canonical — tre split
py run_compute_canonical.py --exercise 21 --n-demos 10
py run_compute_canonical.py --exercise 21 --n-demos 25
py run_compute_canonical.py --exercise 21 --n-demos 55

# 2. BC pipeline (build dataset + train + test) — tre split
py run_bc_approach.py --exercise 21 --n-demos 10
py run_bc_approach.py --exercise 21 --n-demos 25
py run_bc_approach.py --exercise 21 --n-demos 55

# 3. Evaluation — tutti e tre i split in un colpo
py -m evaluation_and_comparison.evaluate --exercise 21 --n-demos 10 25 55

# 4. Ablation study demos
py -m evaluation_and_comparison.evaluate_demos --exercise 21

## TOTALE --------------------------------------------------------------------------------------------------------------------------------------
# 1. Canonical
py run_compute_canonical.py --n-demos 10        ✓
py run_compute_canonical.py --n-demos 25        ✓
py run_compute_canonical.py --n-demos 55        ✓

# 2. BC pipeline
py run_bc_approach.py --exercise a --n-demos 10     ✓
py run_bc_approach.py --exercise a --n-demos 25     ✓
py run_bc_approach.py --exercise a --n-demos 55     ✓

# 3. Evaluation per esercizio (tutti e tre i split)
py -m evaluation_and_comparison.evaluate --all --n-demos 10     ✓
py -m evaluation_and_comparison.evaluate --all --n-demos 25     ✓
py -m evaluation_and_comparison.evaluate --all --n-demos 55     ✓

# 4. Ablation study demos (per-esercizio + globale) 
py -m evaluation_and_comparison.evaluate_demos --all            ✓

# 5. Analisi modality (Stereo vs Mixed vs Mono) 
py -m evaluation_and_comparison.evaluate_modality --n-demos 55    ✓  


## KAGGLE --------------------------------------------------------------------------------------------------------------------------------------

!python run_bc_approach.py --exercise a --start 11 --end 15 --training-runs 5 --n-demos 10
# ----
import zipfile
from pathlib import Path

results_root = DATA_ROOT / 'dataset'
zip_path     = Path('/kaggle/working/results_es11_15_n10.zip')

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for ex in range(11, 16):
        ex_dir = results_root / f'exercise_{ex:03d}'
        for path in sorted(ex_dir.rglob('*')):
            if not path.is_file():
                continue
            if 'n_10' not in path.parts:
                continue
            zf.write(path, path.relative_to(results_root.parent))

print(f'Zip creato: {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)')
# --- per unzippare:
Expand-Archive ~/Downloads/results_es1_5_n10.zip -DestinationPath .