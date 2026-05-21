(MIGLIORARE LA STEREO POSE ESTIMATION ricalibrando)

shapeDBA vs DBA 

togliere human demo dai grafici

fare più run di addestramento e calcolare media e varianza, se varianza è bassa è buono

inserire peak angle error e smoothness nella heatmap
spider chart per la heatmap, define boundariess
ripeti tutte le valutazioni

mettere ordine nella repo 

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