# PyMOL script — FMDV serotype A VP1 positions
# 2WZR was not used: it is serotype SAT1, it has no chain D,
# and its VP1 identity to A22/IRQ/24/64 is 56.1%.
# Structure: 4GH4, FMDV A22 Iraq. VP1 is chain A.
# Reference sequence: A22/IRQ/24/64. Identity 210/210 = 100.0%.
# Residue numbers are the PDB author numbers.

fetch 4GH4
remove solvent
hide everything
show cartoon, chain A
color gray80, chain A

select gh_loop, chain A and resi 132-161
color salmon, gh_loop

# High importance (mean |SHAP| >= 0.04), coordinates present
select high_shap, chain A and resi 173+197
show sticks, high_shap
color red, high_shap

# Medium importance (0.02 <= mean |SHAP| < 0.04), coordinates present
select medium_shap, chain A and resi 46+83+95+96+101+167+171+193
show sticks, medium_shap
color orange, medium_shap

# Low importance (mean |SHAP| < 0.02), coordinates present
select low_shap, chain A and resi 3+40+93+164
show sticks, low_shap
color yellow, low_shap

# These confirmed positions map to PDB numbers but have no coordinates (disordered loop):
# alignment 140 G -> 4GH4 chain A resi 140 (low SHAP)
# alignment 142 G -> 4GH4 chain A resi 142 (low SHAP)
# alignment 143 R -> 4GH4 chain A resi 143 (low SHAP)
# alignment 149 P -> 4GH4 chain A resi 149 (high SHAP)

zoom chain A
set ray_shadows, 0
ray 1200, 900
png vp1_structure_shap_positions.png, dpi=300
