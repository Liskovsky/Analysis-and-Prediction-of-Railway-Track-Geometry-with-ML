import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import os

# Vstupní objekt Gradient Boosting modelu a datová tabulka z Orange.
gb   = in_object
data = in_data

# Názvy příznaků pro popisky ve vizualizaci.
feature_names = [attr.name for attr in gb.domain.attributes]

# Příznak, podle kterého chceme vybrat reprezentativní stromy, volba dle významnosti příznaků..
target_feature = "Zdvih [mm]_G3"

# Maximální počet stromů, které chceme zobrazit.
max_trees = 3

# Extrakce jednotlivých sklearn stromů z Gradient Boosting modelu.
# (Orange GB ukládá stromy jako dvojice, první prvek je skutečný strom.)
real_trees = [e[0] for e in gb.skl_model.estimators_]

# Funkce ověřující, zda daný strom používá konkrétní příznak.
def uses_feature(tm, fname):
    try:
        return feature_names.index(fname) in tm.tree_.feature
    except ValueError:
        return False

# Výběr stromů:
# - pokud existují stromy, které používají target_feature, vezmeme je
# - jinak vezmeme prvních max_trees stromů jako fallback
selected = [t for t in real_trees if uses_feature(t, target_feature)][:max_trees]
if not selected:
    selected = real_trees[:max_trees]

print(f"Zobrazuji {len(selected)} stromů")

# Vykreslení každého vybraného stromu.
for i, t in enumerate(selected):

    # Odhad šířky obrázku podle počtu listů, aby vizualizace byla čitelná.
    n_leaves = min(2**3, t.tree_.n_node_samples.shape[0])
    fig_w    = max(11.69, n_leaves * 1.4)
    fig_h    = 3 * 1.8  # 3 úrovně × výška na úroveň

    fig, ax = plt.subplots(figsize=(fig_h, fig_w), dpi=150)

    # Vykreslení rozhodovacího stromu:
    # - max_depth=3 pro přehlednost
    # - filled=True pro barevné rozlišení
    # - rounded=True pro čitelnější tvary uzlů
    tree.plot_tree(
        t,
        feature_names = feature_names,
        max_depth     = 3,
        impurity      = False,
        filled        = True,
        rounded       = True,
        fontsize      = 7,
        precision     = 1,
        ax            = ax,
    )

    # Úprava rozložení, aby se popisky nepřekrývaly.
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    plt.show()
