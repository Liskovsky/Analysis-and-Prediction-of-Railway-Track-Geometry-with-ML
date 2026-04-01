import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import os

# Vstupní objekt Random Forestu a datová tabulka z Orange.
rf   = in_object
data = in_data

# Názvy příznaků a tříd pro popisky ve vizualizaci.
feature_names = [attr.name for attr in data.domain.attributes]
class_names   = [str(v) for v in data.domain.class_var.values]

# Extrakce jednotlivých sklearn stromů z Orange Random Forest modelu.
# (Orange RF je wrapper, skutečné modely jsou v atributu skl_model.)
skl_trees = [t.skl_model for t in rf.trees if hasattr(t, "skl_model")]

# Příznak, podle kterého chceme vybrat reprezentativní strom, volba dle významnosti příznaků.
target_feature = "Zdvih [mm]_G3"

# Funkce ověřující, zda daný strom používá konkrétní příznak.
def uses_feature(tm, fname):
    try:
        return feature_names.index(fname) in tm.tree_.feature
    except ValueError:
        return False

# Výběr stromu:
# - pokud existuje strom, který používá target_feature, vezmeme ho
# - jinak vezmeme první strom jako fallback
selected = [t for t in skl_trees if uses_feature(t, target_feature)] or skl_trees[:1]
t = selected[0]

print(f"Hloubka stromu: {t.get_depth()}, uzlů: {t.tree_.node_count}")

# Odhad šířky obrázku podle počtu listů, aby vizualizace byla čitelná.
n_leaves = min(2**4, t.tree_.n_node_samples.shape[0])
fig_w = max(11.69, n_leaves * 1.4)
fig_h = 4 * 1.8  # 4 úrovně × výška na úroveň

fig, ax = plt.subplots(figsize=(fig_h, fig_w), dpi=150)

# Vykreslení rozhodovacího stromu:
# - max_depth=3 pro přehlednost
# - filled=True pro barevné rozlišení tříd
# - rounded=True pro čitelnější tvary uzlů
tree.plot_tree(
    t,
    feature_names = feature_names,
    class_names   = class_names,
    max_depth     = 3,
    impurity      = False,
    filled        = True,
    rounded       = True,
    fontsize      = 7,
    precision     = 1,
    ax            = ax,
)

# Titulek s informací o příznaku a skutečné hloubce stromu.
ax.set_title(
    f"Rozhodovací logika RF – příznak: {target_feature}  |  "
    f"zobrazeno: 3 úrovně z {t.get_depth()}",
    fontsize=9, fontweight="bold", pad=10,
)

# Doplňkový text s informací o třídách a počtu uzlů.
fig.text(
    0.5, 0.01,
    f"Třídy: {', '.join(class_names)}  |  Uzlů celkem: {t.tree_.node_count}",
    ha="center", fontsize=6, color="gray"
)

# Úprava rozložení, aby se popisky nepřekrývaly.
plt.tight_layout(rect=[0, 0.03, 1, 1])

plt.show()
