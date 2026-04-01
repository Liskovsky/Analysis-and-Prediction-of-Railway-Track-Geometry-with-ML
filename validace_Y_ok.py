import numpy as np
import pandas as pd
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame,
# což umožňuje pohodlnější práci s daty a výpočty.
df = pd.DataFrame(in_data.X, columns=[v.name for v in in_data.domain.attributes])

# Přidání tříd a meta‑informací, pokud jsou ve vstupní tabulce.
# Orange je někdy ukládá odděleně, proto je doplňujeme ručně.
for cv in in_data.domain.class_vars:
    df[cv.name] = in_data.get_column_view(cv)[0]

for mv in in_data.domain.metas:
    df[mv.name] = in_data.get_column_view(mv)[0]

# Nastavení tolerancí pro posouzení stability geometrické polohy:
# - TOL_Z: maximální povolená odchylka zdvihu
# - TOL_S: maximální povolená odchylka směrového posunu
TOL_Z = 5.0
TOL_S = 10.0

# Definice cílové proměnné Y:
# Y_OK = 1 → geometrická poloha je stabilní (obě odchylky v toleranci)
# Y_OK = 0 → geometrická poloha je riziková (alespoň jedna odchylka mimo toleranci)
df["Y_OK"] = (
    (df["zdvih [mm]_G3"].abs() <= TOL_Z) &
    (df["posun [mm]_G3"].abs() <= TOL_S)
).astype(int)

# Výběr numerických atributů kromě cílové proměnné.
feature_cols = [
    c for c in df.columns
    if c != "Y_OK" and np.issubdtype(df[c].dtype, np.number)
]

attributes = [Orange.data.ContinuousVariable(c) for c in feature_cols]

# Definice cílové proměnné jako diskrétní (0/1).
class_var = Orange.data.DiscreteVariable("Y_OK", values=["0", "1"])

# Meta‑sloupce: všechny nenumerické atributy, které nechceme použít jako vstupní příznaky.
meta_cols = [c for c in df.columns if c not in feature_cols + ["Y_OK"]]
metas = [Orange.data.StringVariable(c) for c in meta_cols]

# Sestavení Orange domény s atributy, třídou a meta‑informacemi.
domain = Orange.data.Domain(attributes, class_var, metas=metas)

# Převod dat zpět do Orange formátu pro další analýzu nebo modelování.
X = df[feature_cols].to_numpy(dtype=float)
y = df["Y_OK"].astype(str).to_numpy()
M = df[meta_cols].astype(str).to_numpy() if meta_cols else None

out_data = Orange.data.Table(domain, X, y, M)
