import pandas as pd
import numpy as np
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame,
# což umožňuje pohodlnější práci s daty a výpočty.
df = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])

# Výpočet driftu mezi dvěma po sobě jdoucími geodetickými měřeními:
# - DriftZ: změna zdvihu mezi G2 a G3
# - DriftS: změna posunu mezi G2 a G3
# Tyto hodnoty slouží jako indikátory krátkodobé stability geometrické polohy.
df["DriftZ"] = df["zdvih [mm]_G3"] - df["zdvih [mm]_G2"]
df["DriftS"] = df["posun [mm]_G3"] - df["posun [mm]_G2"]

# Vytvoření Orange domény obsahující pouze numerické atributy,
# aby bylo možné data dále zpracovávat v Orange Data Mining.
attributes = [Orange.data.ContinuousVariable(name) for name in df.columns]
domain = Orange.data.Domain(attributes)

# Převod zpět do Orange tabulky pro další analýzu nebo modelování.
out_data = Orange.data.Table(domain, df.to_numpy())
