import pandas as pd
import numpy as np
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame pro pohodlnější práci s daty.
df = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])

# Přepočet posunů z měřicích zařízení D1 a D2 – změna znaménka pro sjednocení směru.
df["UpravenyPosun_D1"] = df["FrontOffset_D1"] * -1
df["UpravenyPosun_D2"] = df["FrontOffset_D2"] * -1

# Výpočet relativního rezidua Rs_D1:
# - pokud je posun menší než 3 mm, považujeme ho za nevýznamný a nastavíme 0
# - jinak počítáme procentuální rozdíl mezi geodetickým posunem a technologickým měřením
df["Rs_D1"] = np.where(
    np.abs(df["UpravenyPosun_D1"]) < 3,
    0,
    ((df["UpravenyPosun_D1"] - df["posun [mm]_G2"]) / df["UpravenyPosun_D1"]) * 100
)

# Stejný výpočet rezidua pro druhé zařízení D2.
df["Rs_D2"] = np.where(
    np.abs(df["UpravenyPosun_D2"]) < 3,
    0,
    ((df["UpravenyPosun_D2"] - df["posun [mm]_G3"]) / df["UpravenyPosun_D2"]) * 100
)

# Výpočet rozdílu reziduí mezi D2 a D1 – indikátor asymetrie nebo nesouladu měření.
df["PerRs"] = df["Rs_D2"] - df["Rs_D1"]

# Vytvoření nové Orange domény obsahující pouze numerické atributy.
attributes = [Orange.data.ContinuousVariable(name) for name in df.columns]
domain = Orange.data.Domain(attributes)

# Převod zpět do Orange tabulky pro další zpracování v Orange Data Mining.
out_data = Orange.data.Table(domain, df.to_numpy())
