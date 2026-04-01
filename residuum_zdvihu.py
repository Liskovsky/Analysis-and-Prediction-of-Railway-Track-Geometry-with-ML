import pandas as pd
import numpy as np
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame, což umožňuje pohodlnější práci s daty a výpočty.
df = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])

# Výpočet rozdílu mezi geodetickým a technologickým zdvihem
# pro zařízení D1 a D2 – slouží jako kontrola shody měření.
df["AuditZved_D1"] = df["zdvih [mm]_G1"] - df["LiftRight_D1"]
df["AuditZved_D2"] = df["zdvih [mm]_G2"] - df["LiftRight_D2"]

# Výpočet relativního rezidua zdvihu pro D1:
# - pokud je technologický zdvih 0, nelze počítat procenta → nastavíme 0
# - jinak počítáme procentuální rozdíl mezi technologickým a geodetickým měřením
df["Rz_D1"] = np.where(
    df["LiftRight_D1"] == 0, 0,
    ((df["LiftRight_D1"] - df["zdvih [mm]_G2"]) / df["LiftRight_D1"]) * 100
)

# Stejný výpočet relativního rezidua pro zařízení D2.
df["Rz_D2"] = np.where(
    df["LiftRight_D2"] == 0, 0,
    ((df["LiftRight_D2"] - df["zdvih [mm]_G3"]) / df["LiftRight_D2"]) * 100
)

# Rozdíl reziduí mezi D2 a D1 – ukazatel asymetrie nebo nesouladu zdvihů.
df["PerRz"] = df["Rz_D2"] - df["Rz_D1"]

# Vytvoření Orange domény obsahující pouze numerické atributy,
# aby bylo možné data dále zpracovávat v Orange Data Mining.
attributes = [Orange.data.ContinuousVariable(name) for name in df.columns]
domain = Orange.data.Domain(attributes)

# Převod zpět do Orange tabulky pro další analýzu nebo modelování.
out_data = Orange.data.Table(domain, df.to_numpy())
