import pandas as pd
import numpy as np
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame,
# což umožňuje pohodlnější práci s daty a výpočty.
df = pd.DataFrame(in_data.X, columns=[var.name for var in in_data.domain.attributes])

# Výpočet asymetrie zdvihu pro zařízení D1 –
# rozdíl mezi pravým a levým technologickým zdvihem.
df["ZAsymetrie_D1"] = df["LiftRight_D1"] - df["LiftLeft_D1"]

# Vytvoření Orange domény obsahující pouze numerické atributy,
# aby bylo možné data dále zpracovávat v Orange Data Mining.
attributes = [Orange.data.ContinuousVariable(name) for name in df.columns]
domain = Orange.data.Domain(attributes)

# Převod zpět do Orange tabulky pro další analýzu nebo modelování.
out_data = Orange.data.Table(domain, df.to_numpy())
