import pandas as pd
import numpy as np
import Orange

# ============================================
# 1) ORANGE → PANDAS
# Převod vstupní Orange tabulky na pandas DataFrame,
# včetně tříd a meta‑informací, aby bylo možné s daty
# pracovat jednotně v jednom prostředí.
# ============================================

df = pd.DataFrame(in_data.X, columns=[v.name for v in in_data.domain.attributes])

for cv in in_data.domain.class_vars:
    df[cv.name] = in_data.get_column(cv)

for mv in in_data.domain.metas:
    df[mv.name] = in_data.get_column(mv)

# ============================================
# 2) INTERNÍ PROMĚNNÁ – AuditZved_D1
# Výpočet rozdílu mezi geodetickým a technologickým
# zdvihem pro zařízení D1 – základní kontrolní ukazatel.
# POZOR! LiftRight/LiftLeft závisí na zvoleném ŘP stroje!
# ============================================

audit_zved = df["zdvih [mm]_G1"] - df["LiftRight_D1"]

# ============================================
# 3) KONFIGURACE
# Nastavení prahové hodnoty a příprava vstupních
# proměnných pro logiku DGS.
# ============================================

DGS_THRESHOLD_MM = 60.0
LIFT = audit_zved
LOAD = pd.to_numeric(df["VerticalLoad_D1"], errors="coerce")
FREQ = pd.to_numeric(df["DGS-Frequency_D1"], errors="coerce")

# ============================================
# 4) LOGIKA DGS
# Vyhodnocení potřeby DGS, dostupnosti dat a jejich
# skutečného použití. Výsledkem je stavový kód 0–4,
# který popisuje kombinaci těchto podmínek.
# ============================================

df["DGS_need_D1"] = (LIFT > DGS_THRESHOLD_MM)
df["DGS_known_D1"] = LOAD.notna() | FREQ.notna()
df["DGS_used_D1"] = (LOAD > 0) | (FREQ > 0)

df["DGS_status_D1"] = np.select(
    [
        (~df["DGS_need_D1"] & df["DGS_known_D1"] & ~df["DGS_used_D1"]),  # 0
        (~df["DGS_need_D1"] & df["DGS_known_D1"] &  df["DGS_used_D1"]),  # 1
        ( df["DGS_need_D1"] & df["DGS_known_D1"] &  df["DGS_used_D1"]),  # 2
        ( df["DGS_need_D1"] & df["DGS_known_D1"] & ~df["DGS_used_D1"]),  # 3
        ( df["DGS_need_D1"] & ~df["DGS_known_D1"])                       # 4
    ],
    [0, 1, 2, 3, 4],
    default=np.nan
)

# ============================================
# 5) REKONSTRUKCE ORANGE TABULKY
# Oddělení numerických a nenumerických sloupců,
# vytvoření odpovídající Orange domény a převod
# zpět do formátu Orange pro další zpracování.
# ============================================

feature_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
meta_cols = [c for c in df.columns if c not in feature_cols]

attributes = [Orange.data.ContinuousVariable(c) for c in feature_cols]
metas = [Orange.data.StringVariable(c) for c in meta_cols]

domain = Orange.data.Domain(attributes, metas=metas)

X = df[feature_cols].to_numpy(dtype=float)
M = df[meta_cols].astype(str).to_numpy() if meta_cols else None

out_data = Orange.data.Table(domain, X, metas=M)
