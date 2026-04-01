import pandas as pd
import numpy as np
import Orange

# Převod vstupní Orange tabulky na pandas DataFrame,
# včetně tříd a meta‑informací, aby bylo možné data
# jednotně zpracovávat v jednom prostředí.
def to_df(table):
    if not isinstance(table, Orange.data.Table):
        return None
    try:
        df = pd.DataFrame(table.X, columns=[v.name for v in table.domain.attributes])
        for cv in table.domain.class_vars:
            df[cv.name] = table.get_column_view(cv)[0]
        for mv in table.domain.metas:
            df[mv.name] = table.get_column_view(mv)[0]
        return df
    except:
        return None

# Detekce všech dostupných vstupních tabulek (hlavní vstup,
# Extra Data i více vstupů). Ukládáme pouze ty, které lze
# úspěšně převést na pandas DataFrame.
found_dfs = []

if 'in_data' in locals() and in_data is not None:
    d = to_df(in_data)
    if d is not None:
        found_dfs.append(d)

if 'in_others' in locals() and in_others:
    for item in in_others:
        d = to_df(item)
        if d is not None:
            found_dfs.append(d)

if 'in_datas' in locals() and in_datas:
    for item in in_datas:
        d = to_df(item)
        if d is not None:
            found_dfs.append(d)

# Pokud nebyla nalezena žádná data, ukončíme skript.
if not found_dfs:
    print("!!! CHYBA: Skript nevidí žádná data. Zkontroluj propojení do Extra Data.")
    out_data = None

else:
    print(f"Nalezeno {len(found_dfs)} tabulek. Spojuji...")

    # První tabulka slouží jako základ pro sloučení.
    final_df = found_dfs[0]

    # Postupné sloučení dalších tabulek:
    # - pokud existuje společný klíč Pos_m, použijeme merge
    # - jinak tabulky spojíme horizontálně
    for i in range(1, len(found_dfs)):
        if 'Pos_m' in final_df.columns and 'Pos_m' in found_dfs[i].columns:
            final_df = pd.merge(
                final_df, found_dfs[i],
                on='Pos_m', how='outer', suffixes=('', '_DUP')
            )
        else:
            final_df = pd.concat([final_df, found_dfs[i]], axis=1)

    # Odstranění duplicitních sloupců vzniklých při merge
    # nebo importu z Orange.
    final_df = final_df.loc[:, ~final_df.columns.str.contains('_DUP')]
    final_df = final_df.loc[:, ~final_df.columns.str.contains(r'\(\d+\)$|\.\d+$')]
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # Výpočet cílové proměnné Y_OK:
    # Y_OK = 1 → geometrická poloha je stabilní (obě odchylky v toleranci)
    # Y_OK = 0 → riziková (alespoň jedna odchylka mimo toleranci)
    TOL_Z = 5.0   # tolerance zdvihu [mm]
    TOL_S = 10.0  # tolerance posunu [mm]
    
    z_col, s_col = "zdvih [mm]_G3", "posun [mm]_G3"
    
    if z_col in final_df.columns and s_col in final_df.columns:
        final_df["Y_OK"] = (
            (final_df[z_col].abs() <= TOL_Z) &
            (final_df[s_col].abs() <= TOL_S)
        ).astype(int)
    else:
        final_df["Y_OK"] = 0

    # Rozdělení sloupců na atributy a meta‑informace.
    exclude = ["Y_OK", "Pos_m", "stan [km]", "Pos"]
    attr_names = [
        c for c in final_df.columns
        if c not in exclude and np.issubdtype(final_df[c].dtype, np.number)
    ]
    meta_names = [c for c in final_df.columns if c not in attr_names + ["Y_OK"]]

    # Sestavení Orange domény: numerické atributy, cílová třída a meta‑sloupce.
    domain = Orange.data.Domain(
        [Orange.data.ContinuousVariable(c) for c in attr_names],
        [Orange.data.DiscreteVariable("Y_OK", values=["0", "1"])],
        metas=[Orange.data.StringVariable(c) for c in meta_names]
    )

    # Převod zpět do Orange Table pro další analýzu nebo modelování.
    out_data = Orange.data.Table(
        domain,
        final_df[attr_names].values.astype(float),
        final_df["Y_OK"].astype(str).values,
        final_df[meta_names].astype(str).values
    )

    print(f"ÚSPĚCH: Sloučeno {len(found_dfs)} tabulek do {len(final_df.columns)} sloupců.")
