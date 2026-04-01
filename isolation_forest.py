import pandas as pd
import numpy as np
import Orange
from sklearn.ensemble import IsolationForest

# Pokud nejsou k dispozici žádná vstupní data, skript čeká na propojení.
if in_data is None:
    out_data = None
    print("Čekám na data z předchozího uzlu...")

else:
    # Převod vstupní Orange tabulky na pandas DataFrame,
    # včetně tříd a meta‑informací, aby bylo možné data
    # jednotně zpracovávat v jednom prostředí.
    def to_df(table: Orange.data.Table) -> pd.DataFrame:
        df = pd.DataFrame(table.X, columns=[v.name for v in table.domain.attributes])
        for cv in table.domain.class_vars:
            df[cv.name] = table.get_column(cv)
        for mv in table.domain.metas:
            df[mv.name] = table.get_column(mv)
        return df

    df = to_df(in_data)

    # Výběr numerických příznaků pro model Isolation Forest.
    # Vylučujeme pozici a další sloupce, které nemají být použity
    # jako vstupní proměnné pro detekci anomálií.
    exclude_for_if = {"Y_OK", "Pos_m", "stan [km]", "Pos"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude_for_if and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        out_data = None
        raise ValueError("Nenašel jsem žádné numerické feature sloupce pro Isolation Forest.")

    # Náhrada chybějících hodnot mediánem – model IF nepracuje s NaN.
    X_model = df[feature_cols].copy()
    X_model = X_model.fillna(X_model.median(numeric_only=True))

    # Trénink modelu Isolation Forest pro detekci anomálií.
    # Výstupem je skóre (IF_score) a binární příznak (IF_flag).
    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42
    )
    iso.fit(X_model.to_numpy())

    df["IF_score"] = iso.decision_function(X_model.to_numpy())
    df["IF_flag"] = iso.predict(X_model.to_numpy())  # 1 = normální, -1 = anomálie

    # Výběr sloupců, které chceme mít jako atributy ve výstupní tabulce.
    # Pozice (Pos_m, stan [km]) ponecháváme kvůli vizualizacím v Orange.
    extra_attr = ["Pos_m", "stan [km]"]
    extra_attr = [c for c in extra_attr if c in df.columns]

    # Atributy = pozice + feature sloupce + skóre modelu
    attr_names = extra_attr + feature_cols + ["IF_score"]
    attr_vars = [Orange.data.ContinuousVariable(c) for c in attr_names]

    # IF_flag ukládáme jako diskrétní meta‑informaci.
    if_flag_var = Orange.data.DiscreteVariable("IF_flag", values=["-1", "1"])

    # Pokud existuje cílová proměnná Y_OK, přidáme ji jako class.
    class_var = None
    if "Y_OK" in df.columns:
        class_var = Orange.data.DiscreteVariable("Y_OK", values=["0", "1"])

    # Meta‑sloupce = vše, co není atribut, třída nebo IF_flag.
    used = set(attr_names) | {"IF_flag"}
    if class_var is not None:
        used |= {"Y_OK"}

    meta_cols = [c for c in df.columns if c not in used]

    # Meta proměnné: numerické jako Continuous, ostatní jako String.
    meta_vars = []
    for c in meta_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            meta_vars.append(Orange.data.ContinuousVariable(c))
        else:
            meta_vars.append(Orange.data.StringVariable(c))

    # Sestavení Orange domény s atributy, třídou a meta‑informacemi.
    domain = Orange.data.Domain(
        attr_vars,
        class_var,
        metas=[if_flag_var] + meta_vars
    )

    # Matice atributů (X)
    X_parts = []
    if extra_attr:
        X_parts.append(df[extra_attr].to_numpy(dtype=float))
    X_parts.append(df[feature_cols].to_numpy(dtype=float))
    X_parts.append(df[["IF_score"]].to_numpy(dtype=float))
    X_out = np.hstack(X_parts)

    # Třída (Y)
    Y_out = None
    if class_var is not None:
        Y_out = df["Y_OK"].astype(int).astype(str).to_numpy()

    # Meta‑informace (IF_flag + ostatní meta sloupce)
    metas_parts = [df["IF_flag"].astype(int).astype(str).to_numpy().reshape(-1, 1)]
    if len(meta_cols) > 0:
        meta_matrix = []
        for c in meta_cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                meta_matrix.append(df[c].to_numpy(dtype=float))
            else:
                meta_matrix.append(df[c].astype(str).to_numpy())
        metas_parts.append(np.column_stack(meta_matrix))

    M_out = np.hstack(metas_parts) if metas_parts else None

    # Výstupní Orange tabulka pro další analýzu nebo vizualizaci.
    out_data = Orange.data.Table(domain, X_out, Y_out, M_out)

    n_anom = int((df["IF_flag"] == -1).sum())
    print(f"Detekce hotova. Nalezeno {n_anom} potenciálních anomálií.")
