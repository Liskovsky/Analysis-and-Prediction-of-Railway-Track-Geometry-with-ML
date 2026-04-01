import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Převod vstupní Orange tabulky na pandas DataFrame,
# abychom mohli pohodlně pracovat s hodnotami G4 a predikcemi modelů.
df = pd.DataFrame(in_data.X, columns=[v.name for v in in_data.domain.attributes])

# Názvy modelů, jejichž predikce chceme porovnat.
# Musí odpovídat názvům sloupců z widgetu Predictions.
modely = ['Random Forest', 'Gradient Boosting']

# Dva podgrafy: horní pro reálná měření G4, spodní pro verdikty modelů.
fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(15, 10),
    gridspec_kw={'height_ratios': [2, 1]}
)

# ============================================
# 1) HORNÍ GRAF – REÁLNÁ GEOMETRIE G4
# ============================================

x = df['stan [km]_G4']

# Zdvih a posun z měření G4
ax1.plot(x, df['zdvih [mm]_G4'], label='Zdvih G4', color='#0000FF', linewidth=1.5)
ax1.plot(x, df['posun [mm]_G4'], label='Posun G4', color='#FF0000', linewidth=1.5)

# Toleranční pásma pro vizuální orientaci
ax1.axhline(5.0,  color='blue', linestyle='--', alpha=0.3, label='Tolerance Zdvih ±5mm')
ax1.axhline(-5.0, color='blue', linestyle='--', alpha=0.3)
ax1.axhline(10.0,  color='red', linestyle='-.', alpha=0.3, label='Tolerance Posun ±10mm')
ax1.axhline(-10.0, color='red', linestyle='-.', alpha=0.3)

ax1.set_ylabel('Výchylka [mm]')
ax1.set_title('Srovnání: Realita G4 vs. Predikce Modelů')
ax1.legend(loc='upper left')
ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# ============================================
# 2) SPODNÍ GRAF – VERDIKTY MODELOVÝCH PREDIKCÍ
# ============================================

for i, m_name in enumerate(modely):
    try:
        # Predikce modelu (0 = CHYBA, 1 = OK)
        y_vals = pd.to_numeric(in_data.get_column(m_name), errors="coerce").fillna(1).astype(int)

        # Vizualizační převod:
        # CHYBA (0) → 1.0  (vyskočí nahoru)
        # OK    (1) → 0.0
        y_bin = np.where(y_vals == 0, 1.0, 0.0)

        # Vertikální posun, aby se jednotlivé modely nepřekrývaly
        offset = i * 1.5
        ax2.step(x, y_bin + offset, where='post', label=m_name, linewidth=2.5)

        # Popisek modelu přímo do grafu
        ax2.text(x.iloc[0], offset + 0.6, f' {m_name}',
                 fontweight='bold', fontsize=10)

    except Exception:
        print(f"Model '{m_name}' nenalezen. Zkontroluj propojení widgetu Predictions.")

# Formátování spodního grafu
ax2.set_yticks([0, 1, 1.5, 2.5])
ax2.set_yticklabels(['OK', 'CHYBA', 'OK', 'CHYBA'])
ax2.set_ylabel('Verdikt modelu')
ax2.set_xlabel('Kilometráž [km]')
ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
