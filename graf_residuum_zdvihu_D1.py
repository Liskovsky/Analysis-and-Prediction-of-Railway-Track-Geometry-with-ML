import pandas as pd
import matplotlib.pyplot as plt
import os

# Převod vstupní tabulky na pandas DataFrame
df = pd.DataFrame(in_data)

# Výběr sloupců podle pořadí v Excelu
x             = df[0]   # Pos_m
r_zdvih_G1    = df[1]
LiftRight_D1  = df[2]
r_zdvih_G2    = df[3]
LiftRight_D2  = df[4]
r_zdvih_G3    = df[5]
Rz_D1         = df[6]
Rz_D2         = df[7]

# Filtrace úseku pro přehlednější vizualizaci
xmin, xmax = 13900, 14166
mask = (x >= xmin) & (x <= xmax)

x             = x[mask]
r_zdvih_G1    = r_zdvih_G1[mask]
LiftRight_D1  = LiftRight_D1[mask]
r_zdvih_G2    = r_zdvih_G2[mask]
LiftRight_D2  = LiftRight_D2[mask]
r_zdvih_G3    = r_zdvih_G3[mask]
Rz_D1         = Rz_D1[mask]
Rz_D2         = Rz_D2[mask]

# Vykreslení průběhu residuí zdvihu
plt.figure(figsize=(14, 7))

plt.plot(x, r_zdvih_G1,   label="zdvih [mm]_G1",   color="red")
plt.plot(x, LiftRight_D1, label="LiftRight_D1",    color="blue")
plt.plot(x, r_zdvih_G2,   label="zdvih [mm]_G2",   color="green")
plt.plot(x, Rz_D1,        label="Rz_D1",           color="brown")

plt.title("Residuum zdvihu pro 1. zásah")
plt.xlabel("Poloha [m]")
plt.ylabel("Akční hodnota [mm]")
plt.grid(True, linestyle=":")
plt.legend()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Residuum_zdvihu_1_zasah.pdf", bbox_inches="tight")
plt.close()
