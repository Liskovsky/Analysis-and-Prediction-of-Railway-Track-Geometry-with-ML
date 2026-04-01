import pandas as pd
import matplotlib.pyplot as plt
import os

# Převod vstupní tabulky na pandas DataFrame
df = pd.DataFrame(in_data)

# Výběr sloupců podle pořadí v Excelu
x          = df[0]   # Pos_m
r_posun_G1 = df[1]
r_posun_G2 = df[2]
r_posun_G3 = df[3]
Rs_D1      = df[5]
Rs_D2      = df[7]

# Filtrace úseku
xmin, xmax = 13900, 14166
mask = (x >= xmin) & (x <= xmax)

x          = x[mask]
r_posun_G1 = r_posun_G1[mask]
r_posun_G2 = r_posun_G2[mask]
r_posun_G3 = r_posun_G3[mask]
Rs_D1      = Rs_D1[mask]
Rs_D2      = Rs_D2[mask]

# Vykreslení průběhu residuí posunu
plt.figure(figsize=(14, 7))

plt.plot(x, r_posun_G1, label="r_posun_G1", color="red")
plt.plot(x, r_posun_G2, label="r_posun_G2", color="green")
plt.plot(x, r_posun_G3, label="r_posun_G3", color="purple")
plt.plot(x, Rs_D1,      label="Rs_D1",      color="blue")
plt.plot(x, Rs_D2,      label="Rs_D2",      color="brown")

plt.title("Residuum posunu pro 1. zásah")
plt.xlabel("Poloha [m]")
plt.ylabel("Akční hodnota [mm]")
plt.grid(True, linestyle=":")
plt.legend()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Residuum_posunu_1_zasah.pdf", bbox_inches="tight")
plt.close()
