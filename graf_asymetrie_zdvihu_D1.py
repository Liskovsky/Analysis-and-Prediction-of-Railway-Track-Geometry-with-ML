import pandas as pd
import matplotlib.pyplot as plt
import os

# Převod vstupní tabulky na pandas DataFrame
df = pd.DataFrame(in_data)

# Výběr sloupců podle pořadí v Excelu
x            = df[0]   # Pos_m
LiftLeft_D1  = df[1]
LiftRight_D1 = df[2]
ZAsym        = df[3]   # Asymetrie = Left - Right

# Filtrace úseku
xmin, xmax = 13880, 14000
mask = (x >= xmin) & (x <= xmax)

x            = x[mask]
LiftLeft_D1  = LiftLeft_D1[mask]
LiftRight_D1 = LiftRight_D1[mask]
ZAsym        = ZAsym[mask]

# Vykreslení asymetrie zdvihu
plt.figure(figsize=(14, 7))

plt.plot(x, LiftLeft_D1,  label="LiftLeft_D1",  color="red")
plt.plot(x, LiftRight_D1, label="LiftRight_D1", color="blue")
plt.plot(x, ZAsym,        label="ZAsym = Left - Right", color="green")

plt.title("Asymetrie zdvihu D1")
plt.xlabel("Poloha [m]")
plt.ylabel("Zdvih [mm]")
plt.grid(True, linestyle=":")
plt.legend()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Asymetrie_zdvihu_D1.pdf", bbox_inches="tight")
plt.close()
