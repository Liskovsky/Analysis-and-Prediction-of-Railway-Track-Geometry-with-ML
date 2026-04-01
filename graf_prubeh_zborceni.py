import pandas as pd
import matplotlib.pyplot as plt
import os

# Převod vstupní tabulky na pandas DataFrame
df = pd.DataFrame(in_data)

# Výběr sloupců podle pořadí v Excelu
x        = df[0]   # Pos_m
Twist_D1 = df[1]
Twist_D2 = df[2]
RTwist   = df[3]

# Filtrace úseku
xmin, xmax = 13950, 14000
mask = (x >= xmin) & (x <= xmax)

x        = x[mask]
Twist_D1 = Twist_D1[mask]
Twist_D2 = Twist_D2[mask]
RTwist   = RTwist[mask]

# Vykreslení průběhu zborcení
plt.figure(figsize=(14, 7))

plt.plot(x, Twist_D1, label="Twist_D1", color="red")
plt.plot(x, Twist_D2, label="Twist_D2", color="blue")
plt.plot(x, RTwist,   label="RTwist",   color="green")

plt.title("Průběh zborcení koleje")
plt.xlabel("Poloha [m]")
plt.ylabel("Zborcení [mm]")
plt.grid(True, linestyle=":")
plt.legend()

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/Prubeh_zborceni_koleje.pdf", bbox_inches="tight")
plt.close()
