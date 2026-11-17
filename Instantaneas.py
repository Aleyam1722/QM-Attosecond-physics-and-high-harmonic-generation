from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

carpeta = Path(__file__).resolve().parent

# Leemos la simulacion que ya realizamos.
with np.load(
    carpeta / "resultados_laser_1_dt001_N8192.npz"
) as datos:
    x = datos["x"]
    tiempos = datos["tiempos_fotos"]
    densidades = datos["densidades_fotos"]
    periodo = float(datos["periodo"])

# Mostramos seis momentos de las instantaneas guardadas.
indices = np.linspace(
    0, len(tiempos) - 1, 6, dtype=int
)

fig, ejes = plt.subplots(
    2, 3, figsize=(12, 6), sharex=True, sharey=True
)

for eje, indice in zip(ejes.flat, indices):
    eje.semilogy(
        x,
        np.maximum(densidades[indice], 1e-16)
    )

    eje.set_title(
        f"t / T = {tiempos[indice] / periodo:.3f}"
    )
    eje.set_xlim(-100, 100)
    eje.set_ylim(1e-10, 1)
    eje.grid(alpha=0.3)

fig.supxlabel("Posicion x (bohr)")
fig.supylabel("Densidad |psi(x,t)|² (bohr⁻¹)")
fig.suptitle("Instantaneas de la densidad electronica")
fig.tight_layout()

fig.savefig(
    carpeta / "instantaneas_densidad.png",
    dpi=200
)

plt.show()