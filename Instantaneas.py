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
# Tiempos en periodos del laser, desde la salida hasta el regreso clasico previsto.
tiempos_objetivo = [5.05, 5.20, 5.35, 5.48, 5.60, 5.71]
tiempos_en_ciclos = tiempos / periodo

# Para cada tiempo elegido, usamos la foto guardada mas cercana.
indices = [
    np.argmin(np.abs(tiempos_en_ciclos - objetivo))
    for objetivo in tiempos_objetivo
]

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

from matplotlib.colors import LogNorm

fig_mapa, eje_mapa = plt.subplots(figsize=(9, 5))

# Mostramos la misma region espacial que en las seis instantaneas.
region = np.abs(x) <= 100

mapa = eje_mapa.pcolormesh(
    x[region],
    tiempos / periodo,
    np.maximum(densidades[:, region], 1e-16),
    shading="auto",
    norm=LogNorm(vmin=1e-8, vmax=1),
    cmap="viridis"
)

# Posicion del ion.
eje_mapa.axvline(
    0, color="white", linestyle="--", linewidth=1
)

eje_mapa.set_xlabel("Posicion x (bohr)")
eje_mapa.set_ylabel("Tiempo / periodo del laser")
eje_mapa.set_title("Evolucion de la densidad electronica")

barra = fig_mapa.colorbar(mapa, ax=eje_mapa)
barra.set_label("Densidad |psi(x,t)|² (bohr⁻¹)")

fig_mapa.tight_layout()
fig_mapa.savefig(
    carpeta / "mapa_densidad.png",
    dpi=200
)

plt.show()