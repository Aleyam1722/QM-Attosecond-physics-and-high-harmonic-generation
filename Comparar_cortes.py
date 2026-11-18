from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

carpeta = Path(__file__).resolve().parent
fig, eje = plt.subplots(figsize=(9, 5))

for caso, color in [(1, "tab:blue"), (2, "tab:orange")]:
    archivo = carpeta / f"resultados_laser_{caso}_dt001_N8192.npz"

    with np.load(archivo) as datos:
        t = datos["tiempo"]
        a = datos["aceleracion"]
        omega = float(datos["omega"])
        intensidad = float(datos["intensidad"])
        Ip = -float(datos["energia_inicial"])

    # Calculamos el espectro a partir de la aceleracion del electron.
    dt = t[1] - t[0]
    centro = (t[0] + t[-1]) / 2
    sigma = 5.77 / omega
    ventana = np.exp(-0.5 * ((t - centro) / sigma)**2)

    amplitud = dt * np.fft.rfft(a * ventana) / np.sqrt(2 * np.pi)
    espectro = np.abs(amplitud)**2
    q = 2 * np.pi * np.fft.rfftfreq(len(t), d=dt) / omega

    # Prediccion semiclasica del orden armonico de corte, en unidades atomicas.
    E0 = np.sqrt(intensidad / 3.5094452e16)
    Up = E0**2 / (4 * omega**2)
    q_predicho = (Ip + 3.17 * Up) / omega
        # Tomamos el pico mas alto en cada intervalo de dos ordenes.
    centros = np.arange(41, 90, 2)
    picos = []

    for centro_q in centros:
        intervalo = (q >= centro_q - 1) & (q < centro_q + 1)
        picos.append(espectro[intervalo].max())

    log_picos = np.log10(np.maximum(picos, 1e-300))

    # Buscamos donde la meseta cambia a una caida sostenida.
    mejor_error = np.inf
    q_numerico = None

    for candidato in centros[4:-4]:
        ajuste = np.column_stack((
            np.ones_like(centros),
            centros - candidato,
            np.maximum(centros - candidato, 0)
        ))
        coef, *_ = np.linalg.lstsq(ajuste, log_picos, rcond=None)
        pendiente_antes = coef[1]
        pendiente_despues = coef[1] + coef[2]

        if pendiente_despues < min(0, pendiente_antes):
            error = np.sum((log_picos - ajuste @ coef)**2)
            if error < mejor_error:
                mejor_error = error
                q_numerico = candidato

    if q_numerico is None:
        raise ValueError("No se identifico una caida sostenida.")

    print(f"Caso {caso}: corte estimado del espectro = {q_numerico}")
    print(f"Diferencia con la prediccion = {q_numerico - q_predicho:+.2f} ordenes")

    eje.axvline(q_numerico, color=color, linestyle=":",
                label=f"Corte estimado: {q_numerico}")

    eje.semilogy(
        q, espectro / espectro.max(), color=color,
        label=f"I = {intensidad:.2e} W/cm²"
    )
    eje.axvline(q_predicho, color=color, linestyle="--",
                label=f"Corte predicho: {q_predicho:.1f}")
    print(f"Caso {caso}: I = {intensidad:.2e} W/cm², "
          f"corte predicho = {q_predicho:.2f}")

eje.set_xlim(40, 90)
eje.set_ylim(1e-12, 1)
eje.set_xlabel("Orden armónico q")
eje.set_ylabel("Espectro relativo al máximo de cada corrida")
eje.set_title("Corte armónico para dos intensidades")
eje.legend()
eje.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(carpeta / "comparacion_cortes.png", dpi=200)
plt.show()