from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 1. Leemos los resultados de la evolucion con laser.
carpeta = Path(__file__).resolve().parent
archivo = carpeta / "resultados_laser_1.npz"

with np.load(archivo) as datos:
    tiempo = datos["tiempo"]
    aceleracion = datos["aceleracion"]
    omega = float(datos["omega"])
    intensidad = float(datos["intensidad"])
    energia_inicial = float(datos["energia_inicial"])

dt = tiempo[1] - tiempo[0]

assert tiempo.shape == aceleracion.shape
assert np.all(np.isfinite(aceleracion))
assert np.allclose(np.diff(tiempo), dt)

# 2. Ventana gaussiana centrada en el pulso.
centro = (tiempo[0] + tiempo[-1]) / 2
sigma = 5.77 / omega

ventana = np.exp(
    -0.5 * ((tiempo - centro) / sigma)**2
)

senal = aceleracion * ventana

# 3. Transformada de Fourier de la aceleracion.
# rfft conserva las frecuencias no negativas de una señal real.
transformada = (
    dt / np.sqrt(2 * np.pi)
) * np.fft.rfft(senal)

frecuencia_angular = (
    2 * np.pi * np.fft.rfftfreq(len(tiempo), d=dt)
)

orden_armonico = frecuencia_angular / omega

espectro = np.abs(transformada)**2
assert np.max(espectro) > 0.0

# Normalizacion solo para presentar esta figura.
espectro_relativo = espectro / np.max(espectro)

# 4. Prueba del eje de frecuencias con una señal conocida.
# Una oscilacion a 17*omega debe aparecer cerca de q = 17.
senal_prueba = np.cos(17 * omega * tiempo)

espectro_prueba = np.abs(
    np.fft.rfft(senal_prueba * ventana)
)**2

q_prueba = orden_armonico[np.argmax(espectro_prueba)]
separacion_q = orden_armonico[1] - orden_armonico[0]

assert abs(q_prueba - 17.0) <= separacion_q

# 5. Prediccion semiclasica del corte.
# Esta linea NO mide el corte numerico del espectro.
conversion_intensidad = 3.5094452e16
E0 = np.sqrt(intensidad / conversion_intensidad)

Up = E0**2 / (4 * omega**2)
Ip = -energia_inicial

energia_corte_predicha = Ip + 3.17 * Up
q_corte_predicho = energia_corte_predicha / omega

print("\nESPECTRO DE ARMONICOS")
print(f"Intensidad: {intensidad:.3e} W/cm²")
print(f"Separacion entre frecuencias: {separacion_q:.6f} en q")
print(f"Prueba de frecuencia: q = {q_prueba:.6f}; esperado 17")
print(f"Ip: {Ip:.6f} Hartree")
print(f"Up: {Up:.6f} Hartree")
print(f"Corte semiclasico predicho: q = {q_corte_predicho:.3f}")
print("El corte numerico aun debe analizarse en el espectro.")

# 6. Figura completa y ampliacion de algunos armonicos.
# El piso solo permite dibujar ceros en escala logaritmica.
para_dibujar = np.maximum(espectro_relativo, 1e-30)

fig, ejes = plt.subplots(2, 1, figsize=(9, 8))

ejes[0].semilogy(
    orden_armonico,
    para_dibujar,
    label="Espectro calculado"
)

ejes[0].axvline(
    q_corte_predicho,
    color="red",
    linestyle="--",
    label=f"Corte semiclasico: {q_corte_predicho:.2f}"
)

ejes[0].set_xlim(0, 100)
ejes[0].set_ylim(1e-14, 2)
ejes[0].set_xlabel("Orden armonico q = frecuencia / omega")
ejes[0].set_ylabel("Espectro relativo")
ejes[0].set_title(
    f"HHG: I = {intensidad:.1e} W/cm²\n"
    "Ventana gaussiana: sigma = 5.77 / omega"
)
ejes[0].legend()
ejes[0].grid(alpha=0.3)

ejes[1].semilogy(orden_armonico, para_dibujar)

# Guias para comparar las posiciones de los picos.
# No seleccionamos ni eliminamos componentes del espectro.
for q_impar in range(15, 36, 2):
    ejes[1].axvline(
        q_impar, color="gray", linestyle=":", alpha=0.5
    )

ejes[1].set_xlim(14, 36) 
ejes[1].set_ylim(1e-5, 1e-1)
ejes[1].set_xticks(np.arange(15, 36, 2))
ejes[1].set_xlabel("Orden armonico q")
ejes[1].set_ylabel("Espectro relativo")
ejes[1].set_title("Ampliacion; guias en ordenes impares")
ejes[1].grid(alpha=0.3)

fig.tight_layout()

ruta_figura = carpeta / "espectro_armonicos_1.png"
fig.savefig(ruta_figura, dpi=200)

# Conservamos el espectro sin normalizar para comparaciones posteriores.
np.savez_compressed(
    carpeta / "espectro_armonicos_1.npz",
    orden=orden_armonico,
    espectro=espectro,
    intensidad=intensidad,
    q_corte_predicho=q_corte_predicho,
    sigma=sigma,
)

print("Figura guardada en espectro_armonicos_1.png")
print("Datos guardados en espectro_armonicos_1.npz")

plt.show()

# COMPARACION DEL PROCESAMIENTO DEL MISMO REGISTRO.

ventanas_comparacion = {
    "Gaussiana": ventana,
    "Hann": np.hanning(len(tiempo)),
    "Sin ventana adicional": np.ones(len(tiempo)),
}

fig_comparacion, ejes_comparacion = plt.subplots(
    3, 1, figsize=(10, 9), sharex=True
)

print("\nCOMPARACION DE VENTANAS")
print("Diagnostico en intervalos cercanos a ordenes 15 a 35.")
print("No es una medida de toda la emision par o impar.")

for eje, (nombre, pesos) in zip(
    ejes_comparacion, ventanas_comparacion.items()
):
    amplitud = (
        dt / np.sqrt(2 * np.pi)
    ) * np.fft.rfft(aceleracion * pesos)

    potencia = np.abs(amplitud)**2

    # Medimos por igual alrededor de cada orden entero.
    # El ancho 0.25 es una eleccion para este diagnostico.
    valores_pares = []
    valores_impares = []

    for orden in range(15, 36):
        seleccion = np.abs(orden_armonico - orden) < 0.25
        valor = np.sum(potencia[seleccion]) * separacion_q

        if orden % 2 == 0:
            valores_pares.append(valor)
        else:
            valores_impares.append(valor)

    # Promedios porque hay 10 ordenes pares y 11 impares.
    relacion = (
        np.mean(valores_pares) / np.mean(valores_impares)
    )

    print(
        f"{nombre}: promedio cerca de pares / "
        f"promedio cerca de impares = {relacion:.3f}"
    )

    # Normalizacion independiente para comparar las formas,
    # no las intensidades absolutas entre ventanas.
    region = (orden_armonico >= 14) & (orden_armonico <= 36)
    escala = np.max(potencia[region])

    eje.semilogy(
        orden_armonico,
        np.maximum(potencia / escala, 1e-30)
    )

    for orden in range(15, 36, 2):
        eje.axvline(
            orden, color="gray", linestyle=":", alpha=0.5
        )

    eje.set_title(nombre)
    eje.set_ylabel("Espectro / maximo regional")
    eje.set_ylim(1e-4, 2)
    eje.grid(alpha=0.3)

ejes_comparacion[-1].set_xlim(14, 36)
ejes_comparacion[-1].set_xticks(np.arange(15, 36, 2))
ejes_comparacion[-1].set_xlabel("Orden armonico q")

fig_comparacion.tight_layout()
fig_comparacion.savefig(
    carpeta / "comparacion_ventanas.png",
    dpi=200
)

plt.show()

# COMPARACION DEL ESPECTRO AL REDUCIR EL PASO TEMPORAL.
# Usamos la misma ventana gaussiana en ambos calculos.

with np.load(
    carpeta / "resultados_laser_1_dt001.npz"
) as datos:
    tiempo_fino = datos["tiempo"]
    aceleracion_fina = datos["aceleracion"]
    omega_fino = float(datos["omega"])
    intensidad_fina = float(datos["intensidad"])

# Comprobamos que estamos comparando el mismo laser.
assert np.isclose(omega_fino, omega)
assert np.isclose(intensidad_fina, intensidad)
assert np.isclose(tiempo_fino[-1], tiempo[-1])

dt_fino = tiempo_fino[1] - tiempo_fino[0]

centro_fino = (tiempo_fino[0] + tiempo_fino[-1]) / 2

ventana_fina = np.exp(
    -0.5 * ((tiempo_fino - centro_fino) / sigma)**2
)

transformada_fina = (
    dt_fino / np.sqrt(2 * np.pi)
) * np.fft.rfft(aceleracion_fina * ventana_fina)

q_fino = (
    2 * np.pi
    * np.fft.rfftfreq(len(tiempo_fino), d=dt_fino)
    / omega_fino
)

espectro_fino = np.abs(transformada_fina)**2

# Las frecuencias de ambos registros difieren ligeramente.
# Interpolamos el espectro fino sobre el eje del original.
region_comparacion = (
    (orden_armonico >= 15) & (orden_armonico <= 80)
)

referencia_fina = np.interp(
    orden_armonico[region_comparacion],
    q_fino,
    espectro_fino
)

# Diferencia integrada relativa en la region elegida.
# Comparamos espectros sin normalizarlos por separado.
diferencia_relativa = (
    np.sum(
        np.abs(
            espectro[region_comparacion] - referencia_fina
        )
    )
    / np.sum(referencia_fina)
)

print("\nREFINAMIENTO TEMPORAL DEL ESPECTRO")
print(f"Paso original: {dt:.8f} u.a.")
print(f"Paso refinado: {dt_fino:.8f} u.a.")
print("Comparacion entre ordenes 15 y 80.")
print(
    f"Diferencia integrada relativa: "
    f"{100 * diferencia_relativa:.4f} %"
)

# Misma escala de normalizacion para las dos curvas.
escala_comun = np.max(espectro)

fig_refinamiento, eje_refinamiento = plt.subplots(
    figsize=(9, 5)
)

eje_refinamiento.semilogy(
    orden_armonico,
    np.maximum(espectro / escala_comun, 1e-30),
    label=f"dt = {dt:.3f}"
)

eje_refinamiento.semilogy(
    q_fino,
    np.maximum(espectro_fino / escala_comun, 1e-30),
    "--",
    label=f"dt = {dt_fino:.3f}"
)

eje_refinamiento.set_xlim(15, 85)
eje_refinamiento.set_ylim(1e-12, 1e-1)
eje_refinamiento.set_xlabel("Orden armonico q")
eje_refinamiento.set_ylabel("Espectro / maximo del caso original")
eje_refinamiento.set_title(
    "Refinamiento temporal: misma malla, laser y ventana"
)
eje_refinamiento.legend()
eje_refinamiento.grid(alpha=0.3)

fig_refinamiento.tight_layout()
fig_refinamiento.savefig(
    carpeta / "refinamiento_temporal_espectro.png",
    dpi=200
)

plt.show()