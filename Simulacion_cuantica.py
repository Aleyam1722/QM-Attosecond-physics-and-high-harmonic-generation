import numpy as np
import matplotlib.pyplot as plt

# Usamos unidades atomicas: hbar = masa del electron = carga elemental = 1.

# 1. Malla espacial: intervalo [-L, L), sin repetir el extremo.
N = 2048
L = 100.0
dx = 2 * L / N

x = (np.arange(N) - N // 2) * dx

# Numeros de onda en el orden que utiliza la FFT.
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# 2. Potencial atomico suavizado del modelo 1D.
V = -1.0 / np.sqrt(x**2 + 2.0)


def normalizar(psi):
    """Ajusta la integral de |psi|^2 a uno."""
    norma = np.sum(np.abs(psi)**2) * dx
    return psi / np.sqrt(norma)


def aplicar_hamiltoniano(psi):
    """Calcula H psi: energia cinetica mas potencial."""
    parte_cinetica = np.fft.ifft(
        0.5 * k**2 * np.fft.fft(psi)
    )
    return parte_cinetica + V * psi


# 3. Funcion inicial de prueba.
# La gaussiana NO es el resultado: el algoritmo la modifica.
psi = np.exp(-x**2 / 2.0).astype(complex)
psi = normalizar(psi)

# 4. Parametros del calculo en tiempo imaginario.
dtau = 0.02
max_pasos = 20000
energia_anterior = None
convergio = False

# Separacion simetrica:
# medio paso potencial -> paso cinetico -> medio paso potencial.
factor_V = np.exp(-V * dtau / 2)
factor_T = np.exp(-0.5 * k**2 * dtau)

for paso in range(1, max_pasos + 1):

    # Medio paso del potencial, en el espacio de posiciones.
    psi = factor_V * psi

    # Paso cinetico, en el espacio de numeros de onda.
    psi_k = np.fft.fft(psi)
    psi_k = factor_T * psi_k
    psi = np.fft.ifft(psi_k)

    # Segundo medio paso del potencial.
    psi = factor_V * psi

    # El tiempo imaginario no conserva la norma.
    psi = normalizar(psi)

    # Revisamos cada 100 pasos si nos acercamos a un autoestado.
    if paso % 100 == 0:
        Hpsi = aplicar_hamiltoniano(psi)

        energia = np.real(np.vdot(psi, Hpsi)) * dx

        # Para un autoestado exacto: H psi - E psi = 0.
        residuo = np.sqrt(
            np.sum(np.abs(Hpsi - energia * psi)**2) * dx
        )

        if energia_anterior is not None:
            cambio_energia = abs(energia - energia_anterior)

            if cambio_energia < 1e-10 and residuo < 1e-5:
                convergio = True
                break

        energia_anterior = energia

if not convergio:
    raise RuntimeError(
        "No se alcanzaron los criterios de convergencia."
    )

# 5. Resultados del estado calculado.
norma = np.sum(np.abs(psi)**2) * dx

# V tiende a cero lejos del ion: el umbral del continuo es E = 0.
# Por eso la energia necesaria para ionizar es Ip = 0 - E_base.
Ip = -energia

print("\nESTADO FUNDAMENTAL DEL MODELO 1D")
print(f"Pasos de tiempo imaginario: {paso}")
print(f"Energia: {energia:.10f} Hartree")
print(f"Energia de ionizacion Ip: {Ip:.10f} Hartree")
print(f"Norma: {norma:.12f}")
print(f"Residuo ||H psi - E psi||: {residuo:.3e} Hartree")

assert np.isclose(norma, 1.0, atol=1e-12, rtol=0)
assert energia < 0.0

# 6. Densidad de probabilidad del estado obtenido.
plt.figure()
plt.plot(x, np.abs(psi)**2)
plt.xlim(-10, 10)
plt.xlabel("Posicion x (bohr)")
plt.ylabel("Densidad |psi(x)|² (bohr⁻¹)")
plt.title(
    f"Estado fundamental 1D: E = {energia:.6f} Hartree\n"
    f"N = {N}, L = {L:g} bohr, dtau = {dtau:g} u.a."
)
plt.grid(alpha=0.3)
plt.tight_layout()


# PRUEBA: evolucion en tiempo real sin campo laser.
# Conservamos psi como estado inicial para los siguientes bloques.
psi_prueba = psi.copy()

dt = 0.02
pasos_prueba = 5000

norma_inicial = np.sum(np.abs(psi_prueba)**2) * dx

energia_inicial = np.real(
    np.vdot(psi_prueba, aplicar_hamiltoniano(psi_prueba))
) * dx

# En tiempo REAL, los exponentes incluyen la unidad imaginaria.
factor_V_real = np.exp(-1j * V * dt / 2)
factor_T_real = np.exp(-1j * (k**2 / 2) * dt)

max_error_norma = 0.0
max_cambio_energia = 0.0

for paso_real in range(1, pasos_prueba + 1):
    # Medio paso de potencial.
    psi_prueba = factor_V_real * psi_prueba

    # Paso cinetico.
    psi_prueba = np.fft.ifft(
        factor_T_real * np.fft.fft(psi_prueba)
    )

    # Segundo medio paso de potencial.
    psi_prueba = factor_V_real * psi_prueba

    # No normalizamos: queremos comprobar la conservacion.
    norma_actual = np.sum(np.abs(psi_prueba)**2) * dx

    energia_actual = np.real(
        np.vdot(
            psi_prueba,
            aplicar_hamiltoniano(psi_prueba)
        )
    ) * dx

    max_error_norma = max(
        max_error_norma,
        abs(norma_actual - norma_inicial)
    )

    max_cambio_energia = max(
        max_cambio_energia,
        abs(energia_actual - energia_inicial)
    )

print("\nPRUEBA DE EVOLUCION SIN CAMPO")
print(f"Tiempo total: {pasos_prueba * dt:.2f} u.a.")
print(f"Norma inicial: {norma_inicial:.12f}")
print(f"Norma final: {norma_actual:.12f}")
print(f"Mayor cambio de norma: {max_error_norma:.3e}")
print(
    f"Mayor cambio de energia: "
    f"{max_cambio_energia:.3e} Hartree"
)

# Tolerancias numericas elegidas para esta prueba.
assert max_error_norma < 1e-10, "La norma cambio demasiado."
assert max_cambio_energia < 1e-7, "La energia cambio demasiado."

print("Prueba superada: norma y energia estables sin campo.")

# PRUEBA DE REFINAMIENTO DEL ESTADO FUNDAMENTAL.
# Usamos variables locales para conservar el estado psi anterior.

def energia_refinada(numero_puntos, paso_tau):
    longitud = 100.0
    separacion = 2 * longitud / numero_puntos

    posiciones = (
        np.arange(numero_puntos) - numero_puntos // 2
    ) * separacion

    ondas = 2 * np.pi * np.fft.fftfreq(
        numero_puntos, d=separacion
    )

    potencial = -1 / np.sqrt(posiciones**2 + 2.0)

    estado = np.exp(-posiciones**2 / 2).astype(complex)
    estado /= np.sqrt(
        np.sum(np.abs(estado)**2) * separacion
    )

    medio_potencial = np.exp(-potencial * paso_tau / 2)
    paso_cinetico = np.exp(-ondas**2 * paso_tau / 2)

    # Mismo intervalo de tiempo imaginario en todos los casos.
    tau_total = 40.0
    numero_pasos = round(tau_total / paso_tau)

    for _ in range(numero_pasos):
        estado = medio_potencial * estado
        estado = np.fft.ifft(
            paso_cinetico * np.fft.fft(estado)
        )
        estado = medio_potencial * estado

        estado /= np.sqrt(
            np.sum(np.abs(estado)**2) * separacion
        )

    Hestado = (
        np.fft.ifft(
            (ondas**2 / 2) * np.fft.fft(estado)
        )
        + potencial * estado
    )

    valor_energia = np.real(
        np.vdot(estado, Hestado)
    ) * separacion

    residuo_local = np.sqrt(
        np.sum(
            np.abs(Hestado - valor_energia * estado)**2
        ) * separacion
    )

    return valor_energia, residuo_local


print("\nREFINAMIENTO DEL ESTADO FUNDAMENTAL")

casos = [
    (2048, 0.02),
    (2048, 0.01),
    (4096, 0.01),
]

energias_refinadas = []

for numero_puntos, paso_tau in casos:
    energia_caso, residuo_caso = energia_refinada(
        numero_puntos, paso_tau
    )

    energias_refinadas.append(energia_caso)

    print(
        f"N = {numero_puntos}, dtau = {paso_tau:.3f}: "
        f"E = {energia_caso:.12f} Hartree; "
        f"residuo = {residuo_caso:.3e} Hartree"
    )

cambio_temporal = abs(
    energias_refinadas[1] - energias_refinadas[0]
)

cambio_espacial = abs(
    energias_refinadas[2] - energias_refinadas[1]
)

print(f"Cambio al reducir dtau: {cambio_temporal:.3e} Hartree")
print(f"Cambio al aumentar N: {cambio_espacial:.3e} Hartree")

# Tolerancia elegida para la estabilidad de la energia.
assert cambio_temporal < 1e-8
assert cambio_espacial < 1e-8

print("Prueba superada: energia estable frente al refinamiento.")

# EVOLUCION CUANTICA CON EL PULSO LASER.

from pathlib import Path
from Campo_laser import campo_laser, omega, periodo, duracion


def pedir_intensidad(caso, valor_predeterminado):
    while True:
        entrada = input(
            f"Intensidad del caso {caso} en W/cm² "
            f"[Enter = {valor_predeterminado:.1e}]: "
        ).strip()

        if not entrada:
            return valor_predeterminado

        try:
            intensidad = float(entrada.replace(",", "."))
        except ValueError:
            print("Escribe un número como 1.2e14.")
            continue

        if np.isfinite(intensidad) and intensidad > 0:
            return intensidad

        print("La intensidad debe ser positiva y finita.")


def correr_caso(archivo, intensidad, N_laser, paso):
    L_laser = 400.0
    dx_laser = 2 * L_laser / N_laser
    x_laser = (np.arange(N_laser) - N_laser // 2) * dx_laser
    k_laser = 2 * np.pi * np.fft.fftfreq(N_laser, d=dx_laser)

    V_laser = -1 / np.sqrt(x_laser**2 + 2.0)
    dV_laser = x_laser / (x_laser**2 + 2.0)**1.5

    # Cada corrida empieza desde el mismo estado fundamental.
    psi_laser = (
        np.interp(x_laser, x, psi.real, left=0.0, right=0.0)
        + 1j * np.interp(x_laser, x, psi.imag, left=0.0, right=0.0)
    )
    psi_laser /= np.sqrt(np.sum(np.abs(psi_laser)**2) * dx_laser)

    Hpsi_laser = (
        np.fft.ifft((k_laser**2 / 2) * np.fft.fft(psi_laser))
        + V_laser * psi_laser
    )
    energia_inicial = np.vdot(psi_laser, Hpsi_laser).real * dx_laser
    residuo = np.sqrt(
        np.sum(np.abs(Hpsi_laser - energia_inicial * psi_laser)**2)
        * dx_laser
    )
    assert residuo < 1e-4

    # Potencial absorbente en los bordes de la caja.
    inicio_absorcion = 320.0
    eta = 1.0
    distancia_relativa = np.clip(
        (np.abs(x_laser) - inicio_absorcion)
        / (L_laser - inicio_absorcion),
        0.0,
        1.0,
    )
    W = eta * distancia_relativa**4

    # Ajustar el paso para terminar exactamente al final del pulso.
    pasos_laser = int(np.ceil(duracion / paso))
    dt_laser = duracion / pasos_laser
    t_laser = np.arange(pasos_laser + 1) * dt_laser
    factor_cinetico = np.exp(-1j * (k_laser**2 / 2) * dt_laser)

    campos = campo_laser(t_laser, intensidad=intensidad)
    campos_medios = campo_laser(
        (np.arange(pasos_laser) + 0.5) * dt_laser,
        intensidad=intensidad,
    )

    normas = np.empty(pasos_laser + 1)
    aceleraciones = np.empty(pasos_laser + 1)

    densidad = np.abs(psi_laser)**2
    normas[0] = np.sum(densidad) * dx_laser
    aceleraciones[0] = (
        -np.sum(dV_laser * densidad) * dx_laser - campos[0] * normas[0]
    )

    indices_fotos = set(
        np.linspace(
            int(0.45 * pasos_laser),
            int(0.65 * pasos_laser),
            401,
            dtype=int,
        )
    )
    tiempos_fotos = []
    densidades_fotos = []

    print(f"\nCalculando {archivo.name} con I = {intensidad:.3e} W/cm²")
    print(f"Pasos: {pasos_laser}", flush=True)

    for j, E_medio in enumerate(campos_medios, start=1):
        potencial_total = V_laser + x_laser * E_medio
        factor_medio = np.exp((-1j * potencial_total - W) * dt_laser / 2)

        psi_laser = factor_medio * psi_laser
        psi_laser = np.fft.ifft(factor_cinetico * np.fft.fft(psi_laser))
        psi_laser = factor_medio * psi_laser

        densidad = np.abs(psi_laser)**2
        normas[j] = np.sum(densidad) * dx_laser
        aceleraciones[j] = (
            -np.sum(dV_laser * densidad) * dx_laser - campos[j] * normas[j]
        )

        if j in indices_fotos:
            tiempos_fotos.append(t_laser[j])
            densidades_fotos.append(densidad.copy())

        if j % 10000 == 0:
            print(f"Avance: {100 * j / pasos_laser:.1f} %", flush=True)

    assert np.all(np.isfinite(aceleraciones))
    assert np.all(np.isfinite(normas))
    assert np.max(normas) <= normas[0] + 1e-9
    assert np.max(np.diff(normas)) < 1e-9

    # Guardar primero en un temporal permite conservar el archivo anterior
    # si se interrumpe la escritura.
    temporal = archivo.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporal,
        tiempo=t_laser,
        aceleracion=aceleraciones,
        norma=normas,
        x=x_laser,
        tiempos_fotos=np.array(tiempos_fotos),
        densidades_fotos=np.array(densidades_fotos),
        omega=omega,
        periodo=periodo,
        intensidad=intensidad,
        energia_inicial=energia_inicial,
        dt=dt_laser,
        dx=dx_laser,
        inicio_absorcion=inicio_absorcion,
        eta=eta,
    )
    temporal.replace(archivo)

    print(f"Probabilidad absorbida: {normas[0] - normas[-1]:.6f}")
    print(f"Resultados guardados en {archivo.name}")

    plt.figure()
    plt.plot(t_laser / periodo, normas)
    plt.xlabel("Tiempo / periodo del laser")
    plt.ylabel("Probabilidad dentro de la caja")
    plt.title(
        f"Evolucion con laser: I = {intensidad:.3e} W/cm²\n"
        f"N = {N_laser}, L = {L_laser:g} bohr"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()


def generar_resultados():
    intensidad_1 = pedir_intensidad(1, 1.0e14)
    intensidad_2 = pedir_intensidad(2, 1.1e14)

    while intensidad_2 == intensidad_1:
        print("Elige una intensidad distinta para el caso 2.")
        intensidad_2 = pedir_intensidad(2, 1.1e14)

    carpeta = Path(__file__).resolve().parent
    corridas = [
        ("resultados_laser_1.npz", intensidad_1, 4096, 0.02),
        ("resultados_laser_1_dt001.npz", intensidad_1, 4096, 0.01),
        ("resultados_laser_1_dt001_N8192.npz", intensidad_1, 8192, 0.01),
        ("resultados_laser_2_dt001_N8192.npz", intensidad_2, 8192, 0.01),
    ]

    for nombre, intensidad, puntos, paso in corridas:
        archivo = carpeta / nombre

        if archivo.is_file():
            with np.load(archivo) as datos:
                misma_intensidad = float(datos["intensidad"]) == intensidad

            if misma_intensidad:
                print(f"Usando {nombre}")
                continue

        correr_caso(archivo, intensidad, puntos, paso)


if __name__ == "__main__":
    generar_resultados()
    plt.show()