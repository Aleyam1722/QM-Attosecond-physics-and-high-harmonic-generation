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
plt.show()

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