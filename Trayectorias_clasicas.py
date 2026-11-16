import numpy as np
import matplotlib.pyplot as plt

# Momento del ciclo del láser en que sale el electrón.
# 0.30 rad es un ejemplo de prueba, no un dato experimental.
fase_salida = 0.30

# Seguimos el movimiento durante un ciclo después de la salida.
# Los 2000 puntos se usan para dibujar la trayectoria.
fase = np.linspace(
    fase_salida,
    fase_salida + 2 * np.pi,
    2000
)

# Posición del modelo clásico.
# Está expresada en unidades de x_q = e*E0/(m*omega**2).
posicion = (
    np.cos(fase)
    - np.cos(fase_salida)
    + (fase - fase_salida) * np.sin(fase_salida)
)

# Tiempo desde la salida, expresado en ciclos del láser.
tiempo = (fase - fase_salida) / (2 * np.pi)

# Primera comprobación: el electrón comienza en posición cero.
assert np.isclose(posicion[0], 0.0)
print("Comprobación inicial correcta: x = 0 al salir.")

# Buscamos un cruce desde posiciones negativas hacia cero.
# Esto excluye el cero inicial, que corresponde a la salida.
cruces = np.flatnonzero(
    (posicion[:-1] < 0) & (posicion[1:] >= 0)
)

if len(cruces) == 0:
    raise RuntimeError(
        "No se encontro un regreso en el intervalo estudiado."
    )

# Elegimos el primer cruce posterior a la salida.
indice = cruces[0]

# La raiz esta entre estas dos fases.
izquierda = fase[indice]
derecha = fase[indice + 1]


def posicion_en_fase(phi):
    return (
        np.cos(phi)
        - np.cos(fase_salida)
        + (phi - fase_salida) * np.sin(fase_salida)
    )


# Biseccion: reducimos repetidamente el intervalo del cruce.
for _ in range(40):
    centro = (izquierda + derecha) / 2

    if posicion_en_fase(centro) < 0:
        izquierda = centro
    else:
        derecha = centro

fase_regreso = (izquierda + derecha) / 2

# Tiempo transcurrido desde la salida, en ciclos del laser.
tiempo_regreso = (
    (fase_regreso - fase_salida) / (2 * np.pi)
)

# Velocidad expresada en unidades de e*E0/(m*omega).
velocidad_regreso = (
    np.sin(fase_salida) - np.sin(fase_regreso)
)

# Energia cinetica de regreso expresada en unidades de Up.
energia_regreso = 2 * velocidad_regreso**2

# Comprobamos que el regreso sea posterior a la salida
# y que la posicion calculada sea aproximadamente cero.
assert fase_regreso > fase_salida
assert abs(posicion_en_fase(fase_regreso)) < 1e-10

print(f"Fase de regreso: {fase_regreso:.6f} rad")
print(f"Tiempo hasta regresar: {tiempo_regreso:.6f} ciclos")
print(f"Energia de regreso: {energia_regreso:.6f} Up")
print("Regreso verificado: posicion aproximadamente cero.")

# Dibujamos la trayectoria.
plt.plot(tiempo, posicion, label="Electrón")
plt.axhline(
    0,
    color="black",
    linestyle="--",
    label="Posición del ion"
)

plt.xlabel("Tiempo desde la salida / período del láser")
plt.ylabel("Posición x / x_q")
plt.title("Trayectoria clásica: fase de salida = 0.30 rad")
plt.scatter(
    tiempo_regreso,
    0,
    color="red",
    zorder=3,
    label=f"Regreso: {tiempo_regreso:.3f} ciclos"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Posicion para cualquier fase de salida.
def calcular_posicion(phi, phi_i):
    return (
        np.cos(phi)
        - np.cos(phi_i)
        + (phi - phi_i) * np.sin(phi_i)
    )


# Encuentra el primer regreso posterior a la salida.
def buscar_regreso(phi_i):
    fases = np.linspace(phi_i, phi_i + 2 * np.pi, 2000)
    posiciones = calcular_posicion(fases, phi_i)

    cruces = np.flatnonzero(
        (posiciones[:-1] < 0) & (posiciones[1:] >= 0)
    )

    if len(cruces) == 0:
        raise RuntimeError(
            f"No se encontro regreso para fase {phi_i:.6f}"
        )

    indice = cruces[0]
    izquierda = fases[indice]
    derecha = fases[indice + 1]

    for _ in range(40):
        centro = (izquierda + derecha) / 2

        if calcular_posicion(centro, phi_i) < 0:
            izquierda = centro
        else:
            derecha = centro

    return (izquierda + derecha) / 2


# Fases de salida posteriores al maximo positivo del campo.
# Evitamos los extremos 0 y pi/2 por sus casos limite.
fases_salida = np.linspace(0.01, np.pi / 2 - 0.01, 600)

fases_regreso = np.array([
    buscar_regreso(phi_i)
    for phi_i in fases_salida
])

# Energia cinetica de regreso dividida entre Up.
energias = 2 * (
    np.sin(fases_salida) - np.sin(fases_regreso)
)**2

# Duracion de cada recorrido, en ciclos del laser.
duraciones = (
    (fases_regreso - fases_salida) / (2 * np.pi)
)

# Localizamos la mayor energia entre las fases calculadas.
indice_maximo = np.argmax(energias)

fase_maximo = fases_salida[indice_maximo]
energia_maxima = energias[indice_maximo]

# Comprobaciones de los regresos encontrados.
residuos = calcular_posicion(fases_regreso, fases_salida)

assert np.all(fases_regreso > fases_salida)
assert np.max(np.abs(residuos)) < 1e-10

print("\nRESULTADOS DEL BARRIDO DE FASES")
print(f"Numero de trayectorias: {len(fases_salida)}")
print(f"Energia maxima encontrada: {energia_maxima:.6f} Up")
print(f"Fase de salida correspondiente: {fase_maximo:.6f} rad")
print(
    "Duracion de esa trayectoria: "
    f"{duraciones[indice_maximo]:.6f} ciclos"
)
print(
    "Mayor error de posicion en los regresos: "
    f"{np.max(np.abs(residuos)):.3e}"
)

# Antes del maximo: rama de mayor tiempo de recorrido.
plt.figure()

plt.plot(
    fases_salida[:indice_maximo + 1],
    energias[:indice_maximo + 1],
    label="Trayectorias largas"
)

# Despues del maximo: rama de menor tiempo de recorrido.
plt.plot(
    fases_salida[indice_maximo:],
    energias[indice_maximo:],
    label="Trayectorias cortas"
)

plt.scatter(
    fase_maximo,
    energia_maxima,
    color="red",
    zorder=3,
    label=f"Maximo: {energia_maxima:.4f} Up"
)

plt.xlabel("Fase de salida (rad)")
plt.ylabel("Energia cinetica de regreso / Up")
plt.title("Energia de regreso contra fase de salida")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# PRUEBA 1: repetir el barrido con mas fases de salida.
fases_refinadas = np.linspace(
    0.01, np.pi / 2 - 0.01, 1200
)

regresos_refinados = np.array([
    buscar_regreso(phi_i)
    for phi_i in fases_refinadas
])

energias_refinadas = 2 * (
    np.sin(fases_refinadas) - np.sin(regresos_refinados)
)**2

maximo_refinado = np.max(energias_refinadas)
cambio_maximo = abs(maximo_refinado - energia_maxima)

print("\nPRUEBA DE REFINAMIENTO")
print(f"Maximo con 600 fases:  {energia_maxima:.9f} Up")
print(f"Maximo con 1200 fases: {maximo_refinado:.9f} Up")
print(f"Cambio absoluto: {cambio_maximo:.3e} Up")


# PRUEBA 2: comparar duraciones para una misma energia.
def energia_para_fase(phi_i):
    phi_r = buscar_regreso(phi_i)

    return 2 * (
        np.sin(phi_i) - np.sin(phi_r)
    )**2


def fase_para_energia(izquierda, derecha, objetivo):
    # Buscamos una fase cuya energia coincida con el objetivo.
    valor_izquierdo = (
        energia_para_fase(izquierda) - objetivo
    )
    valor_derecho = (
        energia_para_fase(derecha) - objetivo
    )

    if valor_izquierdo * valor_derecho > 0:
        raise RuntimeError(
            "El intervalo no encierra la energia objetivo."
        )

    for _ in range(40):
        centro = (izquierda + derecha) / 2
        valor_centro = energia_para_fase(centro) - objetivo

        if valor_izquierdo * valor_centro <= 0:
            derecha = centro
        else:
            izquierda = centro
            valor_izquierdo = valor_centro

    return (izquierda + derecha) / 2


energia_objetivo = 2.0  # En unidades de Up.

# Una solucion a cada lado del maximo.
fase_larga = fase_para_energia(
    fases_salida[0], fase_maximo, energia_objetivo
)

fase_corta = fase_para_energia(
    fase_maximo, fases_salida[-1], energia_objetivo
)

duracion_larga = (
    buscar_regreso(fase_larga) - fase_larga
) / (2 * np.pi)

duracion_corta = (
    buscar_regreso(fase_corta) - fase_corta
) / (2 * np.pi)

# Comprobamos las energias y el orden de las duraciones.
assert abs(
    energia_para_fase(fase_larga) - energia_objetivo
) < 1e-8

assert abs(
    energia_para_fase(fase_corta) - energia_objetivo
) < 1e-8

assert duracion_larga > duracion_corta

print("\nCOMPARACION PARA ENERGIA DE REGRESO = 2 Up")
print(
    f"Larga: fase = {fase_larga:.6f} rad; "
    f"duracion = {duracion_larga:.6f} ciclos"
)
print(
    f"Corta: fase = {fase_corta:.6f} rad; "
    f"duracion = {duracion_corta:.6f} ciclos"
)
print("Comprobacion correcta: la larga dura mas que la corta.")