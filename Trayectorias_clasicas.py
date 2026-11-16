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