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
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()