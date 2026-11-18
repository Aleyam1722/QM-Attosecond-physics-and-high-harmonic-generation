import numpy as np
import matplotlib.pyplot as plt

# Conversiones de unidades.
VELOCIDAD_LUZ = 299792458.0       # metros / segundo
TIEMPO_ATOMICO = 2.4188843265864e-17  # segundos

# I(W/cm²) = CONVERSION_INTENSIDAD * E0(u.a.)².
# Se obtiene de I = epsilon_0 * c * E0(SI)² / 2.
CONVERSION_INTENSIDAD = 3.5094452e16

# Parametros del primer caso.
LONGITUD_ONDA_NM = 1270.0
INTENSIDAD_W_CM2 = 1.1e14
NUMERO_CICLOS = 10

longitud_onda_m = LONGITUD_ONDA_NM * 1e-9

# Frecuencia angular expresada en unidades atomicas.
omega = (
    2 * np.pi * VELOCIDAD_LUZ / longitud_onda_m
) * TIEMPO_ATOMICO

periodo = 2 * np.pi / omega
duracion = NUMERO_CICLOS * periodo


def amplitud_campo(intensidad):
    """Convierte intensidad en W/cm² a amplitud de campo en u.a."""
    return np.sqrt(intensidad / CONVERSION_INTENSIDAD)


def campo_laser(t, intensidad=INTENSIDAD_W_CM2):
    """Campo electrico del pulso; t se expresa en tiempo atomico."""
    t = np.asarray(t)

    # Limitamos la envolvente al intervalo del pulso.
    tiempo_interno = np.clip(t, 0.0, duracion)

    envolvente = np.sin(
        np.pi * tiempo_interno / duracion
    )**2

    portadora = np.cos(
        omega * (tiempo_interno - duracion / 2)
    )

    campo = amplitud_campo(intensidad) * envolvente * portadora

    return np.where(
        (t >= 0.0) & (t <= duracion),
        campo,
        0.0
    )


# Este bloque se ejecuta al correr este archivo directamente.
# Al importarlo desde la simulacion, solo cargamos las definiciones.
if __name__ == "__main__":
    E0 = amplitud_campo(INTENSIDAD_W_CM2)

    # Muestras para revisar y dibujar el pulso.
    # No fijan el paso temporal de la futura simulacion cuantica.
    tiempos = np.linspace(0.0, duracion, 10001)
    campos = campo_laser(tiempos)

    envolvente = E0 * np.sin(
        np.pi * tiempos / duracion
    )**2

    # Comprobamos inicio, final, centro y exterior del pulso.
    assert abs(float(campo_laser(0.0))) < 1e-12
    assert abs(float(campo_laser(duracion))) < 1e-12

    assert np.isclose(
        float(campo_laser(duracion / 2)),
        E0
    )

    assert float(campo_laser(-1.0)) == 0.0
    assert float(campo_laser(duracion + 1.0)) == 0.0

    # Para esta forma con 10 ciclos completos, el area es cero
    # salvo errores numericos.
    area = np.trapezoid(campos, tiempos)
    area_relativa = abs(area) / (E0 * duracion)

    assert area_relativa < 1e-10

    # Energia ponderomotiva basada en la amplitud maxima.
    Up = E0**2 / (4 * omega**2)

    duracion_fs = duracion * TIEMPO_ATOMICO * 1e15

    print("PARAMETROS DEL PULSO LASER")
    print(f"Longitud de onda: {LONGITUD_ONDA_NM:.1f} nm")
    print(f"Intensidad: {INTENSIDAD_W_CM2:.3e} W/cm²")
    print(f"Numero de ciclos: {NUMERO_CICLOS}")
    print(f"Frecuencia angular: {omega:.8f} u.a.")
    print(f"Amplitud del campo: {E0:.8f} u.a.")
    print(f"Duracion: {duracion_fs:.4f} fs")
    print(f"Energia ponderomotiva Up: {Up:.6f} Hartree")
    print(f"Area relativa del campo: {area_relativa:.3e}")
    print("Pruebas del pulso superadas.")

    plt.figure()
    plt.plot(
        tiempos / periodo, campos,
        label="Campo electrico"
    )
    plt.plot(
        tiempos / periodo, envolvente,
        "--", color="black", label="Envolvente"
    )
    plt.plot(
        tiempos / periodo, -envolvente,
        "--", color="black"
    )

    plt.xlabel("Tiempo desde el inicio / periodo del laser")
    plt.ylabel("Campo electrico (u.a.)")
    plt.title(
        f"Pulso: {LONGITUD_ONDA_NM:.0f} nm, "
        f"{INTENSIDAD_W_CM2:.1e} W/cm², "
        f"{NUMERO_CICLOS} ciclos"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    