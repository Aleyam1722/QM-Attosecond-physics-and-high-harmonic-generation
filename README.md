# Física de attosegundos y generación de armónicos altos

Este repositorio contiene seis programas en Python para estudiar el regreso de un electrón al ion bajo un campo láser y la generación de armónicos altos. A continuación se indica cómo ejecutar cada programa y reproducir las 15 figuras generadas por el código.

## Preparación

Abre una terminal en la carpeta del repositorio e instala las dependencias:

```bash
python -m pip install "numpy>=2.0" matplotlib
```

## 1. Trayectorias clásicas — `Trayectorias_clasicas.py`

Este programa calcula cuándo regresa el electrón al ion y con qué energía lo hace. Los parámetros están definidos en el archivo, por lo que basta con ejecutar:

```bash
python Trayectorias_clasicas.py
```

### Figuras que se obtienen

1. **Trayectoria clásica del electrón** . La primera ventana muestra la posición del electrón durante un ciclo óptico para una fase de salida de 0.30 rad. El punto rojo marca el primer regreso al ion, después de 0.659017 ciclos.
2. **Energía de regreso contra fase de salida**. Cierra la primera ventana para continuar. La segunda muestra el resultado de un barrido de 600 fases de salida. Las dos ramas corresponden a trayectorias largas y cortas, y el punto rojo indica la mayor energía encontrada.

Para guardar cada gráfica, utiliza el botón **Guardar** de la ventana de Matplotlib antes de cerrarla.

### Comprobación en la terminal

El máximo del barrido de 600 fases es `3.173132 Up`, donde `Up` es la energía ponderomotriz. Al repetir el cálculo con 1200 fases, el máximo cambia en `9.676e-07 Up`. Para una misma energía de regreso de `2 Up`, la trayectoria larga dura 0.807276 ciclos y la corta dura 0.469012 ciclos.

## 2. Campo eléctrico del láser — `Campo_laser.py`
Este programa calcula el campo eléctrico de un pulso láser en función del tiempo. A partir de la longitud de onda, la intensidad y el número de ciclos, genera las oscilaciones del campo y la envolvente que determina cómo cambia su amplitud durante el pulso.
```bash
python Campo_laser.py
```

### Figura que se obtiene

- **Pulso láser.** La curva azul muestra el campo eléctrico y las líneas discontinuas indican su envolvente. Para una longitud de onda de 1270 nm y una intensidad de 1.10 × 10¹⁴ W/cm², el pulso alcanza su amplitud máxima cerca del quinto ciclo y termina en el décimo.

Para guardar la gráfica, utiliza el botón **Guardar** antes de cerrar la ventana.

### Comprobación en la terminal

La duración de los 10 ciclos es de `42.3626 fs`. La frecuencia angular es de `0.03587666 u.a.`, la amplitud máxima del campo es de `0.05598570 u.a.` y la energía ponderomotriz es de `0.608794 Hartree`.

## 3. Simulación cuántica — `Simulacion_cuantica.py`
Este programa calcula el estado fundamental del electrón en una dimensión y simula su evolución bajo el pulso láser. Durante cada corrida registra cuánta probabilidad permanece dentro de la caja. Repite la simulación con distintos pasos temporales y tamaños de malla para comprobar el efecto de la resolución, y usa la malla más fina para comparar las dos intensidades.
```bash
python Simulacion_cuantica.py
```

Cuando el programa solicite las dos intensidades, presiona **Enter** en cada pregunta. Se utilizarán `1.00 × 10¹⁴ W/cm²` para el caso 1 y `1.10 × 10¹⁴ W/cm²` para el caso 2.

### Figuras que se obtienen

La primera gráfica corresponde al estado inicial. Las cuatro siguientes muestran la probabilidad de encontrar al electrón dentro de una caja de semilongitud de 400 bohr durante el pulso láser.

| Nombre de la gráfica | Qué se obtiene |
|---|---|
| **Estado fundamental 1D** | Densidad electrónica inicial calculada con 2048 puntos, semilongitud de 100 bohr y paso de tiempo imaginario de 0.02 u.a. La energía obtenida es de −0.500000 Hartree. |
| **Evolución con láser** | Primera corrida del caso 1, con 4096 puntos y paso temporal de aproximadamente 0.02 u.a. |
| **Evolución con láser** | Segunda corrida del caso 1. Se conservan los 4096 puntos y se reduce el paso temporal a 0.01 u.a. |
| **Evolución con láser** | Tercera corrida del caso 1. Se aumenta la malla a 8192 puntos y se conserva el paso de 0.01 u.a. |
| **Evolución con láser** | Corrida del caso 2, con 8192 puntos y paso de 0.01 u.a., para observar el efecto de aumentar la intensidad. |

Para guardar cada gráfica, utiliza el botón **Guardar** antes de cerrar su ventana.

### Comprobación de los resultados

En el caso 1, con 4096 puntos, la fracción absorbida cambia de `0.138380` a `0.138378` al reducir el paso de 0.02 a 0.01 u.a. La diferencia se distingue en la terminal aunque ambas curvas se vean prácticamente iguales.

Con 8192 puntos, al terminar el pulso permanece dentro de la caja el **86.16 %** de la probabilidad inicial para el caso 1 y el **83.72 %** para el caso 2.

### Archivos generados

| Corrida | Archivo de datos |
|---|---|
| Caso 1, 4096 puntos, paso de 0.02 u.a. | `resultados_laser_1.npz` |
| Caso 1, 4096 puntos, paso de 0.01 u.a. | `resultados_laser_1_dt001.npz` |
| Caso 1, 8192 puntos, paso de 0.01 u.a. | `resultados_laser_1_dt001_N8192.npz` |
| Caso 2, 8192 puntos, paso de 0.01 u.a. | `resultados_laser_2_dt001_N8192.npz` |

Conserva estos archivos en la carpeta del proyecto para los siguientes programas. Si vuelves a ejecutar una corrida para la misma intensidad, el programa reutiliza sus datos. Para regenerar su gráfica, mueve el archivo `.npz` correspondiente a una carpeta de respaldo y ejecuta de nuevo la simulación.

## 4. Espectro de armónicos — `Espectro_armonicos.py`
Este programa toma la aceleración del electrón calculada en la simulación cuántica y aplica una transformada de Fourier para obtener el espectro de armónicos. También comprueba cuánto cambia el espectro al usar distintas ventanas de análisis, reducir el paso temporal o aumentar el número de puntos de la malla.
Después de generar los datos de la simulación cuántica, ejecuta:

```bash
python Espectro_armonicos.py
```

### Figuras que se obtienen

| Nombre de la gráfica | Qué muestra | Archivo generado |
|---|---|---|
| **Espectro de armónicos** | El espectro para la primera intensidad, con el corte predicho en el orden 62.84. El panel inferior amplía la región de los órdenes 15 a 35. | `espectro_armonicos_1.png` |
| **Comparación de ventanas** | El mismo intervalo del espectro procesado con ventana gaussiana, con ventana Hann y sin ventana adicional. | `comparacion_ventanas.png` |
| **Refinamiento temporal del espectro** | La comparación entre los espectros al reducir el paso de 0.02 a 0.01 u.a. con la misma malla de 4096 puntos. | `refinamiento_temporal_espectro.png` |
| **Refinamiento espacial del espectro** | La comparación entre los espectros al aumentar la malla de 4096 a 8192 puntos con el mismo paso de 0.01 u.a. | `refinamiento_espacial_espectro.png` |

El programa también guarda los datos del espectro en `espectro_armonicos_1.npz`.

### Comprobación en la terminal

La prueba de frecuencia devuelve el orden `16.999806` para una señal de orden esperado `17`. En los intervalos cercanos a los órdenes 15 a 35, el cociente entre el promedio cerca de órdenes pares y el promedio cerca de órdenes impares es `0.264` con la ventana gaussiana, `0.418` con Hann y `0.650` sin ventana adicional.

Entre los órdenes 15 y 80, la diferencia integrada relativa es de `0.0305 %` al reducir el paso temporal y de `7.681e-08 %` al duplicar el número de puntos espaciales.

## 5. Instantáneas de la densidad electrónica — `Instantaneas.py`
Este programa usa los datos de la primera intensidad para mostrar cómo se distribuye la densidad electrónica durante el pulso. Genera seis perfiles de densidad en instantes seleccionados y un mapa que permite seguir su evolución en un intervalo de dos ciclos del láser.

Este programa utiliza `resultados_laser_1_dt001_N8192.npz`. Para ejecutarlo:

```bash
python Instantaneas.py
```

### Figuras que se obtienen

- **Instantáneas de la densidad electrónica.** Seis paneles muestran cómo se distribuye la densidad a ambos lados del ion a los 5.05, 5.20, 5.35, 5.48, 5.60 y 5.71 ciclos del láser. Se guarda como `instantaneas_densidad.png`.
- **Evolución de la densidad electrónica.** El mapa muestra 401 instantáneas entre los ciclos 4.5 y 6.5. El eje horizontal indica la posición, el vertical indica el tiempo y el color representa la densidad en escala logarítmica. Se guarda como `mapa_densidad.png`.

## 6. Comparación del corte armónico — `Comparar_cortes.py`
Este programa compara los espectros de las dos intensidades usando las corridas con la malla más fina. Para cada intensidad calcula el corte predicho por el modelo semiclásico y estima el corte a partir del espectro simulado. Con ambos resultados se puede ver cuánto aumenta el orden de corte al subir la intensidad del láser.

Para comparar las dos intensidades se necesitan `resultados_laser_1_dt001_N8192.npz` y `resultados_laser_2_dt001_N8192.npz`. Con ambos archivos en la carpeta del proyecto, ejecuta:

```bash
python Comparar_cortes.py
```

### Figura que se obtiene

- **Corte armónico para dos intensidades.** La gráfica superpone los dos espectros, cada uno normalizado a su propio máximo. Las líneas verticales discontinuas indican los cortes predichos y las punteadas marcan los estimados a partir de los espectros. Se guarda como `comparacion_cortes.png`.

### Comprobación en la terminal

| Intensidad (W/cm²) | Orden de corte predicho | Orden de corte estimado | Diferencia en órdenes |
|---|---:|---:|---:|
| 1.00 × 10¹⁴ | 62.84 | 69 | +6.16 |
| 1.10 × 10¹⁴ | 67.73 | 73 | +5.27 |

Al aumentar la intensidad en un 10 %, el corte estimado pasa del armónico 69 al 73.
