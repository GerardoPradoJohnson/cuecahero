# Percepción y calibración del lector externo

## Alcance

El encoder nuevo recibe exclusivamente RGB. Conserva el grafo completo y estimula
1.473 receptores oficiales dentro de la región UV [0,08; 0,10; 0,92; 0,30].
Esa región se proyecta sobre toda la imagen. Se resta el fondo oscuro y se aplica
un campo receptivo gaussiano antes del muestreo, para evitar puntos ciegos.
Es una adaptación de ingeniería, no una estimación validada del campo visual animal.

El lector usa 87 poblaciones, con 865 interneuronas L1/L2/L3/L5 seleccionadas por
conexiones reales desde esos receptores. Las posiciones de las poblaciones son
centroides de la proyección del encoder ponderados por conexiones, no coordenadas
anatómicas nuevas. Todas las neuronas y conexiones siguen siendo simuladas.

La calibración supervisada modifica únicamente coeficientes del lector externo.
No cambia pesos de MaleCNS ni implementa plasticidad biológica. Las etiquetas y
acciones docentes existen solo durante la recolección offline. La inferencia
recibe únicamente actividad neuronal; no calendario, puntuación ni recompensa.

## Resultados intermedios y fallos conservados

- El encoder inicial muestreaba muy pocas notas; en un ensayo D y J no cambiaban
  los receptores muestreados y el fondo casi saturaba la corriente visual.
- La proyección con contraste responde a los cuatro carriles en seis posiciones
  ensayadas. Esto verifica cobertura de entrada, no comprensión.
- El primer lector sobre células individuales no generalizó la temporización.
- El lector por poblaciones con umbral 0,35 produjo pulsaciones prematuras.
- Seleccionar el umbral con los 20 ensayos de entrenamiento (sin consultar los
  resultados de evaluación) eligió 0,575, con 16 aciertos y cero pulsaciones
  vacías en entrenamiento. En partidas completas nuevas rindió mucho menos:

| Semilla | Aciertos / 48 | Pulsaciones vacías | Aciertos D / F / J / K |
|---|---|---|---|
| 41 | 13 | 3 | 1 / 1 / 3 / 8 |
| 73 | 15 | 0 | 5 / 4 / 1 / 5 |
| 109 | 10 | 2 | 2 / 3 / 1 / 4 |

Con imagen negra y con tablero vacío hubo cero pulsaciones durante una partida
completa en cada control. Esta versión no se activa como configuración por defecto:
responde a la imagen, pero la calibración con notas aisladas no basta.
Los resultados completos permanecen en `outputs/calibration/perception-v3/`.

## Reproducción

Desde la raíz, con entorno virtual activado y datos oficiales verificados:

```sh
python scripts/calibrate_perception.py --output outputs/calibration/isolated-new
python scripts/select_readout_threshold.py --calibration outputs/calibration/isolated-new --output outputs/calibration/threshold-new
python scripts/validate_perception.py --model outputs/calibration/threshold-new/model.json --output outputs/calibration/threshold-new/evaluation.json
python scripts/calibrate_sequence_readout.py --base-model outputs/calibration/threshold-new/model.json --output outputs/calibration/sequences-new
python scripts/fit_sequence_readout.py --calibration outputs/calibration/sequences-new --output outputs/calibration/temporal-new
python scripts/validate_perception.py --model outputs/calibration/temporal-new/model.json --output outputs/calibration/temporal-new/evaluation.json --seeds 857 953 1061
python scripts/activate_readout.py --model outputs/calibration/temporal-new/model.json --evaluation outputs/calibration/temporal-new/evaluation.json
```

Cada script se niega a sobrescribir resultados. Las respuestas de entrenamiento
por episodio se guardan para poder ajustar el lector sin repetir la simulación.
Los datos y resultados grandes permanecen fuera de Git; no se descargaron nuevos
datasets ni se modificó el checkout upstream.

La excepción es `activate_readout.py`, que instala deliberadamente el modelo y
actualiza la configuración activa solo si el informe del mismo hash supera el
criterio. Conserva la configuración legacy y copia el informe compacto a docs.
Después hay que reiniciar el servidor; la evaluación por sí sola no lo cambia.

Antes de evaluar el lector de secuencias se fijó este criterio de activación:
al menos 24/48 aciertos, algún acierto en cada carril y como máximo diez pulsaciones
vacías en **cada** partida nueva; cero pulsaciones en los dos controles sin notas
visibles. Superarlo sería una mejora del controlador, no evidencia de aprendizaje
dentro del conectoma. No se reajusta el modelo con esas partidas de evaluación.

## Resultado de secuencias: lector activado

El lector temporal congelado superó el criterio en las tres semillas nuevas.

| Semilla | Aciertos / 48 | Pulsaciones vacías | Aciertos D / F / J / K |
|---|---|---|---|
| 857 | 30 | 7 | 9 / 5 / 8 / 8 |
| 953 | 24 | 6 | 7 / 6 / 7 / 4 |
| 1061 | 27 | 8 | 7 / 8 / 8 / 4 |

Los dos controles completaron 503 pasos sin pulsaciones. La configuración
por defecto ahora carga `config/perception-readout.json`. Se conservan el
modelo original y los ensayos fallidos. Informe: `validation/perception-evaluation.json`.

25 pruebas de código pasaron, incluidas historia causal, reset, poblaciones,
cobertura retinal y ciclo HTTP. Las dos pruebas HTTP requirieron ejecución fuera
del sandbox para abrir sus puertos locales.

Esto no alcanza un control perfecto: todavía hay 18–24 notas perdidas por partida.
No demuestra plasticidad ni aprendizaje dentro de MaleCNS. El siguiente paso es
un entrenador con recompensas y memoria verificable, usando estos controles.
