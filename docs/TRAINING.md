# Entrenamiento por episodios y memoria del lector

Este entrenador ajusta el **lector externo** mediante regresión supervisada sobre
spikes reales ya registrados. No modifica MaleCNS, no aprende de recompensas del
juego y no continúa aprendiendo automáticamente al pulsar «Iniciar sesión».
El navegador mantiene el lector validado y congelado.

## Uso

Con `.venv` activado y las seis secuencias de calibración disponibles:

```sh
python scripts/train_readout.py --model config/perception-readout.json --episodes outputs/calibration/perception-v4 --output outputs/training/nuevo --stop-after 3
python scripts/train_readout.py --resume outputs/training/nuevo/episode-0003.npz --episodes outputs/calibration/perception-v4 --output outputs/training/reanudado
python scripts/validate_perception.py --model outputs/training/reanudado/model.json --output outputs/training/reanudado/evaluation.json --seeds 1301 1409 1511
```

Utiliza nombres de salida nuevos. El comando se niega a sobrescribir ejecuciones o
checkpoints. Para obtener un control sin ajuste del lector, usa `--stop-after 0`.
Para regenerar los episodios de datos oficiales, sigue `PERCEPTION_VALIDATION.md`;
no se descargan otros datasets. Si se regeneran archivos, utiliza también el modelo
producido por esa calibración, que declara sus hashes.

La lista de archivos y sus hashes define explícitamente el conjunto de entrenamiento.
Un episodio fuera de esa lista o cuyo contenido haya cambiado se rechaza. Cada
reanudar verifica también los archivos ya consumidos y su orden. El informe de
entrenamiento es `progress.json`; los MSE son ajuste de entrenamiento y no puntajes
de partidas. La evaluación utiliza exclusivamente imágenes y actividad neuronal;
las etiquetas docentes no entran al controlador en ejecución.

## Qué conserva un checkpoint

- Normalización de las características y configuración completa del lector.
- Estadísticas suficientes acumuladas de la regresión, regularización y ponderación.
- Episodios consumidos, métricas y hashes; identidad del grafo, sensor e IDs neuronales.
- Digest SHA-256 del contenido y arrays, verificado antes de restaurar.

El algoritmo es determinista y no usa un generador aleatorio interno. La memoria
permite **reanudar en el límite de un episodio**. El estado rápido del cerebro,
colas de spikes y estado del entorno no se guardan: no es un checkpoint neuronal
para reanudar a mitad de una partida. La historia temporal se inicializa a cero al
comenzar cada episodio. La normalización queda fija y procede del conjunto de
entrenamiento de la calibración, no de partidas de evaluación.

Se publica cada archivo solo después de escribirlo y sincronizarlo, sin sobrescribir
un checkpoint anterior. Si se interrumpe el proceso durante la escritura, el último
checkpoint publicado permanece disponible. Los checkpoints están en outputs/, fuera
de Git. Los coeficientes exportados no se activan automáticamente en el navegador.

## Evidencia

Con los seis registros reales (3.018 pasos), restaurar después del tercer episodio
produjo estadísticas **idénticas bit a bit** y el mismo modelo que entrenar de corrido.
La diferencia máxima con la calibración por lote previamente validada fue
4,81 × 10⁻¹⁵ en los coeficientes. Esto verifica la continuidad del algoritmo, no una
mejora sobre el lector anterior, que usa los mismos datos y objetivo.

`docs/validation/episodic-training.json` conserva el resultado y métricas por episodio.
Los tests verifican restauración, rechazo de corrupción, aislamiento del conjunto de
entrenamiento, preservación de checkpoints existentes y ausencia de historia de
un episodio anterior. Reinicializar el entrenador produce coeficientes de salida
nulos; volver a cargar el checkpoint recupera el ajuste.

En una partida independiente (semilla 1201), el modelo restaurado consiguió
30/48 aciertos, cuatro pulsaciones vacías y aciertos D/F/J/K = 5/5/12/8.
Los controles de imagen negra y tablero vacío completaron la duración de la
partida con cero pulsaciones. Las 29 pruebas de código pasaron. Esta evaluación
de restauración usa una semilla nueva; no reemplaza el requisito de tres partidas
para promover un controlador distinto.

## Siguiente paso: aprendizaje por refuerzo implementado

El entrenador por refuerzo durante el juego con exploración y evaluación separadas
se encuentra implementado en `scripts/train_rl.py`, `scripts/evaluate_rl.py` y
`training/rl.py`. Permite optimizar la política del lector externo a partir de la
señal de recompensa del juego (`RewardSignal`), conservando checkpoints atómicos
y evaluando frente a baselines independientes. Ver [docs/RL_TRAINING.md](RL_TRAINING.md).

Para atribuir el aprendizaje directamente al conectoma, el siguiente paso es activar
y evaluar las reglas de plasticidad sináptica sobre conexiones existentes (Paso 3).
