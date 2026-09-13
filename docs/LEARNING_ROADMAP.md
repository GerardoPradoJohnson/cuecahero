# Aprendizaje: diagnóstico y orden de desarrollo

## Hallazgo confirmado — 2026-09-11

Las tres sesiones neuronales completas disponibles produjeron la misma secuencia:
D=0, F=38, J=38, K=0 pulsaciones; 17 aciertos de 48 notas. Las lecturas DNa02-L/R
asignadas a D/K no produjeron spikes. DNpe017-L/R, asignadas a F/J, promediaron
29,82 y 28,03 Hz. La semilla y los pesos fijos explican la repetición determinista;
repetir esas sesiones no constituye entrenamiento.

Esto es una limitación del mapeo inicial del MVP y de la respuesta del modelo bajo
estas entradas. No permite inferir una preferencia biológica de la mosca.

Se ejecutaron seis estímulos estáticos de calibración, un segundo neuronal cada uno,
reiniciando estado, encoder y decoder. Se utilizó el grafo real completo, sin cambios
al servidor en vivo ni al checkout original.

| Estímulo | Spikes D / F / J / K | Pulsaciones D / F / J / K |
|---|---|---|
| Pista vacía | 0 / 29 / 29 / 0 | 0 / 8 / 8 / 0 |
| Nota aislada D | 0 / 29 / 29 / 0 | 0 / 8 / 8 / 0 |
| Nota aislada F | 0 / 28 / 28 / 0 | 0 / 7 / 8 / 0 |
| Nota aislada J | 0 / 29 / 29 / 0 | 0 / 8 / 8 / 0 |
| Nota aislada K | 0 / 28 / 28 / 0 | 0 / 7 / 8 / 0 |
| Imagen negra | 0 / 11 / 11 / 0 | 0 / 6 / 6 / 0 |

D y J generaron exactamente la misma traza temporal de los cuatro readouts que la
pista vacía en este ensayo. La imagen negra conserva el sesgo tónico de lámina:
no equivale a corriente nula. Las diferencias globales de actividad prueban que el
modelo cambia con algunas imágenes; no demuestran una representación útil del carril.
Este ensayo estático no caracteriza toda la respuesta a movimiento ni todos los
estados del cerebro. No se concluye que el sistema visual biológico sea incapaz.

Reproducción del diagnóstico, preservando resultados anteriores:

```sh
python scripts/diagnose_learning.py --output outputs/diagnostics/visual-readouts-new.json
```

Resumen verificable en `validation/learning-readiness.json`; las trazas completas
permanecen en `outputs/diagnostics/visual-readouts.json`, fuera de Git.

## Camino prioritario hacia aprendizaje

**Entrenador por refuerzo y persistencia:** se implementó el entrenador por refuerzo
cerrado (`scripts/train_rl.py`) y evaluador comparativo (`scripts/evaluate_rl.py`),
completando el Paso 2. Permite optimizar la política del lector sobre la recompensa
escalar del juego con asignación de crédito por carril, baselines congelado y
aleatorio, y checkpoints verificables con SHA-256. Ver `RL_TRAINING.md`.

1. **Validar percepción y capacidad de actuar.** (Completado). Encoder con contraste,
   poblaciones de lámina y lector calibrado con aciertos en los cuatro carriles y cero
   pulsaciones espurias en controles. Ver `PERCEPTION_VALIDATION.md`.
2. **Crear un entrenador reproducible.** (Completado). Se implementaron dos modalidades:
   a) ajuste supervisado por episodios (`scripts/train_readout.py`, ver `TRAINING.md`)
   y b) entrenamiento interactivo por refuerzo (`scripts/train_rl.py`, ver `RL_TRAINING.md`),
   con semillas independientes, métricas por carril, baselines aleatorio y congelado,
   y checkpoints atómicos verificables.
3. **Implementar plasticidad explícita.** (Completado). Se implementó la regla Hebbiana
   acotada modulada por `RewardSignal` con trazas de elegibilidad sobre las 4.276
   conexiones monosinápticas entre fotorreceptores de retina e interneuronas de lámina.
   Se verificó selectividad de coactividad, acotación estricta ($\pm 5\%$), checkpoints
   atómicos con SHA-256 y restauración exacta bit a bit mediante `erase()`.
   Ver `PLASTICITY_VALIDATION.md` y `experiments/cueca_hero_plastic.json`.
4. **Persistir la memoria.** Separar reset de entorno, estado rápido y pesos; guardar
   checkpoints completos y verificables de pesos, trazas, colas, encoder, decoder,
   configuración y procedencia. La repetición actual solo conserva observaciones.
5. **Demostrar mejora.** Evaluar con pesos congelados en secuencias no entrenadas,
   comparar con los controles y comprobar retención y pérdida de beneficio al
   borrar la memoria. Mostrar curvas y fallos, no solo el mejor puntaje.

Cambiar únicamente el decoder puede ser una alternativa de ingeniería para aprender
un controlador sobre actividad neuronal. Debe etiquetarse como aprendizaje del
controlador; no presentarlo como plasticidad aprendida dentro de MaleCNS.

## Qué falta del diseño original, además del aprendizaje

| Área | Estado actual | Pendiente |
|---|---|---|
| Motor / datos | Grafo real completo, CPU y C++ | Validación de Windows y Apple Silicon |
| Entornos | Cueca Hero + protocolo genérico | Segundo entorno para demostrar reutilización |
| Sensores | Encoder visual con contraste | Audio, tacto, movimiento, propiocepción según necesidad |
| Decoders / cuerpo | Cuatro pulsaciones y GameController | Salidas continuas y otros cuerpos |
| Recompensa | Completado: RL en juego y plasticidad sináptica en MaleCNS | Integración neuromoduladora profunda (complejo central) |
| Relojes | Juego y cerebro separados del render | Pruebas prolongadas y mejor caracterización temporal |
| Telemetría | Métricas, replay y curvas generacionales | Panel web de comparación de checkpoints |
| Visualización | Somas reales muestreados y escena 3D | Registro de escenas intercambiables y más capas |

Orden recomendado: construir ahora el entrenador con controles independientes y
caracterizar los circuitos candidatos antes de introducir plasticidad. La base de
percepción y acción pasó el criterio de ingeniería documentado; la interpretación
biológica y el aprendizaje siguen abiertos, no son interruptores de configuración.
