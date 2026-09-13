# Entrenamiento por Refuerzo (RL) durante el Juego y Memoria del Lector

Este módulo implementa el **entrenamiento por recompensa / refuerzo (RL) en bucle cerrado** para el lector externo de MaleCNS, completando el **Paso 2** de la hoja de ruta hacia el aprendizaje ([docs/LEARNING_ROADMAP.md](file:///Users/giancarlo/Documents/dev/cuecaherogemini/docs/LEARNING_ROADMAP.md)).

A diferencia del ajuste supervisado offline ([docs/TRAINING.md](file:///Users/giancarlo/Documents/dev/cuecaherogemini/docs/TRAINING.md)), este entrenador aprende **interactuando directamente con el juego** y optimiza la política a partir de la señal escalar de recompensa (`RewardSignal`) de Cueca Hero:
* `+1.0` por acierto perfecto.
* `+0.6` por acierto bueno.
* `-0.1` por pulsación vacía / fuera de ventana.
* `-1.0` por nota perdida (*miss*).

---

## Principios de Ingeniería y Algoritmo

1. **Separación de Exploración y Evaluación:**
   - **Durante el entrenamiento:** La política es estocástica: calcula los logits $z_{t,i} = x_t W_i + b_i$ a partir de las poblaciones de interneuronas de lámina y muestrea acciones con probabilidad $p_{t,i} = \sigma((z_{t,i} - \theta) / \tau_{\text{explore}})$ cuando el carril está rearmado y ha superado el tiempo de enfriamiento (*cooldown*).
   - **Durante la evaluación:** La política es estrictamente determinista: dispara únicamente si $z_{t,i} \ge \theta$ y el carril está armado.
2. **Asignación de Crédito por Carril y Línea de Base Móvil:**
   - Se calculan retornos acumulados con factor de descuento $\gamma = 0.95$ por carril ($G_{t,l}$).
   - Una línea de base móvil $\bar{G}_l$ reduce la varianza para estimar la ventaja $A_{t,l} = G_{t,l} - \bar{G}_l$.
   - La gradiente de la política acumula:
     $$\nabla_\theta J = \frac{1}{T} \sum_{t} A_{t,l} \cdot (a_{t,l} - p_{t,l}) \cdot \frac{x_t}{\tau_{\text{explore}}} - \lambda (\theta - \theta_0)$$
   - Las actualizaciones se aplican mediante un optimizador Adam adaptativo.
3. **Persistencia y Checkpoints Verificables:**
   - En cada límite de episodio se guarda un archivo atómico `.npz` con los pesos, sesgos, momentos de Adam ($m, v$), línea de base acumulada, lista de episodios consumidos y resumen criptográfico SHA-256.
   - El proceso de guardado utiliza un archivo temporal y reemplazo atómico (`os.replace`) con sincronización a disco (`fsync`), evitando la corrupción de checkpoints si el entrenamiento se interrumpe.
   - Se puede reanudar exactamente desde cualquier episodio guardado mediante `--resume`.
4. **Baselines Obligatorios y Evaluación:**
   - Se evalúa en semillas no vistas durante el entrenamiento.
   - Se compara contra:
     * **Baseline aleatorio:** agente que pulsa al azar a una cadencia comparable.
     * **Baseline congelado inicial:** modelo previo sin entrenamiento por refuerzo.
     * **Controles negativos:** pantalla negra y pista vacía sin notas (debe producir 0 pulsaciones).

---

## Comandos de Uso

Con el entorno virtual `.venv` activado:

### 1. Entrenar la política por refuerzo

```sh
python scripts/train_rl.py --base-model config/perception-readout.json --episodes 12 --seeds 211 307 401 503 601 701 --episodes-dir outputs/calibration/perception-v4 --output outputs/training/rl-v1
```

Opciones principales:
* `--base-model`: archivo JSON del modelo base inicial.
* `--episodes`: número total de episodios a entrenar.
* `--seeds`: lista de semillas para alternar durante el entrenamiento.
* `--episodes-dir`: directorio opcional con características neuronales precalculadas para entrenamiento rápido en replay cerrado (cada episodio tarda ~0.1 s). Si no se proporciona o faltan semillas, se simula MaleCNS en vivo.
* `--lr`: tasa de aprendizaje Adam (por defecto: `1e-3`).
* `--explore-temp`: temperatura de exploración (por defecto: `0.25`).
* `--gamma`: factor de descuento (por defecto: `0.95`).

### 2. Reanudar el entrenamiento desde un checkpoint

```sh
python scripts/train_rl.py --resume outputs/training/rl-v1/episode-0006.npz --episodes 12 --seeds 211 307 401 503 601 701 --episodes-dir outputs/calibration/perception-v4 --output outputs/training/rl-v1-resumed
```

### 3. Evaluación de la curva de aprendizaje generación tras generación

```sh
python scripts/evaluate_generations.py --run-dir outputs/training/rl-v3 --output outputs/training/rl-v3/generation_curve.json
```

Evalúa el desempeño determinista del modelo en cada uno de los checkpoints generados a lo largo del entrenamiento sobre semillas de test independientes (`[857, 953, 1061]`).

### 4. Evaluación comparativa en semillas independientes

```sh
python scripts/evaluate_rl.py --model outputs/training/rl-v3/model.json --baseline config/perception-readout.json --seeds 857 953 1061 --output outputs/training/rl-v3/evaluation.json
```

---

## Evidencia Experimental de Aprendizaje Generación tras Generación

Evaluando la progresión completa a través de 24 generaciones/episodios de entrenamiento sobre el conjunto de prueba independiente (`seeds 857, 953, 1061`), se verifica una mejora sostenida de la precisión y reducción de errores:

| Generación | Checkpoint | Aciertos / 48 | Pulsaciones Vacías | Precisión (%) | Puntuación Media |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Gen 0 (Base)** | `baseline` | 27.7 | 7.0 | **57.7%** | 1.713 |
| **Gen 1** | `episode-0001.npz` | 32.0 | 1.7 | **66.7%** | 2.213 |
| **Gen 4** | `episode-0004.npz` | 32.3 | 1.3 | **67.4%** | 2.447 |
| **Gen 7** | `episode-0007.npz` | 33.3 | 0.0 | **69.5%** | 2.720 |
| **Gen 12** | `episode-0012.npz` | 34.0 | 1.0 | **70.8%** | 2.893 |
| **Gen 18** | `episode-0018.npz` | 34.0 | 1.0 | **70.8%** | 2.933 |
| **Gen 24** | `episode-0024.npz` | **34.3** | **0.0** | **71.5%** | **3.027** |

* **Ganancia en precisión:** aumentó de **57.7%** a **71.5%** (+13.8 puntos porcentuales).
* **Supresión de errores:** las pulsaciones vacías fuera de tiempo cayeron de **7.0** a **0.0** (eliminación de disparos espurios).
* **Incremento de puntuación:** subió de **1.713** a **3.027** puntos (+76.7%).

---

## Resultados y Validación

* **Tests unitarios:** `tests/test_rl_training.py` comprueba inicialización, diferenciación entre pasos estocásticos y deterministas, acumulación del gradiente de política, guardado/restauración de checkpoints bit a bit, y rechazo de archivos alterados o corruptos.
* **Integridad:** Las etiquetas docentes no se usan durante la inferencia; el agente recibe únicamente `NeuralActivity` y devuelve `Action`.
* **Límites explícitos:** Este aprendizaje por refuerzo optimiza los coeficientes del lector externo / controlador BCI. No modifica los pesos sinápticos del conectoma de MaleCNS ni simula dopamina biológica.
