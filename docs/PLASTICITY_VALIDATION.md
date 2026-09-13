# Validación de Plasticidad Sináptica Explícita en MaleCNS

Este documento registra la implementación, verificación y límites del **Paso 3 de la hoja de ruta**: la incorporación de plasticidad sináptica explícita modulada por recompensa sobre conexiones biológicas existentes en el conectoma MaleCNS v1.0.

---

## 1. Subcircuito Seleccionado e Hipótesis Anatómica

En lugar de crear sinapsis artificiales o modificar conexiones arbitrarias, se identificó el subcircuito monosináptico directo existente entre la entrada sensorial y las poblaciones del decodificador:
* **Población Presináptica:** 1.473 fotorreceptores de la retina dentro de la región ultravioleta oficial `[0.08, 0.10, 0.92, 0.30]`, estimulados por el sensor de contraste.
* **Población Postsináptica:** 865 interneuronas de lámina (L1, L2, L3, L5) conectadas directamente a dichos receptores, cuyas tasas alimentan el decodificador temporal calibrado.
* **Sinapsis Plásticas:** Exactamente **4.276 conexiones sinápticas existentes** en el grafo oficial.
* **Pesos Basales ($W_0$):** Distribución original verificada en memoria:
  * Rango: $[-25.575, -0.275]$.
  * Media basal: $-7.1099$.
  * El grafo inmutable en disco (`malecns.npz`) y la auditoría de integridad permanecen intactos. Las variaciones ocurren exclusivamente en memoria durante la simulación interactiva.

---

## 2. Regla de Plasticidad Modulada por Recompensa

La regla opera como una hipótesis de ingeniería inspirada en aprendizaje asociativo acotado:

1. **Traza de Elegibilidad ($e_j$):**
   $$e_j(t + \Delta t) = e_j(t) \cdot e^{-\Delta t / \tau} + \mathbb{I}(\text{pre}_j > 0 \land \text{post}_j > 0)$$
   Con $\tau = 750\text{ ms}$ y $\Delta t = 10\text{ ms}$ (paso neuronal). La coincidencia binaria evita sobrestimar el disparo tónico continuo de la lámina.

2. **Modulación por `RewardSignal` ($R$):**
   $$\Delta f_j = \eta \cdot R \cdot \frac{e_j}{\max(1, \max e)}$$
   Con tasa de aprendizaje $\eta = 0.002$.
   * $R > 0$ (aciertos de notas): refuerza la eficacia de las sinapsis con coactividad reciente.
   * $R < 0$ (notas perdidas o pulsaciones vacías fuera de tiempo): debilita la eficacia de las sinapsis activas.

3. **Límites Fisiológicos Acotados:**
   $$f_j \leftarrow \text{clip}(f_j + \Delta f_j, \, 0.95, \, 1.05)$$
   $$W_j(t) = W_{0,j} \cdot f_j$$
   La eficacia sináptica está restringida a una ventana de variación de $\pm 5\%$, preservando la estabilidad global de la dinámica LIF de MaleCNS.

4. **Inmutabilidad y Borrado:**
   * La llamada `plasticity.erase(brain)` restaura los 4.276 pesos exactamente a su valor basal $W_{0,j}$ y reinicia las trazas de elegibilidad a cero.

---

## 3. Comandos de Uso y Reproducción

### Configuración del Experimento
El archivo `experiments/cueca_hero_plastic.json` declara la activación de plasticidad en tiempo de ejecución:
```json
"plasticity": {
  "type": "reward_modulated",
  "enabled": true,
  "eta": 0.002,
  "eligibility_tau_ms": 750.0,
  "minimum_fraction": 0.95,
  "maximum_fraction": 1.05
}
```

### Ejecutar Sesión de Entrenamiento de Plasticidad
```sh
python scripts/train_plasticity.py --config experiments/cueca_hero_plastic.json --episodes 10 --output outputs/training_plasticity/run1
```

### Ejecutar Validación Rigurosa y Controles
```sh
python scripts/validate_plasticity.py --checkpoint outputs/training_plasticity/run1/episode-0010.npz --output outputs/training_plasticity/run1/validation.json
```

---

## 4. Evidencia Experimental

La evaluación experimental sobre el modelo adaptado en 5 episodios (`episode-0005.npz`) frente al conectoma basal en semillas de prueba independientes (`857, 953, 1061`) arrojó los siguientes resultados:

| Condición | Aciertos / 48 (Media) | Pulsaciones Vacías (Media) | Puntuación Media | Spikes Totales (Media) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Pesos Congelados)** | 27,0 | 7,0 | 1.760,0 | 1.485.151 |
| **Adaptado (3.174 sinapsis modificadas)** | 27,3 | 7,7 | 1.653,3 | 1.484.726 |
| **Borrado (`erase()`)** | **27,0** | **7,0** | **1.760,0** | **1.485.151** |

* **Selectividad Sináptica:**
  * **3.174 de las 4.276 conexiones (74,23%)** experimentaron coactividad y modularon su eficacia.
  * El 25,77% restante permaneció inalterado exactamente en su peso basal $W_0$.
  * Límites observados de eficacia: $[0.9906, \, 1.0094]$, estrictamente dentro de la cota acotada de $\pm 5\%$.
* **Restauración Exacta tras Borrado:**
  * Invocar `erase()` restableció los 4.276 pesos exactamente a su valor basal. Las tres partidas de evaluación produjeron **exactamente el mismo número de spikes (1.483.340, 1.485.279 y 1.486.834), puntuación y aciertos bit a bit** que el baseline inicial (`bit_identical: True`).
* **Controles Negativos de Estabilidad:**
  * Pantalla negra: **0 pulsaciones** (503 fotogramas).
  * Tablero vacío sin notas: **0 pulsaciones** (503 fotogramas).
  * Se confirma que la plasticidad sináptica no induce sobreactivación espontánea ni actividad epileptiforme.

---

## 5. Límites Explícitos

1. **Hipótesis de ingeniería:** La señal de recompensa es un escalar algebraico externo derivado de los juicios del entorno de Cueca Hero; no modela la liberación de dopamina biológica ni receptores acoplados a proteínas G (GPCRs).
2. **Conectividad monosináptica:** La plasticidad actual actúa sobre las 4.276 conexiones entre fotorreceptores y lámina. No modifica conexiones interneuronales profundas en el lóbulo óptico o el complejo central.
3. **Puntuaciones del juego:** La adaptación sináptica modula la temporización de llegada de la corriente visual a la lámina; los pesos del decodificador externo permanecen fijos durante este ensayo para aislar el efecto conectómico.
