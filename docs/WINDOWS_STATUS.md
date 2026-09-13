# Rama Windows: estado reproducible

Esta rama corrige la identidad de calibración entre Windows y Unix, usa las grabaciones
aportadas en `frontend/audio/`, admite pulsaciones sostenidas y ofrece un checkpoint
de 200 épocas en el selector del simulador. El cerebro del panel derecho usa un solo
blanco, muestra 6.400 somas pequeños sin dibujar conexiones y gira lentamente.

El catálogo también incluye “Through the Fire and Flames”. Su chart Expert público se
alineó por correlación con la edición de cinco minutos, se redujo de cinco a cuatro
carriles y se ajustaron las repeticiones a la frecuencia de control del juego. Al
seleccionarlo, la escena oculta la bandera chilena y la chupalla de la mosca.

## Modelo que se muestra

La entrada visual se codifica como estímulo, MaleCNS propaga actividad por el grafo
completo y un lector externo convierte los spikes muestreados en D, F, J y K. Las
conexiones y parámetros neuronales de MaleCNS permanecen fijos.

El entrenamiento incluido recoge dos ejecuciones neuronales completas de La
Consentida: una silenciosa y otra con las teclas objetivo. Sobre esos datos ajusta el
lector externo durante 200 épocas supervisadas. Una evaluación posterior carga el
checkpoint congelado, vuelve a ejecutar MaleCNS y guarda las acciones y fotogramas.
El botón **Ver jugar a velocidad normal** reproduce esa evaluación usando el MP3 como
reloj. Es una evaluación grabada y congelada, no entrenamiento ni simulación neuronal
en vivo.

## Rendimiento y límites

- El kernel nativo C++ de CPU fue comparado con Brian2 mediante las pruebas numéricas.
- Incluso con ese kernel, un paso de 10 ms tarda aproximadamente 130 ms en el equipo
  usado para esta rama; no alcanza 30 fotogramas por segundo.
- No existe un backend CUDA/GPU implementado. Añadirlo exige portar y validar el
  kernel disperso; seleccionar una GPU en la interfaz no aceleraría el código actual.
- La animación visual usa el reloj de `requestAnimationFrame`, separado del lento reloj
  neuronal. Las evaluaciones grabadas siguen `audio.currentTime` para evitar cortes.
- El mapa de 337 notas y 7 sostenidas fue derivado automáticamente del audio a 115 BPM.
  Tiene sentido rítmico y soporta sostenidas, pero no sustituye una partitura revisada
  por un músico.
- Las 200 épocas entrenan el lector externo. No son 200 generaciones de plasticidad
  sináptica de MaleCNS y no prueban aprendizaje biológico.

## Uso

```powershell
.venv\Scripts\python scripts\serve.py
```

Abra `http://127.0.0.1:8765/`, seleccione **La Consentida**, elija el checkpoint de
200 épocas y pulse **Ver jugar a velocidad normal**. Active **Sonido** si el navegador
no lo activó todavía.
