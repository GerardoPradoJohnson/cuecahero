# Escena de Cueca Hero

El panel izquierdo presenta una habitación low-poly, una mosca con seis patas,
alas translúcidas con venas, ojos facetados, guitarra con cuerdas y controles de
cuatro colores, amplificador, cable, alfombra, lámpara y televisor.
Son mallas procedurales originales del frontend, no reconstrucciones anatómicas
ni archivos biológicos. Las capturas del usuario se usaron como referencia visual;
no se incorporaron sus textos como instrucciones ni afirmaciones científicas.

## Juego y animación

El televisor recibe el PNG del mismo snapshot que usa la interfaz. Sus filas se
proyectan en perspectiva para crear la pista estilo Guitar Hero. No se generan
notas decorativas ni se consulta el calendario futuro. La textura incluye la
puntuación, combo, aciertos, misses y juicios de los eventos del entorno.

Las pulsaciones mueven el brazo de la guitarra. El movimiento ambiental de cuerpo
y alas sigue el reloj del juego y se detiene con la pausa. La iluminación de
respuesta usa eventos reales de acierto/fallo; no representa dopamina. Las
preferencias de movimiento reducido desactivan esos movimientos corporales.
El mismo camino visual sirve para snapshots en vivo y grabaciones.

El encoder neuronal sigue recibiendo el render 320×400 original, no una captura
de la habitación 3D. Esta modificación es de presentación; no altera la calibración,
las reglas, las recompensas, los pesos o las fuentes de MaleCNS.

## Controles y compatibilidad

- «Escena 3D»: vista completa, arrastre horizontal y flechas para variar el ángulo.
- «Ver televisor»: cámara cercana para leer las notas y jugar con D/F/J/K.
- Sin WebGL o al perder contexto: se muestra la pista 2D original y los controles
  siguen disponibles. La representación 3D requiere soporte WebGL del navegador,
  independiente del backend neuronal CPU.
- Resolución limitada a 1,5× y escena a 30 FPS; sin postprocesamiento ni modelos
  externos. No se presupone NVIDIA/CUDA.

Three.js 0.160.0 proviene del tag oficial r160; la copia local y licencia MIT están
en frontend/vendor, con URL, tamaño y SHA-256 en three.lock.json. La API utiliza
CanvasTexture: https://threejs.org/docs/#api/en/textures/CanvasTexture . No hay CDN,
instalación npm ni descargas de assets durante el uso.

## Validación

Se inspeccionaron en el navegador la habitación, la vista cercana del televisor,
las notas reales durante una sesión manual, la pausa y el replay de un acierto
perfecto (100 puntos, combo 1, 100% de precisión). No hubo errores en la consola
del navegador. Se comprobaron sintaxis de
los módulos y pruebas HTTP. El panel de anatomía y controles existentes se conservan.

## Corrección de feedback y modo neuronal

Los snapshots incluyen contadores y último resultado por carril; sobreviven a
frames que el polling no alcanza a mostrar y se reinician con la sesión. Los
botones conservan esos valores y no se atenúan como indicadores neuronales.
Una banda visible en modo manual explica que el cerebro está detenido y permite
activar MaleCNS. Verificados el reloj neuronal, spikes, somas activos y contadores
en una sesión neuronal real; prueba de regresión para feedback persistente y reset.
