# Validación del MVP — 2026-09-11

Equipo: macOS Intel, 8 GB RAM; Python 3.11.16. DOOMFLY permanece sin cambios versionados.

## Pruebas automatizadas

`python -m pytest tests -q`: **16 aprobadas**. Las advertencias son deprecaciones de
Brian2 y sus dependencias; no fallos de los tests.

Se comprobaron: calendario reproducible, reset, ventana de acierto, ausencia de doble
puntuación, expiración de notas, final de sesión, observaciones sin notas futuras,
separación de relojes, decoder silencioso y cooldown, filtrado visual, reset del filtro,
validación de estímulos, equivalencia de ambos backends contra Brian2 con corrientes
arbitrarias, registros de replay, cola HTTP, controles manuales, pausa y rechazo de
entradas inválidas/orígenes externos. Los grafos numéricos sintéticos están limitados
a fixtures en tests; el servidor siempre usa MaleCNS real.

## Ejecuciones sobre el grafo real

- **CPU/Numba:** 20 ms neuronales con la imagen real de Cueca Hero, 11.751 spikes y
  estado finito. 2,277 s de pared incluyendo el primer JIT en esta ejecución.
- **CPU/C++:** sesión completa de 503 pasos; 16,748 s de juego y 5.030 ms neuronales;
  3.004.101 spikes. Sin errores.
- Resultado observado del mapeo fijo: 17 aciertos, 31 notas perdidas, 59 pulsaciones
  fuera de ventana y 1.340 puntos. No es evidencia de aprendizaje ni evaluación
  estadística de rendimiento frente al azar.
- Mediana de cómputo neuronal/entorno por paso: 41,796 ms; máximo 83,521 ms. Estos
  tiempos dependen de la máquina y carga. El ritmo del juego se limita por cómputo;
  la velocidad elegida no garantiza tiempo real.

Informes compactos: `validation/neural-session.json` y `validation/cpu-visual-smoke.json`.
La procedencia del grafo se verificó contra la auditoría antes de cargar.

## Interfaz y persistencia

Verificados en el navegador local: inicio manual, entrada de teclado, carga neuronal,
visualización de somas reales, selección de replay, pausa, búsqueda al último paso y
coincidencia de la puntuación final. Se inspeccionaron layouts estrecho y ancho, sin
errores de consola. La repetición guardada vuelve a estar disponible tras reiniciar
el servidor. Los controles directos quedan desactivados mientras se reproduce.

La herramienta WebMCP `inspect_cueca_session` se registró y devolvió los mismos datos
visibles; un argumento inesperado fue rechazado intencionalmente. Es de solo lectura
y su ausencia en otros navegadores no impide usar el juego.

## Límites explícitos

Windows y Apple Silicon todavía no se han probado en hardware real. El aprendizaje,
otros entornos, encoders adicionales y checkpoints neuronales no están implementados.
La anatomía visible es una muestra de somas, no una reconstrucción completa de neuritas.
El audio es sintetizado y sigue las actualizaciones de la simulación; no se certifica
latencia audiovisual de un videojuego de ritmo profesional.
