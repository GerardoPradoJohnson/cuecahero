# Validación local — 2026-09-11

Equipo: macOS 15.7.9, Intel x86-64, 8 GB RAM. Python 3.11.16.

- Setup ejecutado y repetido correctamente; dependencias sin conflictos (`pip check`).
- C++17 compilado con clang++; símbolo C exportado y cargado correctamente.
- Tres archivos oficiales descargados y verificados por tamaño y SHA-256.
- Segunda planificación reutiliza los tres archivos sin tráfico de descarga.
- Tres tests del descargador aprobados: reutilización, descarga corrupta y archivo existente corrupto.
- Cuatro tests upstream de referencia Brian2 aprobados para CPU/Numba y CPU/C++.
  Se emitieron 50 advertencias de deprecación de dependencias upstream.
- Importación y auditoría oficial completas: 166.700 neuronas, 25.582.938 conexiones,
  124.177.617 contactos. Se comprobaron todas las filas fuente y conexiones de ejecución.
- Prueba de 1 ms neuronal sobre el grafo real, sin corriente: ambos backends completaron
  con cero spikes. CPU: 0,657 s incluyendo primera compilación JIT; C++: 0,00355 s.
  No es un benchmark de actividad sostenida ni demuestra ejecución en tiempo real.
- DOOMFLY permanece sin cambios en archivos versionados.
- Datos originales y derivados ocupan aproximadamente 1,6 GiB; excluidos de Git.

Los informes compactos están en `docs/validation/`. Los tiempos no son comparables
con una simulación estimulada. Esto verifica integridad e implementación numérica,
no validez biológica ni aprendizaje. Windows x86-64 y macOS Apple Silicon cuentan
con scripts, pero no se han ejecutado en esas plataformas durante esta sesión.
El juego Cueca Hero, la visualización y el framework modular aún no se implementan.

## Actualización posterior

El MVP de Cueca Hero ya fue implementado después de esta validación inicial del setup.
Consultar `MVP_VALIDATION.md` para el estado actual, pruebas del juego y resultados
sobre el grafo real.
