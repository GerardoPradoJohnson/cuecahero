# FlyLab · Cueca Hero

MVP local de una plataforma experimental que conecta MaleCNS a entornos interactivos.
Cueca Hero es el primer entorno: un catálogo de canciones con cuatro carriles y notas sostenidas.

## Abrir el juego en este equipo

Desde la raíz del proyecto:

```sh
.venv/bin/python scripts/serve.py
```

En Windows:

```powershell
.venv\Scripts\python scripts/serve.py
```

Abrir **http://127.0.0.1:8765/**. El servidor escucha exclusivamente en este computador.
Si el puerto ya está ocupado, usar `--port 8767` y abrir la dirección que imprime.

- **Tú / Manual:** iniciar y pulsar **D F J K**, o los botones de carril, cuando las notas crucen la línea.
- **MaleCNS / Neuronal:** carga el grafo completo; iniciar para que las cuatro lecturas neuronales controlen la pista.
- **Pausa / Reiniciar:** pausa los relojes o reinicia entorno, filtros, decoder y estado neuronal. Cambiar de controlador inicia una sesión nueva.
- **Sonido:** reproduce `frontend/audio/consentida.mp3`. En partidas evaluadas, el audio es el reloj maestro para mantener sincronizadas música y animación.
- **DragonForce:** “Through the Fire and Flames” usa un chart Expert público adaptado a cuatro teclas y alineado con la edición de cinco minutos aportada.
- **Checkpoint entrenado:** el selector permite ver una evaluación completa congelada a velocidad normal. El cálculo neuronal se hizo antes; la reproducción no se presenta como simulación en vivo.
- **Visor:** arrastrar para rotar; con foco, usar las flechas. Son coordenadas oficiales de somas, con actividad de la muestra visible.
- **Repetición:** reproduce la última sesión, permite pausar y buscar un instante. **Exportar** descarga observaciones, acciones y métricas en JSON.

Las sesiones se guardan automáticamente en `outputs/sessions/`. La más reciente vuelve
a estar disponible después de reiniciar el servidor. Son registros de la ejecución;
no checkpoints para continuar el estado neuronal.

## Preparar otro computador

Se necesita Git para clonar este repositorio. Python se instala mediante uv cuando falta.

### macOS Intel / Apple Silicon

```sh
git clone <URL-de-este-repositorio>
cd cuecahero
bash scripts/setup_mac.sh
source .venv/bin/activate
python scripts/download_malecns.py
python scripts/verify_data.py
python scripts/prepare_data.py --audit
python scripts/serve.py
```

### Windows x86-64

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
.venv\Scripts\python scripts/download_malecns.py
.venv\Scripts\python scripts/verify_data.py
.venv\Scripts\python scripts/prepare_data.py --audit
.venv\Scripts\python scripts/serve.py
```

CPU/Numba es el baseline. El kernel C++ acelera CPU cuando está compilado y verificado.
Esta rama no implementa CUDA ni otro backend GPU. Para C++ en macOS, Command Line Tools; en Windows,
Developer PowerShell con MSVC C++. Si falta el compilador, permanece disponible Numba.
Windows y Apple Silicon tienen scripts, pero aún necesitan validación en esos equipos.

Los tres archivos originales suman 1,109 GB decimales. Reservar al menos 8 GB libres
para datos, derivados y dependencias. El grafo utiliza varios GB de RAM durante la
preparación. `FLYLAB_DATA_DIR` cambia la raíz de datos; usarla en todos los pasos.
`download_malecns.py --plan` muestra archivos y tamaños sin descargar.
`--search-dir /ruta/existente` busca copias válidas y las enlaza sin duplicarlas.
Los enlaces requieren Developer Mode en Windows; alternativamente puede usarse una
raíz existente con `connectome_data/malecns_v1/` mediante `FLYLAB_DATA_DIR`.
Las descargas completas válidas se omiten; un archivo existente corrupto no se sobrescribe.
No ejecutar dos descargas simultáneas sobre la misma raíz.

## Configuración y pruebas

`experiments/cueca_hero.json` selecciona entorno, sensor, decoder, cuerpo, backend y
relojes. `python scripts/serve.py --config /ruta/experimento.json` carga otra configuración.
El registro de componentes permite añadir implementaciones sin modificar el core.

```sh
python -m pytest tests -q
python scripts/test_numerics.py  # comparación upstream; requiere kernel C++
python scripts/run.py --backend cpu  # smoke neuronal sin entorno
```

La interfaz no necesita Node, npm, CDN ni instalación de frontend. Three.js r160
se distribuye localmente con su licencia MIT y hash en `frontend/vendor/`. Python sirve los
archivos estáticos y la API. `node --check frontend/app.js` es una comprobación opcional
de sintaxis para desarrollo. Dependencias fijadas en `requirements.txt` y
`config/python-constraints.txt`; cada setup guarda `outputs/environment-freeze.txt`.

## Qué demuestra este MVP

El ciclo es **píxeles → encoder → MaleCNS → decoder → cuerpo → entorno**. El decoder
no conoce el calendario de notas. La simulación conserva 166.700 neuronas y
25.582.938 conexiones. Solo se muestrea el visor, no la simulación.

MaleCNS mantiene fijas sus conexiones. El checkpoint de La Consentida ajusta durante
200 épocas supervisadas un lector externo de actividad neuronal, usando una pasada
silenciosa y otra con etiquetas de profesor. Después se evalúa una canción completa
sin que el lector vea el calendario de notas. Esto no demuestra comprensión del ritmo
ni plasticidad dentro del cerebro de la mosca.
La dinámica LIF, la retina y el mapeo de salidas son aproximaciones experimentales.
El mapa de teclas se extrajo del MP3 y es una aproximación automática, no una
transcripción musical certificada. La simulación neuronal completa todavía calcula
más lento que tiempo real en este equipo; por eso las evaluaciones guardadas se
reproducen después con el reloj del audio. El reloj neuronal tiene su propia escala.

DOOMFLY se mantiene sin cambios en `external/doomfly`, fijado a un commit. Datos,
compilados, sesiones y entorno virtual están excluidos de Git. No se instala ViZDoom.

- [Arquitectura y extensión](docs/ARCHITECTURE.md)
- [Fuentes, tamaños, licencias y hashes](docs/DATA_SOURCES.md)
- [Validación del setup](docs/VALIDATION.md)
- [Validación del MVP](docs/MVP_VALIDATION.md)
- [Estado y limitaciones de la rama Windows](docs/WINDOWS_STATUS.md)

## Siguiente etapa: aprendizaje

El diagnóstico encontró salidas D/K silenciosas y pulsaciones F/J incluso sin notas.
Ver [diagnóstico y hoja de ruta](docs/LEARNING_ROADMAP.md) antes de activar plasticidad.
`python scripts/diagnose_learning.py --output outputs/diagnostics/nuevo-ensayo.json`
ejecuta una prueba visual controlada con el grafo real y no altera sesiones en vivo.

La [validación de percepción](docs/PERCEPTION_VALIDATION.md) documenta el encoder
con contraste, el lector externo calibrado sobre spikes y los ensayos que fallaron.
Incluye comandos reproducibles para calibración y evaluación en partidas separadas.
Esta calibración no equivale a aprendizaje dentro de MaleCNS.

El [entrenador por episodios](docs/TRAINING.md) permite guardar y reanudar el ajuste
supervisado del lector externo. Verifica hashes de los episodios y checkpoints,
preserva ejecuciones anteriores y exporta un modelo para evaluación independiente.

El [entrenador por refuerzo (RL)](docs/RL_TRAINING.md) permite optimizar la política
del lector en bucle cerrado interactuando directamente con el juego a partir de la
señal de recompensa (`RewardSignal`), con exploración estocástica, checkpoints atómicos
y evaluación frente a baselines independientes.
Las partidas del navegador siguen usando un lector congelado.

La escena low-poly muestra una mosca con guitarra frente a un televisor que usa
los fotogramas reales del juego. «Ver televisor» acerca la cámara. Las animaciones
siguen acciones y eventos registrados, también en replay; no cambian la entrada
del cerebro. Ver [escena 3D y validación](docs/SCENE_3D.md).
