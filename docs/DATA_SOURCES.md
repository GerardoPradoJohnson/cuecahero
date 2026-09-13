# Fuentes de datos

MaleCNS v1.0, colaboración MaleCNS / FlyEM, HHMI Janelia.

Fuente oficial: https://male-cns.janelia.org/download/

Licencia: CC BY 4.0, según la publicación y THIRD_PARTY.md de DOOMFLY.

Se utilizan exactamente los archivos de doom/datasets.json del commit `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`.

Checksums SHA-256 y tamaños proceden del source.lock.json publicado por DOOMFLY; no son hashes inventados ni una firma del productor. Copia versionada: config/malecns.lock.json.

## annotations.feather

- Versión: MaleCNS v1.0
- Origen: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
- Licencia: CC BY 4.0
- Tamaño exacto: 14483314 bytes (14.48 MB)
- SHA-256: `2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2`
- Uso: Identificadores, clases, lateralidad y anotaciones, incluidas coordenadas de columnas ópticas usadas por el preparador.

## neurotransmitters.feather

- Versión: MaleCNS v1.0
- Origen: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-neurotransmitters-male-cns-v1.0.feather
- Licencia: CC BY 4.0
- Tamaño exacto: 43282834 bytes (43.28 MB)
- SHA-256: `95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621`
- Uso: Predicciones/anotaciones de neurotransmisor; el signo sináptico posterior es una decisión del modelo.

## edges.feather

- Versión: MaleCNS v1.0
- Origen: https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
- Licencia: CC BY 4.0
- Tamaño exacto: 1051241946 bytes (1051.24 MB)
- SHA-256: `e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1`
- Uso: Conectividad dirigida y número de contactos. El importador oficial conserva todas las conexiones entre neuronas retenidas.

## Código y límites

DOOMFLY: https://github.com/nftechie/doomfly ; código original MIT. Se preservan LICENSE, THIRD_PARTY.md, THIRD_PARTY_NOTICES.md y licenses/ en el checkout.

No se requieren archivos adicionales de posiciones, esqueletos ni meshes para este flujo oficial. No se descargan ni se inventan coordenadas anatómicas. La proyección retinal de DOOMFLY es inferida, no una medición de pose ocular.

No se crean datos simulados para reemplazar MaleCNS. Los grafos pequeños del oracle upstream son únicamente fixtures de tests numéricos.

## Uso en el MVP Cueca Hero

El visor utiliza `somaLocation` del archivo oficial `annotations.feather`, con el
mismo SHA-256 ya registrado. Estas coordenadas están incluidas en el dataset existente;
no se descargaron archivos de posiciones adicionales. Se visualizan 1.800 entradas
válidas mediante muestreo determinista y escalado uniforme. La muestra no modifica el
grafo neuronal. Las coordenadas se describen en unidades fuente, sin atribuirles una
conversión física no verificada.

La pista «Primer pañuelo», su calendario y la melodía sintetizada son contenido original
de prueba del proyecto en 6/8. No son una grabación tradicional ni un dataset biológico.
No se descargaron música o assets externos para el MVP.

## Calibración externa del lector (derivado local, 2026-09-11)

- Archivo: `config/perception-readout.json`, 111540 bytes.
- SHA-256: `0cfcf3c5fbdafe2bdae5049a933d35bf6185b9d84434e3660973139ddf370bd7`.
- Origen: scripts locales de calibración, grafo oficial ya citado y pista original de prueba.
- Los IDs y clases derivados de anotaciones MaleCNS conservan la atribución y licencia CC BY 4.0 de su fuente. Los coeficientes son resultados del experimento local, no otro dataset oficial.
- Uso: convertir spikes de poblaciones de lámina en acciones del controlador.
- El archivo incluye el hash del grafo y hashes de los seis registros de entrenamiento. No contiene cambios a los pesos biológicos ni datos anatómicos fabricados.
