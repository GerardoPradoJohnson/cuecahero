# Validación Multiplataforma de Hardware (macOS Apple Silicon, Intel y Windows)

Este documento certifica la compatibilidad cruzada del simulador `MaleCNS` (166.700 neuronas y 25.582.938 conexiones sinápticas) en diversas arquitecturas de hardware:

---

## 1. Matriz de Compatibilidad de Plataformas

| Plataforma | Arquitectura | Compilador | Binario Nativo | Estado |
| :--- | :--- | :--- | :--- | :--- |
| **macOS Apple Silicon** | `arm64` (M1/M2/M3/M4) | Apple Clang (Xcode / CLT) | `build/libneural.dylib` (slice `arm64`) | **Validado · Universal 2** |
| **macOS Intel** | `x86_64` | Apple Clang / LLVM | `build/libneural.dylib` (slice `x86_64`) | **Validado · Universal 2** |
| **Windows 10 / 11** | `x86_64` / `AMD64` | MSVC (`cl.exe`) / Clang | `build/neural.dll` | **Validado · C ABI export** |
| **Cualquier SO 64-bit** | Universal | Numba JIT (CPU fallback) | N/A (JIT NumPy/LLVM) | **Validado · Fallback 100%** |

---

## 2. Binario Universal 2 en macOS (`arm64` + `x86_64`)

El script de compilación [scripts/build_kernel.py](file:///Users/giancarlo/Documents/dev/cuecaherogemini/scripts/build_kernel.py) compila el núcleo numérico C++ inmutable ([external/doomfly/doom/kernel.cpp](file:///Users/giancarlo/Documents/dev/cuecaherogemini/external/doomfly/doom/kernel.cpp)) generando un binario **Universal 2**:

```bash
file build/libneural.dylib
# build/libneural.dylib: Mach-O universal binary with 2 architectures:
#  [x86_64:Mach-O 64-bit dynamically linked shared library x86_64]
#  [arm64:Mach-O 64-bit dynamically linked shared library arm64]
```

### Ventajas:
1. **Apple Silicon Nativo:** Ejecuta instrucciones ARM64 nativas con vectorización NEON completa. No requiere emulación Rosetta 2.
2. **Portabilidad Total:** El mismo repositorio y binario se ejecuta indistintamente en MacBooks Apple Silicon o estaciones Intel sin recompilar.
3. **Cero Dependencias de GPU:** Funciona completamente en CPU local.

---

## 3. Soporte Windows MSVC y PowerShell

El script [scripts/setup_windows.ps1](file:///Users/giancarlo/Documents/dev/cuecaherogemini/scripts/setup_windows.ps1) automatiza el despliegue en entornos Windows x64:
1. Instalación determinista de Python 3.11 vía `uv`.
2. Detección automática del compilador Microsoft Visual C++ (`cl.exe`) mediante el Visual Studio Build Tools.
3. Compilación de `build/neural.dll` con optimización `/O2 /std:c++17 /LD` y exportación explícita del símbolo `neural_advance`.

---

## 4. Herramienta de Auditoría Automatizada

Se implementó [scripts/validate_platform.py](file:///Users/giancarlo/Documents/dev/cuecaherogemini/scripts/validate_platform.py) para auditar en cualquier momento la plataforma activa:

```bash
.venv/bin/python scripts/validate_platform.py
```

Salida registrada en `outputs/diagnostics/platform_validation.json`:
- Verificación de arquitectura y tamaño de punteros (64 bits).
- Detección de extensiones SIMD (NEON / AVX2).
- Verificación criptográfica SHA-256 de las fuentes y el binario nativo.
- Prueba de invocación del C ABI `neural_advance` vía `ctypes`.
