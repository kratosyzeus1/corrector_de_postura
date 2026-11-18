"""
=============================================================================
MÓDULO: Monitor de Consumo de Recursos (RAM y CPU)
PROYECTO: Asistente Ergonómico Inteligente - Politécnico Grancolombiano
=============================================================================
Mide el consumo real de memoria RAM (Working Set, Memoria Privada y Pico)
y CPU del proceso actual en tiempo real sin requerir dependencias externas
obligatorias (utiliza psutil si está disponible o la API nativa de Windows
vía ctypes como respaldo directo).
=============================================================================
"""

import ctypes
import gc
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    import psutil
    TIENE_PSUTIL = True
except ImportError:
    TIENE_PSUTIL = False


if sys.platform == "win32":
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    class FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    _kernel32 = ctypes.windll.kernel32
    _psapi = ctypes.windll.psapi

    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    _kernel32.GetProcessTimes.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
        ctypes.POINTER(FILETIME),
    ]
    _kernel32.GetProcessTimes.restype = wintypes.BOOL


def _filetime_a_segundos(ft) -> float:
    valor = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
    return valor / 10_000_000.0


@dataclass
class MuestraRecursos:
    timestamp: float
    ram_actual_mb: float
    ram_privada_mb: float
    ram_pico_mb: float
    cpu_pct: float


class MonitorRecursos:
    """
    Monitorea el consumo de memoria RAM y CPU del proceso del asistente.
    Mantiene un historial acotado (deque) para no generar consumo adicional
    creciente en sesiones prolongadas.
    """

    def __init__(self, max_muestras: int = 600, umbral_alerta_ram_mb: float = 450.0):
        self.umbral_alerta_ram_mb = umbral_alerta_ram_mb
        self.historial: deque[MuestraRecursos] = deque(maxlen=max_muestras)
        self._psutil_proc = psutil.Process(os.getpid()) if TIENE_PSUTIL else None
        self._ultima_lectura_ts: Optional[float] = None
        self._ultimo_cpu_seg: Optional[float] = None
        self._num_cpus = max(1, os.cpu_count() or 1)
        self.ultima_muestra: Optional[MuestraRecursos] = None
        self.ram_baseline_mb: float = 0.0
        self.ram_pico_registrado_mb: float = 0.0
        self.fijar_baseline()

    def _leer_memoria_windows_mb(self) -> tuple[float, float, float]:
        pmc = PROCESS_MEMORY_COUNTERS_EX()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        handle = _kernel32.GetCurrentProcess()
        ok = _psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb)
        if not ok:
            return 0.0, 0.0, 0.0
        mb = 1024.0 * 1024.0
        return (
            pmc.WorkingSetSize / mb,
            pmc.PrivateUsage / mb,
            pmc.PeakWorkingSetSize / mb,
        )

    def _leer_cpu_windows_pct(self) -> float:
        creation = FILETIME()
        exit_t = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        handle = _kernel32.GetCurrentProcess()
        if not _kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            return 0.0

        ahora = time.perf_counter()
        cpu_total = _filetime_a_segundos(kernel) + _filetime_a_segundos(user)
        if self._ultima_lectura_ts is None or self._ultimo_cpu_seg is None:
            self._ultima_lectura_ts = ahora
            self._ultimo_cpu_seg = cpu_total
            return 0.0

        dt_wall = max(1e-6, ahora - self._ultima_lectura_ts)
        dt_cpu = max(0.0, cpu_total - self._ultimo_cpu_seg)
        self._ultima_lectura_ts = ahora
        self._ultimo_cpu_seg = cpu_total
        pct = (dt_cpu / dt_wall) * 100.0 / self._num_cpus
        return min(100.0, max(0.0, pct))

    def muestrear(self) -> MuestraRecursos:
        """Toma una muestra instantánea de RAM y CPU del proceso."""
        ahora = time.time()
        ram_actual = 0.0
        ram_privada = 0.0
        ram_pico = 0.0
        cpu_pct = 0.0

        if self._psutil_proc is not None:
            try:
                mem = self._psutil_proc.memory_info()
                mb = 1024.0 * 1024.0
                ram_actual = mem.rss / mb
                ram_privada = getattr(mem, "private", mem.vms) / mb
                ram_pico = getattr(mem, "peak_wset", mem.rss) / mb
                cpu_pct = self._psutil_proc.cpu_percent(interval=None) / self._num_cpus
            except Exception:
                pass
        elif sys.platform == "win32":
            ram_actual, ram_privada, ram_pico = self._leer_memoria_windows_mb()
            cpu_pct = self._leer_cpu_windows_pct()
        else:
            try:
                import resource
                uso = resource.getrusage(resource.RUSAGE_SELF)
                ram_actual = uso.ru_maxrss / 1024.0
                ram_privada = ram_actual
                ram_pico = ram_actual
            except Exception:
                pass

        self.ram_pico_registrado_mb = max(self.ram_pico_registrado_mb, ram_actual, ram_pico)
        muestra = MuestraRecursos(
            timestamp=ahora,
            ram_actual_mb=round(ram_actual, 2),
            ram_privada_mb=round(ram_privada, 2),
            ram_pico_mb=round(self.ram_pico_registrado_mb, 2),
            cpu_pct=round(cpu_pct, 1),
        )
        self.ultima_muestra = muestra
        self.historial.append(muestra)
        return muestra

    def fijar_baseline(self) -> float:
        """Fija la referencia inicial de memoria después de inicializar el sistema."""
        self.historial.clear()
        m = self.muestrear()
        self.ram_baseline_mb = m.ram_actual_mb
        return self.ram_baseline_mb

    def liberar_memoria_temporal(self) -> float:
        """Ejecuta recolección de basura controlada y devuelve la RAM resultante."""
        gc.collect()
        return self.muestrear().ram_actual_mb

    def obtener_estado_memoria(self) -> str:
        """Clasifica el consumo actual de RAM."""
        m = self.ultima_muestra or self.muestrear()
        if m.ram_actual_mb < 250.0:
            return "OPTIMO"
        if m.ram_actual_mb < self.umbral_alerta_ram_mb:
            return "MODERADO"
        return "ALTO"

    def obtener_resumen(self) -> Dict[str, object]:
        """Devuelve un resumen estadístico completo del consumo de RAM y CPU."""
        if not self.historial:
            self.muestrear()

        valores_ram = [m.ram_actual_mb for m in self.historial]
        valores_priv = [m.ram_privada_mb for m in self.historial]
        valores_cpu = [m.cpu_pct for m in self.historial]

        ram_min = min(valores_ram)
        ram_max = max(valores_ram)
        ram_prom = sum(valores_ram) / len(valores_ram)
        ram_actual = valores_ram[-1]
        delta_vs_baseline = ram_actual - self.ram_baseline_mb

        # Estimar tendencia (primera mitad vs segunda mitad) para detectar posibles fugas
        crecimiento_estable = True
        deriva_mb = 0.0
        if len(valores_ram) >= 10:
            mitad = len(valores_ram) // 2
            prom_inicio = sum(valores_ram[:mitad]) / mitad
            prom_final = sum(valores_ram[mitad:]) / (len(valores_ram) - mitad)
            deriva_mb = prom_final - prom_inicio
            crecimiento_estable = deriva_mb < 25.0

        return {
            "backend_medicion": "psutil" if self._psutil_proc is not None else "windows_api_ctypes",
            "muestras_recolectadas": len(self.historial),
            "ram_baseline_mb": round(self.ram_baseline_mb, 2),
            "ram_actual_mb": round(ram_actual, 2),
            "ram_privada_actual_mb": round(valores_priv[-1], 2),
            "ram_min_mb": round(ram_min, 2),
            "ram_promedio_mb": round(ram_prom, 2),
            "ram_max_sesion_mb": round(ram_max, 2),
            "ram_pico_proceso_mb": round(self.ram_pico_registrado_mb, 2),
            "delta_vs_baseline_mb": round(delta_vs_baseline, 2),
            "deriva_memoria_mb": round(deriva_mb, 2),
            "memoria_estable_sin_fugas": crecimiento_estable,
            "cpu_promedio_pct": round(sum(valores_cpu) / len(valores_cpu), 1),
            "estado_ram": self.obtener_estado_memoria(),
        }

    def exportar_serie(self) -> List[Dict[str, float]]:
        """Exporta las muestras históricas para graficación o auditoría."""
        if not self.historial:
            return []
        t0 = self.historial[0].timestamp
        return [
            {
                "segundos": round(m.timestamp - t0, 2),
                "ram_actual_mb": m.ram_actual_mb,
                "ram_privada_mb": m.ram_privada_mb,
                "ram_pico_mb": m.ram_pico_mb,
                "cpu_pct": m.cpu_pct,
            }
            for m in self.historial
        ]
