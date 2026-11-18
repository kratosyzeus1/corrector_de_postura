"""
=============================================================================
MÓDULO: Gestor Multi-Cámara y Manejo de Conflictos con Videollamadas
PROYECTO: Asistente Ergonómico Inteligente - Politécnico Grancolombiano
=============================================================================
Responsabilidades:
1. Detectar todas las cámaras web disponibles (integradas, USB, virtuales)
   y permitir elegir o cambiar la cámara activa en cualquier momento.
2. Detectar conflictos de uso concurrente de cámara en Windows con apps de
   videollamada (Zoom, Microsoft Teams, Google Meet/Chrome, Edge, Firefox,
   Discord, Webex, OBS, etc.) consultando el registro nativo de Windows
   (CapabilityAccessManager\\ConsentStore\\webcam).
3. Ceder/liberar automáticamente el hardware de la cámara ante un conflicto
   y reanudar el monitoreo en cuanto la videollamada finalice.
4. Priorizar el backend Windows Media Foundation (CAP_MSMF / Frame Server)
   y configurar resolución óptima (640x480) para minimizar consumo de RAM.
=============================================================================
"""

from dataclasses import dataclass
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

if sys.platform == "win32":
    import winreg


# Silenciar mensajes verbosos de OpenCV durante el escaneo de índices
def _silenciar_logs_opencv():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


_silenciar_logs_opencv()


# Mapa de nombres amigables para aplicaciones comunes de videollamadas
NOMBRES_APPS_CONOCIDAS: Dict[str, str] = {
    "zoom.exe": "Zoom",
    "teams.exe": "Microsoft Teams",
    "ms-teams.exe": "Microsoft Teams",
    "msteams": "Microsoft Teams",
    "chrome.exe": "Google Chrome (Meet/Web)",
    "msedge.exe": "Microsoft Edge (Meet/Teams Web)",
    "firefox.exe": "Mozilla Firefox (Videollamada)",
    "brave.exe": "Brave Browser (Videollamada)",
    "opera.exe": "Opera (Videollamada)",
    "discord.exe": "Discord",
    "skype.exe": "Skype",
    "skypeapp": "Skype",
    "ciscocollabhost.exe": "Cisco Webex",
    "webexmta.exe": "Cisco Webex",
    "obs64.exe": "OBS Studio",
    "obs32.exe": "OBS Studio",
    "whatsapp": "WhatsApp",
    "telegram.exe": "Telegram",
    "slack.exe": "Slack",
    "windowscamera": "Cámara de Windows",
}

# Procesos propios que no deben considerarse conflicto externo
PROCESOS_PROPIOS = {
    "python.exe",
    "pythonw.exe",
    "python3.exe",
    "python3.10.exe",
    "python3.11.exe",
    "python3.12.exe",
    "python3.13.exe",
    "py.exe",
}


@dataclass
class InfoCamara:
    indice: int
    nombre: str
    ancho: int
    alto: int
    backend: str

    def etiqueta(self) -> str:
        return f"Cámara #{self.indice}: {self.nombre} ({self.ancho}x{self.alto})"


def _obtener_nombres_camaras_windows() -> List[str]:
    """
    Intenta consultar los nombres amigables de las cámaras conectadas en Windows
    sin requerir dependencias de terceros. Devuelve lista vacía si no es posible.
    """
    if sys.platform != "win32":
        return []
    try:
        cmd = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-PnpDevice -Class Camera,Image -Status OK -ErrorAction SilentlyContinue "
            "| Select-Object -ExpandProperty FriendlyName",
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2.5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if res.returncode == 0 and res.stdout.strip():
            nombres = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            return nombres
    except Exception:
        pass
    return []


def _formatear_nombre_app_registro(clave_sub: str) -> Tuple[str, bool]:
    """
    Convierte una subclave del registro de Windows en un nombre legible y determina
    si corresponde a nuestro propio intérprete Python.
    """
    limpio = clave_sub.replace("#", "\\")
    base = os.path.basename(limpio) if "\\" in limpio else limpio
    base_lower = base.lower()

    # Verificar si es nuestro propio intérprete Python
    if base_lower in PROCESOS_PROPIOS or (
        base_lower.startswith("python") and base_lower.endswith(".exe")
    ):
        return base, True

    # Buscar coincidencia amigable
    for clave, nombre_amigable in NOMBRES_APPS_CONOCIDAS.items():
        if clave in base_lower or clave in limpio.lower():
            return nombre_amigable, False

    # Limpiar nombres de paquetes UWP (ej. Microsoft.WindowsCamera_8wekyb3d8bbwe)
    if "_" in base and "." in base:
        base = base.split("_")[0].split(".")[-1]
    return base, False


def detectar_apps_usando_camara() -> List[str]:
    """
    Consulta el registro nativo de Windows (CapabilityAccessManager) para detectar
    qué aplicaciones externas están usando activamente la cámara en este instante.

    En Windows 10/11, una aplicación está usando la cámara actualmente cuando:
        LastUsedTimeStart > 0  AND  LastUsedTimeStop == 0
    """
    if sys.platform != "win32":
        return []

    apps_activas: List[str] = []
    rutas_registro = [
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam",
        r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged",
    ]

    for ruta in rutas_registro:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, ruta) as clave_base:
                i = 0
                while True:
                    try:
                        nombre_sub = winreg.EnumKey(clave_base, i)
                        i += 1
                    except OSError:
                        break

                    if nombre_sub == "NonPackaged":
                        continue

                    try:
                        with winreg.OpenKey(clave_base, nombre_sub) as sub:
                            inicio, _ = winreg.QueryValueEx(sub, "LastUsedTimeStart")
                            fin, _ = winreg.QueryValueEx(sub, "LastUsedTimeStop")
                            if int(inicio) > 0 and int(fin) == 0:
                                nombre_app, es_propio = _formatear_nombre_app_registro(nombre_sub)
                                if not es_propio and nombre_app not in apps_activas:
                                    apps_activas.append(nombre_app)
                    except OSError:
                        continue
        except OSError:
            continue

    return apps_activas


def detectar_camaras_disponibles(max_indices: int = 6) -> List[InfoCamara]:
    """
    Escanea los índices 0..max_indices-1 para descubrir todas las cámaras físicas
    o virtuales utilizables en el sistema.
    """
    _silenciar_logs_opencv()
    nombres_sistema = _obtener_nombres_camaras_windows()
    encontradas: List[InfoCamara] = []

    backends = (
        [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MediaFoundation")]
        if sys.platform == "win32"
        else [(cv2.CAP_ANY, "Default")]
    )

    for idx in range(max_indices):
        cam_valida = None
        for backend_id, backend_nombre in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(idx, backend_id)
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        alto, ancho = frame.shape[:2]
                        nombre = (
                            nombres_sistema[idx]
                            if idx < len(nombres_sistema)
                            else f"Dispositivo de Video {idx}"
                        )
                        cam_valida = InfoCamara(
                            indice=idx,
                            nombre=nombre,
                            ancho=ancho,
                            alto=alto,
                            backend=backend_nombre,
                        )
                        break
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()

        if cam_valida is not None:
            encontradas.append(cam_valida)

    return encontradas


class GestorCamara:
    """
    Gestiona el ciclo de vida de la cámara con:
    - Soporte multi-cámara (selección inicial y cambio en caliente).
    - Manejo automático de conflictos con videollamadas (ceder hardware + reanudar).
    - Optimización de resolución y buffer para mantener bajo consumo de RAM.
    """

    ESTADO_INACTIVA = "INACTIVA"
    ESTADO_ACTIVA = "ACTIVA"
    ESTADO_PAUSADA_CONFLICTO = "PAUSADA_POR_CONFLICTO"
    ESTADO_PAUSADA_MANUAL = "PAUSADA_MANUAL"
    ESTADO_SIN_CAMARA = "SIN_CAMARA"

    def __init__(
        self,
        indice_camara: int = 0,
        camaras_disponibles: Optional[List[InfoCamara]] = None,
        anti_conflicto: bool = True,
        ancho_objetivo: int = 640,
        alto_objetivo: int = 480,
        intervalo_verificacion_seg: float = 1.5,
        detector_apps_externas: Optional[Callable[[], List[str]]] = None,
    ):
        self.indice_actual = indice_camara
        self.camaras_disponibles: List[InfoCamara] = camaras_disponibles or []
        self.anti_conflicto = anti_conflicto
        self.ancho_objetivo = ancho_objetivo
        self.alto_objetivo = alto_objetivo
        self.intervalo_verificacion_seg = intervalo_verificacion_seg
        self._detector_apps = detector_apps_externas or detectar_apps_usando_camara

        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_activo: str = "N/A"
        self.estado: str = self.ESTADO_INACTIVA
        self.app_en_conflicto: str = ""
        self.ultimo_chequeo_conflicto: float = 0.0
        self.ultimo_intento_reconexion: float = 0.0
        self.fallos_lectura_consecutivos: int = 0
        self.frames_negros_consecutivos: int = 0

        # Telemetría de conflictos para reporte de validación
        self.eventos_conflicto: List[Dict[str, object]] = []
        self.total_conflictos_resueltos: int = 0

    @property
    def esta_abierta(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def obtener_nombre_camara_actual(self) -> str:
        for cam in self.camaras_disponibles:
            if cam.indice == self.indice_actual:
                return f"CAM #{cam.indice} ({cam.nombre[:22]})"
        return f"CAM #{self.indice_actual}"

    def _liberar_hardware(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _registrar_evento_conflicto(self, tipo: str, detalle: str):
        self.eventos_conflicto.append(
            {
                "timestamp": round(time.time(), 2),
                "tipo": tipo,
                "detalle": detalle,
                "camara_indice": self.indice_actual,
            }
        )

    def verificar_conflicto_externo(self, forzar: bool = False) -> Tuple[bool, List[str]]:
        """
        Verifica si alguna aplicación de videollamada está usando la cámara.
        """
        if not self.anti_conflicto:
            return False, []

        ahora = time.time()
        if not forzar and (ahora - self.ultimo_chequeo_conflicto) < self.intervalo_verificacion_seg:
            return (self.estado == self.ESTADO_PAUSADA_CONFLICTO, [self.app_en_conflicto] if self.app_en_conflicto else [])

        self.ultimo_chequeo_conflicto = ahora
        apps = self._detector_apps()
        return (len(apps) > 0), apps

    def abrir(self, indice: Optional[int] = None) -> bool:
        """
        Abre la cámara indicada priorizando CAP_MSMF (compatible con Windows
        Camera Frame Server) y aplicando resolución óptima de bajo consumo RAM.
        """
        if indice is not None:
            self.indice_actual = indice

        self._liberar_hardware()

        # Antes de abrir, verificar si ya hay una videollamada activa
        hay_conflicto, apps = self.verificar_conflicto_externo(forzar=True)
        if hay_conflicto:
            self.app_en_conflicto = ", ".join(apps)
            self.estado = self.ESTADO_PAUSADA_CONFLICTO
            self.total_conflictos_resueltos += 1
            self._registrar_evento_conflicto(
                "CONFLICTO_AL_INICIAR",
                f"Cámara en uso por: {self.app_en_conflicto}. Monitoreo en espera sin bloquear.",
            )
            return False

        # En Windows priorizamos CAP_MSMF (Frame Server / multi-cliente) con fallback a CAP_DSHOW
        backends = (
            [(cv2.CAP_MSMF, "MediaFoundation"), (cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "Auto")]
            if sys.platform == "win32"
            else [(cv2.CAP_ANY, "Auto")]
        )

        for backend_id, nombre_backend in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(self.indice_actual, backend_id)
                if cap is not None and cap.isOpened():
                    # Configuración optimizada para reducir consumo de RAM y latencia
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho_objetivo)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto_objetivo)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.cap = cap
                        self.backend_activo = nombre_backend
                        self.estado = self.ESTADO_ACTIVA
                        self.app_en_conflicto = ""
                        self.fallos_lectura_consecutivos = 0
                        self.frames_negros_consecutivos = 0
                        return True
            except Exception:
                pass
            if cap is not None:
                cap.release()

        self.estado = self.ESTADO_SIN_CAMARA
        return False

    def leer_frame(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Lee un fotograma verificando conflictos con videollamadas.
        Retorna: (exito, frame_o_None, estado_actual)
        """
        if self.estado == self.ESTADO_PAUSADA_MANUAL:
            return False, None, self.estado

        # 1. Revisar periódicamente si otra app inició videollamada
        hay_conflicto, apps = self.verificar_conflicto_externo(forzar=False)
        if hay_conflicto:
            app_str = ", ".join(apps)
            if self.estado != self.ESTADO_PAUSADA_CONFLICTO:
                # Liberar inmediatamente el hardware para no bloquear la videollamada
                self._liberar_hardware()
                self.app_en_conflicto = app_str
                self.estado = self.ESTADO_PAUSADA_CONFLICTO
                self.total_conflictos_resueltos += 1
                self._registrar_evento_conflicto(
                    "CAMARA_CEDIDA_VIDEOLLAMADA",
                    f"Cámara liberada automáticamente para: {app_str}",
                )
                print(f"[ANTI-CONFLICTO] Cámara cedida automáticamente a: {app_str}")
            return False, None, self.estado

        # 2. Si estaba pausada por conflicto y la videollamada ya terminó -> reanudar
        if self.estado == self.ESTADO_PAUSADA_CONFLICTO:
            ahora = time.time()
            if (ahora - self.ultimo_intento_reconexion) >= 2.0:
                self.ultimo_intento_reconexion = ahora
                if self.abrir(self.indice_actual):
                    self._registrar_evento_conflicto(
                        "CAMARA_RECUPERADA",
                        "Videollamada finalizada; monitoreo reanudado automáticamente.",
                    )
                    print("[ANTI-CONFLICTO] Videollamada finalizada. Cámara recuperada automáticamente.")
                else:
                    return False, None, self.estado
            else:
                return False, None, self.estado

        # 3. Si la cámara no está abierta, intentar reconexión periódica segura
        if not self.esta_abierta:
            ahora = time.time()
            if (ahora - self.ultimo_intento_reconexion) >= 2.5:
                self.ultimo_intento_reconexion = ahora
                if not self.abrir(self.indice_actual):
                    return False, None, self.estado
            else:
                return False, None, self.estado

        # 4. Lectura efectiva de fotograma con detección de bloqueo exclusivo
        try:
            ret, frame = self.cap.read()
        except Exception:
            ret, frame = False, None

        if not ret or frame is None or frame.size == 0:
            self.fallos_lectura_consecutivos += 1
            if self.fallos_lectura_consecutivos >= 5:
                # Posible toma exclusiva de la cámara por otra app o desconexión
                self._liberar_hardware()
                self.app_en_conflicto = "Otra aplicación / Stream ocupado"
                self.estado = self.ESTADO_PAUSADA_CONFLICTO
                self.total_conflictos_resueltos += 1
                self._registrar_evento_conflicto(
                    "CONFLICTO_POR_PERDIDA_STREAM",
                    "Fallos consecutivos de lectura; cámara liberada para evitar bloqueo.",
                )
                print("[ANTI-CONFLICTO] Pérdida de stream detectada. Cámara liberada; reintentando al desocuparse...")
            return False, None, self.estado

        self.fallos_lectura_consecutivos = 0

        # Detección ligera de frames totalmente negros (en algunos drivers indica cámara tomada)
        if int(frame.max()) == 0:
            self.frames_negros_consecutivos += 1
            if self.frames_negros_consecutivos >= 25:
                # Verificar explícitamente en el registro si hay conflicto
                hay_conf, apps_conf = self.verificar_conflicto_externo(forzar=True)
                if hay_conf:
                    self._liberar_hardware()
                    self.app_en_conflicto = ", ".join(apps_conf)
                    self.estado = self.ESTADO_PAUSADA_CONFLICTO
                    self.total_conflictos_resueltos += 1
                    self._registrar_evento_conflicto(
                        "CONFLICTO_FRAME_NEGRO",
                        f"Frames negros detectados con app activa: {self.app_en_conflicto}",
                    )
                    return False, None, self.estado
        else:
            self.frames_negros_consecutivos = 0

        return True, frame, self.estado

    def cambiar_camara(self, nuevo_indice: int) -> bool:
        """Cambia en caliente a otra cámara física/virtual."""
        print(f"[CÁMARA] Cambiando a cámara #{nuevo_indice}...")
        return self.abrir(nuevo_indice)

    def siguiente_camara(self) -> bool:
        """Cicla a la siguiente cámara disponible (re-escaneando si solo había una)."""
        if len(self.camaras_disponibles) <= 1:
            self._liberar_hardware()
            self.camaras_disponibles = detectar_camaras_disponibles()

        if not self.camaras_disponibles:
            self.estado = self.ESTADO_SIN_CAMARA
            return False

        indices = [c.indice for c in self.camaras_disponibles]
        if self.indice_actual in indices:
            pos = (indices.index(self.indice_actual) + 1) % len(indices)
            siguiente = indices[pos]
        else:
            siguiente = indices[0]

        return self.cambiar_camara(siguiente)

    def pausar_manual(self):
        """Libera la cámara manualmente (ideal antes de entrar a una videollamada)."""
        self._liberar_hardware()
        self.estado = self.ESTADO_PAUSADA_MANUAL
        self.app_en_conflicto = ""
        print("[CÁMARA] Monitoreo pausado y cámara liberada manualmente.")

    def reanudar_manual(self) -> bool:
        """Reanuda el monitoreo tras una pausa manual."""
        self.estado = self.ESTADO_INACTIVA
        return self.abrir(self.indice_actual)

    def alternar_pausa_manual(self) -> bool:
        if self.estado == self.ESTADO_PAUSADA_MANUAL:
            return self.reanudar_manual()
        self.pausar_manual()
        return False

    def cerrar(self):
        """Libera definitivamente el recurso de cámara."""
        self._liberar_hardware()
        self.estado = self.ESTADO_INACTIVA
