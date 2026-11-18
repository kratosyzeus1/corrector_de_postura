"""
=============================================================================
PROYECTO: Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
INSTITUCIÓN: Politécnico Grancolombiano
AUTORES: Valery Sofía Meléndez Gallego, Juan Guillermo Ramírez Vásquez, Samuel Felipe Marín Lara
VERSIÓN: 3.0 - Correcciones Docentes Integradas:
  1. Selección multi-cámara (inicio + cambio en caliente con tecla [V]).
  2. Manejo automático de conflictos de cámara con videollamadas (Windows).
  3. Interfaz no disruptiva en 2 vistas:
     - Vista 1 (Compacta): Avatar biomecánico sin verse en cámara + paneles pequeños.
     - Vista 2 (Overlay / Background): Pantalla 100% normal + solo notificaciones flotantes.
  4. Validación y optimización real del consumo de recursos (RAM y CPU).
=============================================================================
DELIMITACIÓN TÉCNICA Y ÉTICA:
El sistema delimita estrictamente el rol de la Inteligencia Artificial a la
prevención ergonómica ocupacional y a la modulación de pausas activas.
No diagnostica patologías clínicas ni prescribe tratamientos médicos.
=============================================================================
"""

import argparse
from collections import deque
from datetime import datetime
import gc
import math
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:
    import joblib
    TIENE_JOBLIB = True
except ImportError:
    TIENE_JOBLIB = False

from gestor_camara import (
    GestorCamara,
    detectar_camaras_disponibles,
)
from interfaz_overlay import (
    VentanaOverlayNotificaciones,
    mostrar_selector_inicio,
)
from monitor_recursos import MonitorRecursos


def _rectangulo_translucido(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float = 0.80,
) -> None:
    """
    Aplica sombreado translúcido únicamente sobre la región ROI (sin copiar todo el frame),
    reduciendo en >85% las asignaciones temporales de memoria RAM por fotograma.
    """
    alto, ancho = frame.shape[:2]
    rx1 = max(0, min(ancho, x1))
    ry1 = max(0, min(alto, y1))
    rx2 = max(0, min(ancho, x2))
    ry2 = max(0, min(alto, y2))
    if rx2 <= rx1 or ry2 <= ry1:
        return

    roi = frame[ry1:ry2, rx1:rx2]
    capa = np.empty_like(roi)
    capa[:] = color
    cv2.addWeighted(capa, alpha, roi, 1.0 - alpha, 0, dst=roi)


class AsistenteErgonomico:
    def __init__(
        self,
        model_complexity: int = 0,
        mostrar_video_camara: bool = True,
        ancho_vista: int = 720,
        alto_vista: int = 460,
    ):
        # Inicialización de MediaPipe Pose (complexity=0 Lite optimiza RAM y CPU)
        self.mp_pose = mp.solutions.pose
        self.model_complexity = model_complexity
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Configuración visual: Vista 1 muestra la cámara en vivo con HUD compacto ([F] alterna a avatar)
        self.mostrar_video_camara = mostrar_video_camara
        self.ancho_vista = ancho_vista
        self.alto_vista = alto_vista
        # Doble buffer (evita condiciones de carrera / parpadeo entre hilos al dibujar)
        self._buffers_canvas = [
            np.zeros((alto_vista, ancho_vista, 3), dtype=np.uint8),
            np.zeros((alto_vista, ancho_vista, 3), dtype=np.uint8),
        ]
        self._idx_buffer = 0

        # Variables biomecánicas, suavizado EMA (evita parpadeo de lecturas) y calibración
        self.calibrado = False
        self.auto_calibrado = False
        self._auto_calib_muestras = []
        self.vista_corporal = "FRONTAL"  # Detección automática: "FRONTAL" o "PERFIL"
        self.baseline_cervical = 15.0
        self.baseline_hombros = 0.0
        self.baseline_distancia = 0.36
        self.baseline_ratio_cuello = 0.42
        self.baseline_ratio_nariz = 0.44
        self.baseline_pitch_nariz = 0.02
        self.baseline_3d_sagital = 20.0
        self._ema_inicializado = False
        self._ema_cervical = 14.0
        self._ema_hombros = 0.0
        self._ema_distancia = 0.36
        self._ultimo_color_estado = (90, 210, 80)
        self._frames_sin_deteccion = 0
        self._ultimos_landmarks = None

        # Umbrales ergonómicos (ISO 9241-5 / OSHA)
        self.UMBRAL_CUELLO_MAX = 28.0
        self.UMBRAL_HOMBROS_MAX = 7.0
        self.TIEMPO_ALERTA_SEGUNDOS = 5.0

        # Estado del sistema y temporizadores
        self.tiempo_mala_postura_inicio: Optional[float] = None
        self.segundos_mala_postura = 0.0
        self.alerta_activa = False
        self.mensaje_alerta = ""
        self.tipo_diagnostico = "POSTURA OPTIMA"
        self.score_ergonomico = 100
        self.ultimo_angulo_cervical = 0.0
        self.ultimo_desnivel_hombros = 0.0
        self.ultima_distancia_relativa = 0.0

        # Estadísticas de la sesión
        self.tiempo_inicio_sesion = time.time()
        self.total_frames = 0
        self.frames_correctos = 0
        self.frames_riesgo = 0
        self.total_alertas_disparadas = 0

        # Buffers acotados (deque maxlen=300 evita fugas de memoria en sesiones largas)
        self.historial_cervical: deque = deque(maxlen=300)
        self.historial_hombros: deque = deque(maxlen=300)
        self.historial_score: deque = deque(maxlen=300)

        # Modelo K-Means
        self.modelo_kmeans_cargado = False
        self.kmeans_scaler = None
        self.kmeans_model = None
        self.kmeans_names_map: Dict[int, str] = {}
        self.cluster_actual_fatiga = "Baja fatiga"
        self.cluster_color = (90, 210, 80)  # Verde suave
        self.accion_ergonomica = "Ritmo optimo; pausa a los 45 min."
        self.cargar_modelo_kmeans()

        # Modo de ejecución y telemetría de recursos
        self.modo_simulacion = False
        self.sim_phase = 0.0
        self.ram_actual_mb = 0.0
        self.estado_ram = "OPTIMO"
        self.camara_label = "CAM #0"

    def cerrar(self):
        """Libera recursos de MediaPipe y memoria asociada."""
        try:
            if self.pose is not None:
                self.pose.close()
                self.pose = None
        except Exception:
            pass
        gc.collect()

    def cargar_modelo_kmeans(self):
        """Carga el modelo de K-Means entrenado en el archivo pickle o JSON."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pkl_path = os.path.join(base_dir, "modelo_kmeans_fatiga.pkl")
        if TIENE_JOBLIB and os.path.exists(pkl_path):
            try:
                data = joblib.load(pkl_path)
                self.kmeans_model = data["kmeans"]
                self.kmeans_scaler = data["scaler"]
                self.kmeans_names_map = data["cluster_names_map"]
                self.modelo_kmeans_cargado = True
                print("[K-MEANS] Modelo cargado exitosamente desde:", os.path.basename(pkl_path))
                return
            except Exception as e:
                print(f"[WARN] No se pudo cargar {pkl_path}: {e}")

        json_path = os.path.join(base_dir, "modelo_kmeans_fatiga.json")
        if os.path.exists(json_path):
            print("[K-MEANS] Usando perfiles desde:", os.path.basename(json_path))
            self.modelo_kmeans_cargado = True

    def evaluar_bloque_con_kmeans(self):
        """Clasifica el bloque reciente en Baja fatiga, Fatiga moderada o Alta fatiga."""
        if self.total_frames == 0:
            return

        pct_riesgo = (self.frames_riesgo / max(1, self.total_frames)) * 100.0
        cerv_rec = list(self.historial_cervical)[-150:]
        homb_rec = list(self.historial_hombros)[-150:]
        score_rec = list(self.historial_score)[-150:]

        media_cervical = float(np.mean(cerv_rec)) if cerv_rec else 15.0
        media_hombros = float(np.mean(homb_rec)) if homb_rec else 2.0
        media_score = float(np.mean(score_rec)) if score_rec else 90.0
        sedentario_min = (time.time() - self.tiempo_inicio_sesion) / 60.0

        if self.modelo_kmeans_cargado and self.kmeans_model is not None:
            vec = np.array(
                [
                    [
                        pct_riesgo,
                        media_cervical,
                        media_hombros,
                        self.total_alertas_disparadas,
                        sedentario_min,
                        media_score,
                    ]
                ]
            )
            vec_scaled = self.kmeans_scaler.transform(vec)
            raw_cluster = int(self.kmeans_model.predict(vec_scaled)[0])
            self.cluster_actual_fatiga = self.kmeans_names_map.get(raw_cluster, "Indeterminado")
        else:
            if pct_riesgo < 25.0 and media_cervical < 22.0:
                self.cluster_actual_fatiga = "Baja fatiga"
            elif pct_riesgo < 55.0 and media_cervical < 29.0:
                self.cluster_actual_fatiga = "Fatiga moderada"
            else:
                self.cluster_actual_fatiga = "Alta fatiga"

        if self.cluster_actual_fatiga == "Baja fatiga":
            self.cluster_color = (90, 210, 80)
            self.TIEMPO_ALERTA_SEGUNDOS = 7.0
            self.accion_ergonomica = "Hidratacion a los 45 min."
        elif self.cluster_actual_fatiga == "Fatiga moderada":
            self.cluster_color = (40, 185, 245)
            self.TIEMPO_ALERTA_SEGUNDOS = 4.5
            self.accion_ergonomica = "Micro-pausa 30s (trapecios)."
        else:
            self.cluster_color = (70, 80, 245)
            self.TIEMPO_ALERTA_SEGUNDOS = 3.5
            self.accion_ergonomica = "Pausa activa 3-5 min + 20-20-20."

    def calcular_angulo(self, p1, p2, aspect_ratio: float = 1.0) -> float:
        dx = p2[0] - p1[0]
        dy = (p2[1] - p1[1]) * aspect_ratio
        return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

    def calcular_inclinacion_hombros(self, hombro_izq, hombro_der, aspect_ratio: float = 1.0) -> float:
        dx = abs(hombro_der[0] - hombro_izq[0])
        dy = abs(hombro_der[1] - hombro_izq[1]) * aspect_ratio
        return math.degrees(math.atan2(dy, dx + 1e-6))

    def extraer_metricas_postura(
        self,
        landmarks,
        world_landmarks=None,
        aspect_ratio: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Extrae métricas biomecánicas completas tanto en vista FRONTAL (frente a la pantalla)
        como en vista PERFIL (lateral), detectando automáticamente la orientación corporal.
        """
        lm = self.mp_pose.PoseLandmark
        nariz = [landmarks[lm.NOSE.value].x, landmarks[lm.NOSE.value].y]
        oreja_izq = [landmarks[lm.LEFT_EAR.value].x, landmarks[lm.LEFT_EAR.value].y]
        oreja_der = [landmarks[lm.RIGHT_EAR.value].x, landmarks[lm.RIGHT_EAR.value].y]
        hombro_izq = [landmarks[lm.LEFT_SHOULDER.value].x, landmarks[lm.LEFT_SHOULDER.value].y]
        hombro_der = [landmarks[lm.RIGHT_SHOULDER.value].x, landmarks[lm.RIGHT_SHOULDER.value].y]

        p_orejas = [(oreja_izq[0] + oreja_der[0]) / 2.0, (oreja_izq[1] + oreja_der[1]) / 2.0]
        p_hombros = [(hombro_izq[0] + hombro_der[0]) / 2.0, (hombro_izq[1] + hombro_der[1]) / 2.0]

        ancho_hombros_x = abs(hombro_der[0] - hombro_izq[0])
        dist_hombros = math.hypot(
            hombro_der[0] - hombro_izq[0],
            (hombro_der[1] - hombro_izq[1]) * aspect_ratio,
        )

        # Detección automática de orientación: FRONTAL vs PERFIL
        es_perfil = ancho_hombros_x < 0.14
        ang_3d_sagital: Optional[float] = None
        if world_landmarks is not None:
            w_h_izq = world_landmarks[lm.LEFT_SHOULDER.value]
            w_h_der = world_landmarks[lm.RIGHT_SHOULDER.value]
            w_o_izq = world_landmarks[lm.LEFT_EAR.value]
            w_o_der = world_landmarks[lm.RIGHT_EAR.value]

            if abs(w_h_der.x - w_h_izq.x) < 0.17 and abs(w_h_der.z - w_h_izq.z) > 0.14:
                es_perfil = True

            w_h_y = (w_h_izq.y + w_h_der.y) / 2.0
            w_h_z = (w_h_izq.z + w_h_der.z) / 2.0
            w_o_y = (w_o_izq.y + w_o_der.y) / 2.0
            w_o_z = (w_o_izq.z + w_o_der.z) / 2.0
            dy_3d = max(0.05, w_h_y - w_o_y)
            dz_3d = max(0.0, w_h_z - w_o_z)
            ang_3d_sagital = math.degrees(math.atan2(dz_3d, dy_3d))

        vista_corporal = "PERFIL" if es_perfil else "FRONTAL"
        ang_2d_plano = self.calcular_angulo(p_hombros, p_orejas, aspect_ratio)
        desnivel_hombros = 0.0 if es_perfil else self.calcular_inclinacion_hombros(hombro_izq, hombro_der, aspect_ratio)

        escala = max(0.08, dist_hombros)
        ratio_cuello = ((p_hombros[1] - p_orejas[1]) * aspect_ratio) / escala
        ratio_nariz = ((p_hombros[1] - nariz[1]) * aspect_ratio) / escala
        pitch_nariz = ((nariz[1] - p_orejas[1]) * aspect_ratio) / escala

        return {
            "vista_corporal": vista_corporal,
            "ang_2d_plano": ang_2d_plano,
            "desnivel_hombros": desnivel_hombros,
            "dist_hombros": dist_hombros,
            "ratio_cuello": ratio_cuello,
            "ratio_nariz": ratio_nariz,
            "pitch_nariz": pitch_nariz,
            "ang_3d_sagital": ang_3d_sagital,
        }

    def evaluar_metricas_postura(self, metricas: Dict[str, Any]) -> Tuple[float, float, float, int, list]:
        """
        Evalúa las métricas biomecánicas aplicando el modelo dual:
        - En PERFIL: ángulo sagital 2D oreja-hombro.
        - En FRONTAL: fusión de compresión vertical cuello/nariz, inclinación de cabeza
          hacia abajo (pitch), profundidad 3D sagital (eje Z) e inclinación lateral.
        """
        vista = metricas["vista_corporal"]
        self.vista_corporal = vista
        homb_raw = float(metricas["desnivel_hombros"])
        dist_raw = float(metricas["dist_hombros"])

        # Auto-calibración inicial silenciosa (primeros 15 frames frontales) si el usuario no presionó [C]
        if vista == "FRONTAL" and not self.calibrado and not self.auto_calibrado:
            self._auto_calib_muestras.append(metricas)
            if len(self._auto_calib_muestras) >= 15:
                self.baseline_ratio_cuello = max(0.20, float(np.median([m["ratio_cuello"] for m in self._auto_calib_muestras])))
                self.baseline_ratio_nariz = max(0.20, float(np.median([m["ratio_nariz"] for m in self._auto_calib_muestras])))
                self.baseline_pitch_nariz = float(np.median([m["pitch_nariz"] for m in self._auto_calib_muestras]))
                self.baseline_distancia = max(0.18, float(np.median([m["dist_hombros"] for m in self._auto_calib_muestras])))
                self.baseline_hombros = min(4.0, float(np.median([m["desnivel_hombros"] for m in self._auto_calib_muestras])))
                vals_3d = [m["ang_3d_sagital"] for m in self._auto_calib_muestras if m["ang_3d_sagital"] is not None]
                if vals_3d:
                    self.baseline_3d_sagital = float(np.median(vals_3d))
                self.auto_calibrado = True

        if vista == "PERFIL":
            ang_raw = float(metricas["ang_2d_plano"])
        else:
            # 1. Compresión vertical cuello-hombros y nariz-hombros (encorvamiento frontal)
            caida_cuello = max(
                0.0,
                (self.baseline_ratio_cuello - float(metricas["ratio_cuello"])) / max(0.15, self.baseline_ratio_cuello),
            )
            caida_nariz = max(
                0.0,
                (self.baseline_ratio_nariz - float(metricas["ratio_nariz"])) / max(0.15, self.baseline_ratio_nariz),
            )
            caida = max(caida_cuello, caida_nariz)
            ang_comp = 14.0 + max(0.0, caida - 0.05) * 95.0

            # 2. Inclinación de cabeza hacia abajo (pitch nariz respecto a orejas - text neck)
            delta_pitch = max(0.0, float(metricas["pitch_nariz"]) - self.baseline_pitch_nariz)
            ang_pitch = 14.0 + max(0.0, delta_pitch - 0.03) * 160.0

            # 3. Profundidad 3D sagital (adelantamiento de cabeza en eje Z)
            ang_3d_val = metricas.get("ang_3d_sagital")
            if ang_3d_val is not None:
                delta_3d = max(0.0, float(ang_3d_val) - self.baseline_3d_sagital)
                ang_3d = 14.0 + max(0.0, delta_3d - 3.0) * 1.4
            else:
                ang_3d = 14.0

            # 4. Inclinación lateral excesiva de la cabeza
            ang_lat = 14.0 + max(0.0, float(metricas["ang_2d_plano"]) - 8.0) * 1.3

            ang_raw = min(55.0, max(ang_comp, ang_pitch, ang_3d, ang_lat))

        # Suavizado exponencial (EMA) para estabilidad visual
        if not self._ema_inicializado:
            self._ema_cervical = ang_raw
            self._ema_hombros = homb_raw
            self._ema_distancia = dist_raw
            self._ema_inicializado = True
        else:
            alpha_ema = 0.28
            self._ema_cervical = alpha_ema * ang_raw + (1.0 - alpha_ema) * self._ema_cervical
            self._ema_hombros = alpha_ema * homb_raw + (1.0 - alpha_ema) * self._ema_hombros
            self._ema_distancia = alpha_ema * dist_raw + (1.0 - alpha_ema) * self._ema_distancia

        angulo_cervical = self._ema_cervical
        desnivel_hombros = self._ema_hombros
        distancia_relativa = self._ema_distancia

        umbral_cerv = (
            (self.baseline_cervical + 12.0)
            if (vista == "PERFIL" and self.calibrado)
            else self.UMBRAL_CUELLO_MAX
        )
        es_cabeza_adelantada = angulo_cervical > umbral_cerv
        es_hombros_desalineados = (vista == "FRONTAL") and (
            desnivel_hombros > (self.baseline_hombros + self.UMBRAL_HOMBROS_MAX)
        )
        es_muy_cerca = (vista == "FRONTAL") and (distancia_relativa > (self.baseline_distancia * 1.25))

        penalizacion = 0
        motivos = []
        if es_cabeza_adelantada:
            penalizacion += 40
            motivos.append("Cervical adelantada")
        if es_hombros_desalineados:
            penalizacion += 30
            motivos.append("Hombros desnivelados")
        if es_muy_cerca:
            penalizacion += 20
            motivos.append("Muy cerca a pantalla")

        return angulo_cervical, desnivel_hombros, distancia_relativa, penalizacion, motivos

    def _obtener_canvas_libre(self) -> np.ndarray:
        """Alterna entre dos buffers internos para eliminar por completo el parpadeo entre hilos."""
        self._idx_buffer = 1 - self._idx_buffer
        return self._buffers_canvas[self._idx_buffer]

    def _preparar_fondo_vista_compacta(self, frame: Optional[np.ndarray]) -> np.ndarray:
        """
        Prepara el lienzo de la Vista 1:
        - Por defecto (mostrar_video_camara=True): imagen nítida en vivo de la cámara
          (redimensionada al buffer libre de Vista 1).
        - Si el usuario presiona [F] (mostrar_video_camara=False): fondo oscuro limpio
          para el avatar biomecánico sin verse en cámara.
        """
        canvas = self._obtener_canvas_libre()
        if self.mostrar_video_camara and frame is not None:
            cv2.resize(frame, (self.ancho_vista, self.alto_vista), dst=canvas, interpolation=cv2.INTER_LINEAR)
            return canvas

        # Fondo oscuro minimalista (modo avatar opcional con tecla [F])
        canvas[:] = (28, 22, 18)  # Slate oscuro (BGR)

        # Rejilla ergonómica muy sutil en el centro
        cx = self.ancho_vista // 2
        cy = self.alto_vista // 2
        cv2.line(canvas, (cx, 58), (cx, self.alto_vista - 38), (48, 42, 36), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx - 120, cy + 15), (cx + 120, cy + 15), (42, 38, 32), 1, cv2.LINE_AA)
        return canvas

    def _dibujar_esqueleto_sobre_camara(
        self,
        canvas: np.ndarray,
        landmarks,
        color_estado: Tuple[int, int, int],
    ) -> None:
        """Dibuja un esqueleto ergonómico ligero y limpio sobre el video real sin tapar el rostro."""
        alto, ancho = canvas.shape[:2]
        lm = self.mp_pose.PoseLandmark

        def pt(idx_lm):
            return (
                int(landmarks[idx_lm.value].x * ancho),
                int(landmarks[idx_lm.value].y * alto),
            )

        h_izq = pt(lm.LEFT_SHOULDER)
        h_der = pt(lm.RIGHT_SHOULDER)
        o_izq = pt(lm.LEFT_EAR)
        o_der = pt(lm.RIGHT_EAR)

        cv2.line(canvas, h_izq, h_der, color_estado, 2, cv2.LINE_AA)
        cv2.line(canvas, h_izq, o_izq, (235, 195, 70), 2, cv2.LINE_AA)
        cv2.line(canvas, h_der, o_der, (235, 195, 70), 2, cv2.LINE_AA)

        for punto in (h_izq, h_der, o_izq, o_der):
            cv2.circle(canvas, punto, 5, (245, 248, 252), -1, cv2.LINE_AA)
            cv2.circle(canvas, punto, 3, color_estado, -1, cv2.LINE_AA)

    def _dibujar_avatar_biomecanico(
        self,
        canvas: np.ndarray,
        landmarks,
        color_estado: Tuple[int, int, int],
        ang_cervical: float,
    ) -> None:
        """
        Dibuja un avatar postural minimalista y abstracto a partir de los landmarks reales.
        Permite entender la postura al instante sin verse en cámara.
        """
        alto, ancho = canvas.shape[:2]
        lm = self.mp_pose.PoseLandmark

        def pt(idx_lm):
            return (
                int(landmarks[idx_lm.value].x * ancho),
                int(landmarks[idx_lm.value].y * alto),
            )

        h_izq = pt(lm.LEFT_SHOULDER)
        h_der = pt(lm.RIGHT_SHOULDER)
        o_izq = pt(lm.LEFT_EAR)
        o_der = pt(lm.RIGHT_EAR)
        nariz = pt(lm.NOSE)

        centro_hombros = ((h_izq[0] + h_der[0]) // 2, (h_izq[1] + h_der[1]) // 2)
        centro_cabeza = ((o_izq[0] + o_der[0]) // 2, (o_izq[1] + o_der[1]) // 2)
        ancho_hombros = max(40, int(math.hypot(h_der[0] - h_izq[0], h_der[1] - h_izq[1])))
        radio_cabeza = max(18, min(42, int(ancho_hombros * 0.24)))

        # Línea vertical ideal de referencia desde el centro de hombros hacia arriba
        alto_guia = max(55, int(math.hypot(centro_cabeza[0] - centro_hombros[0], centro_cabeza[1] - centro_hombros[1])))
        pt_ideal = (centro_hombros[0], max(54, centro_hombros[1] - alto_guia))
        cv2.line(canvas, centro_hombros, pt_ideal, (85, 95, 110), 1, cv2.LINE_AA)

        # Torso estilizado (trapecio suave)
        base_torso_y = min(alto - 34, centro_hombros[1] + int(ancho_hombros * 0.65))
        torso_pts = np.array(
            [
                h_izq,
                h_der,
                (centro_hombros[0] + int(ancho_hombros * 0.36), base_torso_y),
                (centro_hombros[0] - int(ancho_hombros * 0.36), base_torso_y),
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(canvas, [torso_pts], (42, 36, 32))
        cv2.polylines(canvas, [torso_pts], True, (75, 85, 100), 1, cv2.LINE_AA)

        # Vector cervical (centro de hombros -> centro de cabeza)
        cv2.line(canvas, centro_hombros, centro_cabeza, color_estado, 3, cv2.LINE_AA)
        # Línea escapular (hombro izq <-> hombro der)
        cv2.line(canvas, h_izq, h_der, color_estado, 3, cv2.LINE_AA)

        # Cabeza abstracta estilizada
        cv2.circle(canvas, centro_cabeza, radio_cabeza, (38, 34, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, centro_cabeza, radio_cabeza, color_estado, 2, cv2.LINE_AA)
        # Punto discreto de orientación facial
        cv2.circle(canvas, nariz, 3, (220, 230, 240), -1, cv2.LINE_AA)

        # Articulaciones clave
        for punto in (h_izq, h_der, centro_hombros):
            cv2.circle(canvas, punto, 5, (235, 240, 245), -1, cv2.LINE_AA)
            cv2.circle(canvas, punto, 3, color_estado, -1, cv2.LINE_AA)

        # Etiqueta discreta del ángulo cerca del cuello
        cv2.putText(
            canvas,
            f"{ang_cervical:.0f} deg",
            (centro_cabeza[0] + radio_cabeza + 6, centro_cabeza[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (195, 205, 220),
            1,
            cv2.LINE_AA,
        )

    def procesar_frame(self, frame: np.ndarray, renderizar: bool = True) -> Optional[np.ndarray]:
        """
        Analiza un frame con MediaPipe BlazePose y actualiza el estado ergonómico.
        Cuando renderizar=False (Vista 2 / Background), omite todo el dibujo OpenCV
        para ahorrar CPU y RAM.
        """
        self.total_frames += 1

        # Preprocesamiento ligero en RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        diagnostico = "SIN DETECCION"
        color_estado = (140, 150, 165)
        angulo_cervical = 0.0
        desnivel_hombros = 0.0
        distancia_relativa = 0.0

        if results.pose_landmarks:
            self._frames_sin_deteccion = 0
            landmarks = results.pose_landmarks.landmark
            world_lm = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
            self._ultimos_landmarks = landmarks
            alto_f, ancho_f = frame.shape[:2]
            aspect_ratio = alto_f / max(1.0, float(ancho_f))

            metricas = self.extraer_metricas_postura(landmarks, world_lm, aspect_ratio)
            angulo_cervical, desnivel_hombros, distancia_relativa, penalizacion, motivos = (
                self.evaluar_metricas_postura(metricas)
            )

            self.score_ergonomico = max(10, 100 - penalizacion)
            self.historial_cervical.append(angulo_cervical)
            self.historial_hombros.append(desnivel_hombros)
            self.historial_score.append(self.score_ergonomico)

            if penalizacion == 0:
                diagnostico = "POSTURA OPTIMA"
                color_estado = (90, 210, 80)
                self.frames_correctos += 1
                self.tiempo_mala_postura_inicio = None
                self.segundos_mala_postura = 0.0
                self.alerta_activa = False
                self.mensaje_alerta = ""
            else:
                self.frames_riesgo += 1
                diagnostico = "RIESGO: " + " + ".join(motivos)
                color_estado = (70, 80, 245) if penalizacion >= 40 else (40, 185, 245)

                ahora = time.time()
                if self.tiempo_mala_postura_inicio is None:
                    self.tiempo_mala_postura_inicio = ahora
                self.segundos_mala_postura = ahora - self.tiempo_mala_postura_inicio

                if self.segundos_mala_postura >= self.TIEMPO_ALERTA_SEGUNDOS:
                    if not self.alerta_activa:
                        self.total_alertas_disparadas += 1
                    self.alerta_activa = True
                    self.mensaje_alerta = f"{motivos[0]} — Alinea espalda y mirada"

            self._ultimo_color_estado = color_estado
            if self.total_frames % 60 == 0:
                self.evaluar_bloque_con_kmeans()
        else:
            # Tolerancia de 6 frames ante micro-pérdidas de MediaPipe (evita parpadeo)
            self._frames_sin_deteccion += 1
            if self._frames_sin_deteccion <= 6 and self._ema_inicializado:
                diagnostico = self.tipo_diagnostico
                color_estado = self._ultimo_color_estado
                angulo_cervical = self._ema_cervical
                desnivel_hombros = self._ema_hombros
                distancia_relativa = self._ema_distancia
            else:
                diagnostico = "USUARIO NO DETECTADO"
                color_estado = (130, 140, 155)
                self.score_ergonomico = 50
                self.alerta_activa = False
                self._ultimos_landmarks = None

        self.ultimo_angulo_cervical = angulo_cervical
        self.ultimo_desnivel_hombros = desnivel_hombros
        self.ultima_distancia_relativa = distancia_relativa
        self.tipo_diagnostico = diagnostico

        if not renderizar:
            return None

        canvas = self._preparar_fondo_vista_compacta(frame)
        lm_dibujo = results.pose_landmarks.landmark if results.pose_landmarks else self._ultimos_landmarks
        if lm_dibujo is not None:
            if self.mostrar_video_camara:
                self._dibujar_esqueleto_sobre_camara(canvas, lm_dibujo, color_estado)
            else:
                self._dibujar_avatar_biomecanico(canvas, lm_dibujo, color_estado, angulo_cervical)
        elif not self.mostrar_video_camara:
            cv2.putText(
                canvas,
                "Sin persona frente a la camara",
                (self.ancho_vista // 2 - 115, self.alto_vista // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (145, 155, 170),
                1,
                cv2.LINE_AA,
            )

        self.renderizar_hud_compacto(
            canvas,
            diagnostico,
            color_estado,
            angulo_cervical,
            desnivel_hombros,
        )
        return canvas

    def renderizar_pantalla_pausa(self, motivo: str, detalle: str, es_conflicto: bool = True) -> np.ndarray:
        """Renderiza una vista limpia cuando la cámara fue cedida a una videollamada o pausada."""
        canvas = self._obtener_canvas_libre()
        canvas[:] = (28, 22, 18)
        color_acento = (235, 180, 45) if es_conflicto else (150, 160, 175)

        cx = self.ancho_vista // 2
        cy = self.alto_vista // 2

        # Tarjeta central discreta
        x1, y1, x2, y2 = cx - 220, cy - 68, cx + 220, cy + 68
        _rectangulo_translucido(canvas, x1, y1, x2, y2, (36, 30, 24), alpha=0.92)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_acento, 1)
        cv2.rectangle(canvas, (x1, y1), (x1 + 4, y2), color_acento, -1)

        cv2.putText(
            canvas,
            motivo,
            (x1 + 18, y1 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            detalle[:56],
            (x1 + 18, y1 + 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (185, 198, 215),
            1,
            cv2.LINE_AA,
        )
        sub = (
            "Se reanudara automaticamente al terminar la llamada."
            if es_conflicto
            else "Presiona [P] para reanudar el monitoreo."
        )
        cv2.putText(
            canvas,
            sub,
            (x1 + 18, y1 + 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (130, 150, 170),
            1,
            cv2.LINE_AA,
        )

        self.renderizar_hud_compacto(
            canvas,
            "PAUSA POR VIDEOLLAMADA" if es_conflicto else "MONITOREO PAUSADO",
            color_acento,
            0.0,
            0.0,
        )
        return canvas

    def renderizar_hud_compacto(
        self,
        frame: np.ndarray,
        diagnostico: str,
        color_estado: Tuple[int, int, int],
        ang_cervical: float,
        des_hombros: float,
    ) -> None:
        """
        Vista 1 mejorada: mensajes, paneles y notificaciones mucho más pequeños,
        limpios y proporcionados para no saturar la pantalla.
        """
        alto, ancho = frame.shape[:2]

        # 1. Barra superior delgada (38 px vs 80 px anterior)
        _rectangulo_translucido(frame, 10, 8, ancho - 10, 46, (22, 18, 15), alpha=0.82)
        cv2.rectangle(frame, (10, 8), (ancho - 10, 46), (55, 65, 80), 1)
        cv2.rectangle(frame, (10, 8), (14, 46), color_estado, -1)

        cv2.putText(
            frame,
            diagnostico[:40],
            (22, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color_estado,
            1,
            cv2.LINE_AA,
        )

        # Indicadores compactos a la derecha (Cámara, RAM y Score)
        info_mini = f"{self.camara_label[:14]} | RAM {self.ram_actual_mb:.0f}MB"
        cv2.putText(
            frame,
            info_mini,
            (ancho - 285, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (170, 185, 200),
            1,
            cv2.LINE_AA,
        )
        score_color = (90, 210, 80) if self.score_ergonomico >= 80 else (70, 80, 245)
        cv2.putText(
            frame,
            f"{self.score_ergonomico}%",
            (ancho - 65, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            score_color,
            2,
            cv2.LINE_AA,
        )

        # 2. Panel izquierdo compacto de telemetría (196x138 px)
        p_x1, p_y1, p_x2, p_y2 = 10, 56, 206, 194
        _rectangulo_translucido(frame, p_x1, p_y1, p_x2, p_y2, (22, 18, 15), alpha=0.78)
        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (55, 65, 80), 1)

        cv2.putText(frame, "TELEMETRIA", (p_x1 + 10, p_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 190, 60), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"Cervical: {ang_cervical:.1f} deg",
            (p_x1 + 10, p_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (225, 232, 240),
            1,
            cv2.LINE_AA,
        )
        # Barra fina cervical
        pct_c = min(1.0, ang_cervical / 45.0)
        cv2.rectangle(frame, (p_x1 + 10, p_y1 + 46), (p_x2 - 12, p_y1 + 50), (50, 55, 65), -1)
        bar_c = (90, 210, 80) if ang_cervical < 28 else (70, 80, 245)
        cv2.rectangle(frame, (p_x1 + 10, p_y1 + 46), (p_x1 + 10 + int((p_x2 - p_x1 - 22) * pct_c), p_y1 + 50), bar_c, -1)

        cv2.putText(
            frame,
            f"Hombros: {des_hombros:.1f} deg",
            (p_x1 + 10, p_y1 + 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (225, 232, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Riesgo: {self.segundos_mala_postura:.1f}s / {self.TIEMPO_ALERTA_SEGUNDOS:.0f}s",
            (p_x1 + 10, p_y1 + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (205, 215, 225),
            1,
            cv2.LINE_AA,
        )
        calib_txt = (
            f"Calibrado [{self.vista_corporal}]"
            if self.calibrado
            else (f"Auto [{self.vista_corporal}] · [C] fijar" if self.auto_calibrado else f"Vista: {self.vista_corporal} · [C] calib")
        )
        calib_col = (90, 210, 80) if (self.calibrado or self.auto_calibrado) else (235, 190, 60)
        cv2.putText(frame, calib_txt, (p_x1 + 10, p_y1 + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.34, calib_col, 1, cv2.LINE_AA)

        modo_fondo = "Fondo: Camara en vivo" if self.mostrar_video_camara else "Fondo: Avatar (Sin espejo)"
        cv2.putText(frame, modo_fondo, (p_x1 + 10, p_y1 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (145, 160, 175), 1, cv2.LINE_AA)

        # 3. Panel derecho mini K-Means (205x76 px)
        k_x2 = ancho - 10
        k_x1 = k_x2 - 205
        k_y1, k_y2 = 56, 132
        _rectangulo_translucido(frame, k_x1, k_y1, k_x2, k_y2, (22, 18, 15), alpha=0.78)
        cv2.rectangle(frame, (k_x1, k_y1), (k_x2, k_y2), (55, 65, 80), 1)
        cv2.rectangle(frame, (k_x1, k_y1), (k_x1 + 3, k_y2), self.cluster_color, -1)

        cv2.putText(frame, "K-MEANS FATIGA", (k_x1 + 10, k_y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (235, 190, 60), 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            self.cluster_actual_fatiga.upper(),
            (k_x1 + 10, k_y1 + 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            self.cluster_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.accion_ergonomica[:28],
            (k_x1 + 10, k_y1 + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (190, 200, 215),
            1,
            cv2.LINE_AA,
        )

        # 4. Notificación Toast pequeña en esquina inferior derecha (en lugar de banner gigante)
        if self.alerta_activa:
            tw, th = 295, 50
            tx2 = ancho - 12
            tx1 = tx2 - tw
            ty2 = alto - 32
            ty1 = ty2 - th
            _rectangulo_translucido(frame, tx1, ty1, tx2, ty2, (28, 20, 45), alpha=0.92)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (80, 90, 235), 1)
            cv2.rectangle(frame, (tx1, ty1), (tx1 + 4, ty2), (70, 80, 245), -1)

            cv2.putText(
                frame,
                "Correccion postural sugerida",
                (tx1 + 12, ty1 + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (140, 155, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                self.mensaje_alerta[:38],
                (tx1 + 12, ty1 + 39),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (230, 235, 242),
                1,
                cv2.LINE_AA,
            )

        # 5. Pie de página ultra compacto (24 px)
        cv2.rectangle(frame, (0, alto - 24), (ancho, alto), (16, 14, 12), -1)
        controles = "[T] Overlay  [V] Cam  [P] Pausa  [F] Fondo  [C] Calib  [M] Demo  [K] KMeans  [S] Foto  [Q] Salir"
        cv2.putText(frame, controles, (10, alto - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 172, 188), 1, cv2.LINE_AA)

    def calibrar(self, frame: np.ndarray) -> bool:
        """Calibra la postura neutra de referencia del usuario (tanto frontal como perfil)."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            alto_f, ancho_f = frame.shape[:2]
            aspect_ratio = alto_f / max(1.0, float(ancho_f))
            world_lm = results.pose_world_landmarks.landmark if results.pose_world_landmarks else None
            m = self.extraer_metricas_postura(results.pose_landmarks.landmark, world_lm, aspect_ratio)

            self.vista_corporal = m["vista_corporal"]
            self.baseline_cervical = float(m["ang_2d_plano"])
            self.baseline_hombros = float(m["desnivel_hombros"])
            self.baseline_distancia = max(0.18, float(m["dist_hombros"]))
            self.baseline_ratio_cuello = max(0.20, float(m["ratio_cuello"]))
            self.baseline_ratio_nariz = max(0.20, float(m["ratio_nariz"]))
            self.baseline_pitch_nariz = float(m["pitch_nariz"])
            if m["ang_3d_sagital"] is not None:
                self.baseline_3d_sagital = float(m["ang_3d_sagital"])
            self.calibrado = True
            self.auto_calibrado = True
            self._ema_inicializado = False
            print(
                f"[CALIBRACIÓN OK - {self.vista_corporal}] "
                f"Ratio cuello: {self.baseline_ratio_cuello:.2f} | "
                f"Hombros: {self.baseline_hombros:.1f}° | Dist: {self.baseline_distancia:.2f}"
            )
            return True
        return False

    def generar_frame_simulado(self, renderizar: bool = True) -> Optional[np.ndarray]:
        """Genera telemetría (y avatar opcional) en modo demostración sintético sin fugas de RAM."""
        self.sim_phase += 0.06
        ciclo = math.sin(self.sim_phase)
        offset_cabeza = int(ciclo * 42)
        desnivel = int(math.cos(self.sim_phase * 0.7) * 18)

        ang_cerv = abs(offset_cabeza) * 0.72 + 10.0
        desn_h = abs(desnivel) * 0.42
        es_riesgo = ang_cerv > self.UMBRAL_CUELLO_MAX or desn_h > self.UMBRAL_HOMBROS_MAX

        diag = "POSTURA OPTIMA" if not es_riesgo else "RIESGO: Cervical adelantada"
        color_st = (90, 210, 80) if not es_riesgo else (70, 80, 245)

        self.total_frames += 1
        if es_riesgo:
            self.frames_riesgo += 1
            self.segundos_mala_postura += 0.12
            if self.segundos_mala_postura >= self.TIEMPO_ALERTA_SEGUNDOS:
                if not self.alerta_activa:
                    self.total_alertas_disparadas += 1
                self.alerta_activa = True
                self.mensaje_alerta = "Cervical adelantada — Alinea espalda"
        else:
            self.frames_correctos += 1
            self.segundos_mala_postura = 0.0
            self.alerta_activa = False
            self.mensaje_alerta = ""

        self.score_ergonomico = max(15, int(100 - (ang_cerv - 15) * 3)) if ang_cerv > 15 else 100
        self.ultimo_angulo_cervical = ang_cerv
        self.ultimo_desnivel_hombros = desn_h
        self.tipo_diagnostico = diag

        # Deques con maxlen=300 garantizan consumo de memoria constante
        self.historial_cervical.append(ang_cerv)
        self.historial_hombros.append(desn_h)
        self.historial_score.append(self.score_ergonomico)

        if self.total_frames % 50 == 0:
            self.evaluar_bloque_con_kmeans()

        if not renderizar:
            return None

        ancho, alto = self.ancho_vista, self.alto_vista
        canvas = self._obtener_canvas_libre()
        canvas[:] = (28, 22, 18)

        cx, cy = ancho // 2, alto // 2 + 10
        cv2.line(canvas, (cx, 58), (cx, alto - 38), (48, 42, 36), 1, cv2.LINE_AA)

        cabeza_pt = (cx + offset_cabeza, cy - 95)
        cuello_pt = (cx + offset_cabeza // 3, cy - 40)
        h_izq_pt = (cx - 88, cy - 12 - desnivel)
        h_der_pt = (cx + 88, cy - 12 + desnivel)
        torso_base = (cx, cy + 95)

        cv2.circle(canvas, cabeza_pt, 28, (38, 34, 30), -1, cv2.LINE_AA)
        cv2.circle(canvas, cabeza_pt, 28, color_st, 2, cv2.LINE_AA)
        cv2.line(canvas, cabeza_pt, cuello_pt, color_st, 3, cv2.LINE_AA)
        cv2.line(canvas, h_izq_pt, h_der_pt, color_st, 3, cv2.LINE_AA)
        cv2.line(canvas, cuello_pt, torso_base, (85, 95, 110), 2, cv2.LINE_AA)
        cv2.circle(canvas, h_izq_pt, 5, (235, 240, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, h_der_pt, 5, (235, 240, 245), -1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "[MODO DEMOSTRACION]",
            (cx - 72, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 190, 60),
            1,
            cv2.LINE_AA,
        )

        self.renderizar_hud_compacto(canvas, diag, color_st, ang_cerv, desn_h)
        return canvas


class ControladorMonitoreo:
    """
    Orquesta el hilo de visión/cámara en segundo plano y permite alternar sin cortes
    entre la Vista 1 (Panel Compacto OpenCV) y la Vista 2 (Overlay Transparente Tkinter)
    en el hilo principal.
    """

    def __init__(
        self,
        asistente: AsistenteErgonomico,
        gestor_camara: GestorCamara,
        monitor_recursos: MonitorRecursos,
        vista_inicial: str = "overlay",
    ):
        self.asistente = asistente
        self.gestor_camara = gestor_camara
        self.monitor_recursos = monitor_recursos
        self.vista_actual = vista_inicial

        self._lock = threading.Lock()
        self._detener = threading.Event()
        self._ultimo_frame_vista: Optional[np.ndarray] = None
        self._frame_seq: int = 0
        self._solicitud_calibrar = False
        self._solicitud_cambiar_cam = False
        self._solicitud_pausa_toggle = False
        self._solicitud_demo_toggle = False
        self._solicitud_kmeans = False
        self._hilo: Optional[threading.Thread] = None

    def iniciar_hilo_vision(self):
        self._hilo = threading.Thread(target=self._bucle_vision, name="HiloVisionErgonomia", daemon=True)
        self._hilo.start()

    def detener(self):
        self._detener.set()
        if self._hilo is not None and self._hilo.is_alive():
            self._hilo.join(timeout=2.5)

    def solicitar_calibracion(self):
        with self._lock:
            self._solicitud_calibrar = True

    def solicitar_cambio_camara(self):
        with self._lock:
            self._solicitud_cambiar_cam = True

    def solicitar_pausa_toggle(self):
        with self._lock:
            self._solicitud_pausa_toggle = True

    def solicitar_demo_toggle(self):
        with self._lock:
            self._solicitud_demo_toggle = True

    def solicitar_evaluacion_kmeans(self):
        with self._lock:
            self._solicitud_kmeans = True

    def establecer_vista(self, nueva_vista: str):
        with self._lock:
            self.vista_actual = nueva_vista

    def obtener_frame_para_vista(self, ultimo_seq: int = -1) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._ultimo_frame_vista is None or self._frame_seq == ultimo_seq:
                return None, self._frame_seq
            return self._ultimo_frame_vista.copy(), self._frame_seq

    def obtener_estado_ui(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "detener_solicitado": self._detener.is_set(),
                "score": self.asistente.score_ergonomico,
                "alerta_activa": self.asistente.alerta_activa,
                "mensaje_alerta": self.asistente.mensaje_alerta,
                "diagnostico": self.asistente.tipo_diagnostico,
                "segundos_mala_postura": self.asistente.segundos_mala_postura,
                "ventana_alerta": self.asistente.TIEMPO_ALERTA_SEGUNDOS,
                "cluster": self.asistente.cluster_actual_fatiga,
                "angulo_cervical": self.asistente.ultimo_angulo_cervical,
                "desnivel_hombros": self.asistente.ultimo_desnivel_hombros,
                "ram_mb": self.asistente.ram_actual_mb,
                "estado_ram": self.asistente.estado_ram,
                "estado_camara": self.gestor_camara.estado,
                "app_conflicto": self.gestor_camara.app_en_conflicto,
                "camara_indice": self.gestor_camara.indice_actual,
                "camara_label": self.asistente.camara_label,
                "modo_demo": self.asistente.modo_simulacion,
            }

    def _bucle_vision(self):
        ultimo_muestreo_ram = 0.0

        while not self._detener.is_set():
            # Procesar solicitudes desde cualquiera de las 2 vistas
            with self._lock:
                vista = self.vista_actual
                req_cal = self._solicitud_calibrar
                req_cam = self._solicitud_cambiar_cam
                req_pausa = self._solicitud_pausa_toggle
                req_demo = self._solicitud_demo_toggle
                req_kmeans = self._solicitud_kmeans
                self._solicitud_calibrar = False
                self._solicitud_cambiar_cam = False
                self._solicitud_pausa_toggle = False
                self._solicitud_demo_toggle = False
                self._solicitud_kmeans = False

            if req_demo:
                self.asistente.modo_simulacion = not self.asistente.modo_simulacion
                if self.asistente.modo_simulacion:
                    self.gestor_camara.cerrar()
                    self.asistente.camara_label = "MODO DEMO"
                else:
                    self.gestor_camara.abrir()
                    self.asistente.camara_label = self.gestor_camara.obtener_nombre_camara_actual()

            if req_cam and not self.asistente.modo_simulacion:
                self.gestor_camara.siguiente_camara()
                self.asistente.camara_label = self.gestor_camara.obtener_nombre_camara_actual()

            if req_pausa and not self.asistente.modo_simulacion:
                self.gestor_camara.alternar_pausa_manual()

            if req_kmeans:
                self.asistente.evaluar_bloque_con_kmeans()

            # Monitoreo de RAM cada 2 segundos + limpieza suave periódica
            ahora = time.time()
            if (ahora - ultimo_muestreo_ram) >= 2.0:
                ultimo_muestreo_ram = ahora
                muestra = self.monitor_recursos.muestrear()
                self.asistente.ram_actual_mb = muestra.ram_actual_mb
                self.asistente.estado_ram = self.monitor_recursos.obtener_estado_memoria()

            if self.asistente.total_frames > 0 and self.asistente.total_frames % 600 == 0:
                gc.collect()

            renderizar_vista = vista == "compacta"
            frame_salida = None

            if self.asistente.modo_simulacion:
                self.asistente.camara_label = "MODO DEMO"
                if req_cal:
                    self.asistente.calibrado = True
                frame_salida = self.asistente.generar_frame_simulado(renderizar=renderizar_vista)
                time.sleep(0.11 if not renderizar_vista else 0.045)
            else:
                self.asistente.camara_label = self.gestor_camara.obtener_nombre_camara_actual()
                ok, frame_cam, estado_cam = self.gestor_camara.leer_frame()

                if ok and frame_cam is not None:
                    frame_cam = cv2.flip(frame_cam, 1)
                    if req_cal:
                        self.asistente.calibrar(frame_cam)
                    frame_salida = self.asistente.procesar_frame(frame_cam, renderizar=renderizar_vista)
                    # En Vista Overlay (background) bajamos a ~8 FPS para mínimo consumo RAM/CPU
                    time.sleep(0.12 if not renderizar_vista else 0.02)
                else:
                    self.asistente.alerta_activa = False
                    if renderizar_vista:
                        if estado_cam == GestorCamara.ESTADO_PAUSADA_CONFLICTO:
                            frame_salida = self.asistente.renderizar_pantalla_pausa(
                                "Camara cedida a videollamada",
                                f"App activa: {self.gestor_camara.app_en_conflicto}",
                                es_conflicto=True,
                            )
                        elif estado_cam == GestorCamara.ESTADO_PAUSADA_MANUAL:
                            frame_salida = self.asistente.renderizar_pantalla_pausa(
                                "Camara liberada manualmente",
                                "Hardware disponible para otras aplicaciones.",
                                es_conflicto=False,
                            )
                        else:
                            frame_salida = self.asistente.renderizar_pantalla_pausa(
                                "Esperando camara disponible...",
                                "Presiona [M] para Modo Demo o [V] para otra camara.",
                                es_conflicto=False,
                            )
                    time.sleep(0.25)

            if renderizar_vista and frame_salida is not None:
                with self._lock:
                    self._ultimo_frame_vista = frame_salida
                    self._frame_seq += 1


def _ejecutar_vista_compacta(controlador: ControladorMonitoreo) -> str:
    """
    Ejecuta la Vista 1 (Cámara en vivo con HUD y notificaciones compactas) en el hilo principal.
    Retorna: "overlay" si el usuario presionó [T], o "salir" si cerró la ventana o presionó [Q].
    """
    nombre_ventana = "PostuFix | Vista 1 - Monitor Compacto"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, controlador.asistente.ancho_vista, controlador.asistente.alto_vista)

    siguiente_accion = "salir"
    ultimo_seq_mostrado = -1
    ultimo_frame_mostrado = None
    try:
        while True:
            frame, seq = controlador.obtener_frame_para_vista(ultimo_seq_mostrado)
            if frame is not None:
                ultimo_seq_mostrado = seq
                ultimo_frame_mostrado = frame
                cv2.imshow(nombre_ventana, frame)

            # Detectar si el usuario cerró la ventana con la X
            try:
                if cv2.getWindowProperty(nombre_ventana, cv2.WND_PROP_VISIBLE) < 1:
                    siguiente_accion = "salir"
                    break
            except Exception:
                pass

            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                siguiente_accion = "salir"
                break
            elif key in (ord("t"), ord("T"), 9):  # T o TAB -> cambiar a Vista 2 Overlay
                siguiente_accion = "overlay"
                break
            elif key in (ord("c"), ord("C")):
                controlador.solicitar_calibracion()
            elif key in (ord("v"), ord("V")):
                controlador.solicitar_cambio_camara()
            elif key in (ord("p"), ord("P")):
                controlador.solicitar_pausa_toggle()
            elif key in (ord("f"), ord("F")):
                controlador.asistente.mostrar_video_camara = not controlador.asistente.mostrar_video_camara
                estado_f = "Camara en vivo" if controlador.asistente.mostrar_video_camara else "Avatar sin espejo"
                print(f"[INTERFAZ] Modo visual cambiado a: {estado_f}")
            elif key in (ord("m"), ord("M")):
                controlador.solicitar_demo_toggle()
            elif key in (ord("k"), ord("K")):
                controlador.solicitar_evaluacion_kmeans()
                print(
                    f"[K-MEANS] Fatiga: {controlador.asistente.cluster_actual_fatiga.upper()} -> "
                    f"{controlador.asistente.accion_ergonomica}"
                )
            elif key in (ord("s"), ord("S")) and ultimo_frame_mostrado is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evidencia_postura_{ts}.png"
                cv2.imwrite(filename, ultimo_frame_mostrado)
                print(f"[CAPTURA] Guardada en: {filename}")
    finally:
        cv2.destroyAllWindows()

    return siguiente_accion


def parsear_argumentos():
    parser = argparse.ArgumentParser(
        description="Asistente Ergonómico Inteligente (Multi-cámara, Anti-conflicto, 2 Vistas, Monitor RAM)"
    )
    parser.add_argument("--camara", type=int, default=None, help="Índice de la cámara a utilizar (ej. 0, 1)")
    parser.add_argument("--listar-camaras", action="store_true", help="Lista las cámaras disponibles y sale")
    parser.add_argument(
        "--vista",
        choices=["overlay", "compacta"],
        default=None,
        help="Vista inicial: 'overlay' (segundo plano/notificaciones) o 'compacta' (avatar sin espejo)",
    )
    parser.add_argument("--demo", action="store_true", help="Iniciar en modo demostración sintético")
    parser.add_argument("--sin-dialogo", action="store_true", help="Omitir el selector visual al iniciar")
    parser.add_argument("--sin-anti-conflicto", action="store_true", help="Desactivar cesión automática en videollamadas")
    parser.add_argument("--solo-avatar", action="store_true", help="Ocultar video de cámara en Vista 1 y mostrar solo avatar")
    parser.add_argument(
        "--complexity",
        type=int,
        choices=[0, 1],
        default=0,
        help="Complejidad MediaPipe BlazePose (0=Lite menor RAM [defecto], 1=Full)",
    )
    return parser.parse_args()


def main():
    args = parsear_argumentos()
    monitor_ram = MonitorRecursos()

    print("=" * 72)
    print(" ASISTENTE ERGONÓMICO INTELIGENTE v3.0 - POLITÉCNICO GRANCOLOMBIANO")
    print("=" * 72)
    print(" Escaneando cámaras disponibles...")
    camaras = detectar_camaras_disponibles()

    if camaras:
        for cam in camaras:
            print(f"  -> {cam.etiqueta()} [{cam.backend}]")
    else:
        print("  -> No se detectaron cámaras físicas conectadas.")

    if args.listar_camaras:
        return

    # Selección de cámara y vista inicial
    vista_inicial = args.vista or "overlay"
    anti_conflicto = not args.sin_anti_conflicto
    modo_demo = args.demo
    indice_camara = args.camara if args.camara is not None else (camaras[0].indice if camaras else 0)

    # Mostrar diálogo si no se desactivó explícitamente (y especialmente si hay >1 cámara o no se fijó vista)
    debe_mostrar_dialogo = (not args.sin_dialogo) and (
        args.camara is None and args.vista is None and not args.demo
    )
    if debe_mostrar_dialogo:
        seleccion = mostrar_selector_inicio(camaras, vista_defecto=vista_inicial)
        if not seleccion.get("iniciar", False):
            print("[INFO] Ejecución cancelada por el usuario.")
            return
        modo_demo = bool(seleccion["modo_demo"])
        indice_camara = int(seleccion["indice_camara"])
        vista_inicial = str(seleccion["vista_inicial"])
        anti_conflicto = bool(seleccion["anti_conflicto"])
    elif not camaras and not modo_demo:
        print("[AVISO] Sin cámara física detectada; activando Modo Demostración Sintético.")
        modo_demo = True

    asistente = AsistenteErgonomico(
        model_complexity=args.complexity,
        mostrar_video_camara=not args.solo_avatar,
    )
    asistente.modo_simulacion = modo_demo

    gestor = GestorCamara(
        indice_camara=indice_camara,
        camaras_disponibles=camaras,
        anti_conflicto=anti_conflicto,
    )

    if not modo_demo:
        abierta = gestor.abrir(indice_camara)
        if not abierta and gestor.estado == GestorCamara.ESTADO_PAUSADA_CONFLICTO:
            print(f"[ANTI-CONFLICTO] Cámara actualmente en uso por {gestor.app_en_conflicto}. En espera sin bloquear.")
        elif not abierta:
            print("[AVISO] No se pudo abrir la cámara seleccionada. Activando Modo Demostración.")
            asistente.modo_simulacion = True

    # Fijar baseline de RAM tras cargar MediaPipe + K-Means + Cámara
    ram_inicial = monitor_ram.fijar_baseline()
    asistente.ram_actual_mb = ram_inicial

    print("-" * 72)
    print(f" Fuente activa    : {'Modo Demostración' if asistente.modo_simulacion else gestor.obtener_nombre_camara_actual()}")
    print(f" Vista inicial    : {'Vista 2 (Overlay / Solo notificaciones)' if vista_inicial == 'overlay' else 'Vista 1 (Panel compacto sin espejo)'}")
    print(f" Anti-conflicto   : {'ACTIVO (cede cámara en videollamadas)' if anti_conflicto else 'DESACTIVADO'}")
    print(f" RAM tras carga   : {ram_inicial:.1f} MB (MediaPipe complexity={args.complexity})")
    print("-" * 72)

    controlador = ControladorMonitoreo(asistente, gestor, monitor_ram, vista_inicial=vista_inicial)
    controlador.iniciar_hilo_vision()

    vista = vista_inicial
    try:
        while True:
            controlador.establecer_vista(vista)
            if vista == "overlay":
                overlay = VentanaOverlayNotificaciones(
                    obtener_estado=controlador.obtener_estado_ui,
                    on_calibrar=controlador.solicitar_calibracion,
                    on_cambiar_camara=controlador.solicitar_cambio_camara,
                    on_alternar_pausa=controlador.solicitar_pausa_toggle,
                    on_alternar_demo=controlador.solicitar_demo_toggle,
                    on_evaluar_kmeans=controlador.solicitar_evaluacion_kmeans,
                )
                siguiente = overlay.ejecutar()
            else:
                siguiente = _ejecutar_vista_compacta(controlador)

            if siguiente in ("overlay", "compacta"):
                vista = siguiente
            else:
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción recibida. Cerrando limpiamente...")
    finally:
        controlador.detener()
        gestor.cerrar()
        asistente.cerrar()

    resumen_ram = monitor_ram.obtener_resumen()
    tiempo_total = time.time() - asistente.tiempo_inicio_sesion

    print("\n" + "=" * 72)
    print(" RESUMEN DE LA SESIÓN DE MONITOREO ERGONÓMICO")
    print("=" * 72)
    print(f" - Duración de la sesión                 : {tiempo_total:.1f} s")
    print(f" - Fotogramas evaluados                  : {asistente.total_frames}")
    if asistente.total_frames > 0:
        cumplimiento = (asistente.frames_correctos / asistente.total_frames) * 100.0
        print(f" - Cumplimiento ergonómico general       : {cumplimiento:.1f}%")
    print(f" - Alertas posturales disparadas         : {asistente.total_alertas_disparadas}")
    print(f" - Clasificación final K-Means           : {asistente.cluster_actual_fatiga.upper()}")
    print(f" - Conflictos de videollamada resueltos  : {gestor.total_conflictos_resueltos}")
    print(
        f" - Consumo RAM (Inicio / Prom / Pico)    : "
        f"{resumen_ram['ram_baseline_mb']:.1f} / {resumen_ram['ram_promedio_mb']:.1f} / "
        f"{resumen_ram['ram_pico_proceso_mb']:.1f} MB [{resumen_ram['estado_ram']}]"
    )
    print(
        f" - Estabilidad de memoria (sin fugas)    : "
        f"{'VERIFICADA' if resumen_ram['memoria_estable_sin_fugas'] else 'REVISAR'} "
        f"(deriva: {resumen_ram['deriva_memoria_mb']:+.2f} MB)"
    )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
