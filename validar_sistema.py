"""
=============================================================================
SCRIPT DE VALIDACIÓN INTEGRAL DE CORRECCIONES DOCENTES
PROYECTO: Asistente Ergonómico Inteligente - Politécnico Grancolombiano
=============================================================================
Valida de forma reproducible y genera evidencia cuantitativa para los 4
puntos solicitados por el docente:
  1. Selección multi-cámara (escaneo + cambio entre múltiples dispositivos).
  2. Manejo de conflictos de cámara con videollamadas (detección Windows +
     cesión automática + recuperación sin bloqueo).
  3. Interfaz menos disruptiva (Vista 1 sin espejo + Vista 2 en background
     sin renderizado de video + buffers acotados).
  4. Validación del consumo de recursos (RAM actual, privada, pico, deriva
     y comparación Vista 1 vs Vista 2) + gráfica evidencia_consumo_ram.png.
=============================================================================
"""

from datetime import datetime
import gc
import json
import os
import time
from typing import Any, Dict, List

import numpy as np

from gestor_camara import (
    GestorCamara,
    InfoCamara,
    detectar_apps_usando_camara,
    detectar_camaras_disponibles,
)
from monitor_recursos import MonitorRecursos


class _FakeCapture:
    """Captura simulada para probar liberación de hardware en escenarios de conflicto."""

    def __init__(self, fallar_lectura: bool = False):
        self._abierta = True
        self.fallar_lectura = fallar_lectura
        self.release_llamado = False

    def isOpened(self) -> bool:
        return self._abierta

    def read(self):
        if self.fallar_lectura or not self._abierta:
            return False, None
        return True, np.full((480, 640, 3), 120, dtype=np.uint8)

    def release(self):
        self._abierta = False
        self.release_llamado = True


def validar_multicamara() -> Dict[str, Any]:
    print("\n[1/4] VALIDACIÓN MULTI-CÁMARA")
    print("-" * 65)
    camaras_reales = detectar_camaras_disponibles(max_indices=5)
    print(f" -> Cámaras físicas detectadas en este equipo: {len(camaras_reales)}")
    for cam in camaras_reales:
        print(f"    * {cam.etiqueta()} [{cam.backend}]")

    # Prueba lógica de rotación con 2 cámaras simuladas
    camaras_mock = [
        InfoCamara(0, "Cámara Integrada", 640, 480, "MediaFoundation"),
        InfoCamara(1, "Cámara USB Externa", 1280, 720, "MediaFoundation"),
    ]
    gestor = GestorCamara(indice_camara=0, camaras_disponibles=camaras_mock, anti_conflicto=False)
    gestor.abrir = lambda idx=None: (setattr(gestor, "indice_actual", idx if idx is not None else gestor.indice_actual) or True)

    assert gestor.indice_actual == 0
    gestor.siguiente_camara()
    paso_1 = gestor.indice_actual
    gestor.siguiente_camara()
    paso_2 = gestor.indice_actual
    assert paso_1 == 1 and paso_2 == 0, "Error en rotación multi-cámara"
    print(f" -> Prueba de selección y rotación en caliente (0 -> {paso_1} -> {paso_2}): [APROBADA]")

    return {
        "camaras_fisicas_detectadas": [
            {
                "indice": c.indice,
                "nombre": c.nombre,
                "resolucion": f"{c.ancho}x{c.alto}",
                "backend": c.backend,
            }
            for c in camaras_reales
        ],
        "rotacion_multicamara_ok": True,
    }


def validar_manejo_conflictos() -> Dict[str, Any]:
    print("\n[2/4] VALIDACIÓN DE MANEJO DE CONFLICTOS CON VIDEOLLAMADAS")
    print("-" * 65)

    # 2A. Verificar lectura real del registro de Windows
    apps_actuales = detectar_apps_usando_camara()
    print(f" -> Consulta nativa Windows (CapabilityAccessManager): {apps_actuales or 'Sin apps externas activas'}")

    # 2B. Simular llegada de videollamada (Microsoft Teams) mientras el asistente está activo
    estado_sim = {"apps": []}
    gestor = GestorCamara(
        indice_camara=0,
        anti_conflicto=True,
        intervalo_verificacion_seg=0.0,
        detector_apps_externas=lambda: list(estado_sim["apps"]),
    )
    fake_cap = _FakeCapture()
    gestor.cap = fake_cap
    gestor.estado = GestorCamara.ESTADO_ACTIVA

    # Lectura normal sin conflicto
    ok1, frame1, st1 = gestor.leer_frame()
    assert ok1 and frame1 is not None and st1 == GestorCamara.ESTADO_ACTIVA

    # Inicia videollamada en Teams/Zoom -> el asistente debe liberar inmediatamente cap.release()
    estado_sim["apps"] = ["Microsoft Teams"]
    ok2, frame2, st2 = gestor.leer_frame()
    assert not ok2 and frame2 is None
    assert st2 == GestorCamara.ESTADO_PAUSADA_CONFLICTO
    assert fake_cap.release_llamado and not gestor.esta_abierta
    assert "Microsoft Teams" in gestor.app_en_conflicto
    print(" -> Detección de videollamada y liberación inmediata (cap.release()): [APROBADA]")

    # Finaliza la videollamada -> el asistente debe recuperar la cámara automáticamente
    estado_sim["apps"] = []
    gestor.ultimo_intento_reconexion = 0.0

    def _mock_reabrir(_idx=None):
        gestor.cap = _FakeCapture()
        gestor.estado = GestorCamara.ESTADO_ACTIVA
        gestor.app_en_conflicto = ""
        return True

    gestor.abrir = _mock_reabrir
    ok3, frame3, st3 = gestor.leer_frame()
    assert ok3 and frame3 is not None and st3 == GestorCamara.ESTADO_ACTIVA
    print(" -> Recuperación automática al terminar la videollamada: [APROBADA]")

    # 2C. Simular pérdida de stream exclusiva (otra app toma DirectShow sin registro)
    gestor2 = GestorCamara(
        indice_camara=0,
        anti_conflicto=True,
        intervalo_verificacion_seg=99.0,
        detector_apps_externas=lambda: [],
    )
    fake_cap_roto = _FakeCapture(fallar_lectura=True)
    gestor2.cap = fake_cap_roto
    gestor2.estado = GestorCamara.ESTADO_ACTIVA

    for _ in range(5):
        gestor2.leer_frame()

    assert fake_cap_roto.release_llamado and gestor2.estado == GestorCamara.ESTADO_PAUSADA_CONFLICTO
    print(" -> Protección ante bloqueo exclusivo / pérdida de stream: [APROBADA]")

    return {
        "apps_detectadas_actualmente_windows": apps_actuales,
        "cesion_automatica_videollamada_ok": True,
        "proteccion_perdida_stream_ok": True,
        "eventos_registrados": gestor.eventos_conflicto + gestor2.eventos_conflicto,
    }


def validar_interfaces_y_ram() -> Dict[str, Any]:
    print("\n[3/4] VALIDACIÓN DE 2 VISTAS NO DISRUPTIVAS Y BUFFERS ACOTADOS")
    print("-" * 65)

    monitor = MonitorRecursos(max_muestras=1000)
    m_antes = monitor.muestrear()
    print(f" -> RAM base (antes de cargar IA): {m_antes.ram_actual_mb:.2f} MB")

    from asistente_ergonomico import AsistenteErgonomico

    asistente = AsistenteErgonomico(model_complexity=0, mostrar_video_camara=True)
    assert asistente.mostrar_video_camara is True, "Vista 1 debe mostrar la cámara en vivo por defecto"

    # Warm-up inicial de 1 frame para que TensorFlow Lite / XNNPACK reserve sus buffers estáticos
    frame_prueba = np.full((480, 640, 3), 40, dtype=np.uint8)
    _ = asistente.procesar_frame(frame_prueba, renderizar=False)

    m_carga = monitor.muestrear()
    monitor.fijar_baseline()
    print(f" -> RAM tras cargar BlazePose Lite + K-Means (warm-up): {m_carga.ram_actual_mb:.2f} MB")

    print("\n[4/4] BENCHMARK DE CONSUMO DE RAM (VISTA 1 COMPACTA vs VISTA 2 OVERLAY)")
    print("-" * 65)

    serie_fases: List[Dict[str, Any]] = []
    t_inicio = time.perf_counter()

    def registrar_punto(fase: str):
        m = monitor.muestrear()
        serie_fases.append(
            {
                "t_seg": round(time.perf_counter() - t_inicio, 2),
                "fase": fase,
                "ram_actual_mb": m.ram_actual_mb,
                "ram_privada_mb": m.ram_privada_mb,
                "ram_pico_mb": m.ram_pico_mb,
            }
        )

    registrar_punto("Carga IA")

    # Fase A: 180 ciclos en Vista 1 Compacta (renderizado Avatar + HUD ROI sin copias completas)
    t_v1_ini = time.perf_counter()
    ram_v1 = []
    for i in range(180):
        frame = asistente.generar_frame_simulado(renderizar=True)
        assert frame is not None and frame.shape == (460, 720, 3)
        if i % 15 == 0:
            registrar_punto("Vista 1 (Compacta)")
            ram_v1.append(monitor.ultima_muestra.ram_actual_mb)
    dt_v1 = time.perf_counter() - t_v1_ini
    fps_v1 = 180.0 / max(1e-6, dt_v1)

    # Fase B: 180 ciclos en Vista 2 Overlay/Background (sin renderizar canvas OpenCV)
    t_v2_ini = time.perf_counter()
    ram_v2 = []
    for i in range(180):
        frame = asistente.generar_frame_simulado(renderizar=False)
        assert frame is None, "Vista 2 background no debe reservar/renderizar frames OpenCV"
        if i % 15 == 0:
            registrar_punto("Vista 2 (Overlay/BG)")
            ram_v2.append(monitor.ultima_muestra.ram_actual_mb)
    dt_v2 = time.perf_counter() - t_v2_ini
    fps_v2 = 180.0 / max(1e-6, dt_v2)

    # Fase C: Prueba de inferencia real con MediaPipe sobre frame sintético 640x480 (60 frames)
    frame_prueba = np.full((480, 640, 3), 40, dtype=np.uint8)
    ram_mp = []
    for i in range(60):
        _ = asistente.procesar_frame(frame_prueba, renderizar=False)
        if i % 10 == 0:
            registrar_punto("Inferencia BlazePose")
            ram_mp.append(monitor.ultima_muestra.ram_actual_mb)

    # Prueba visual rápida (auto-cierre) del selector inicial y del Overlay transparente
    from interfaz_overlay import VentanaOverlayNotificaciones, mostrar_selector_inicio

    sel = mostrar_selector_inicio(
        [InfoCamara(0, "Cámara Prueba", 640, 480, "MediaFoundation")],
        vista_defecto="overlay",
        auto_cerrar_ms=350,
    )
    assert sel["iniciar"] is True and sel["vista_inicial"] == "overlay"

    overlay = VentanaOverlayNotificaciones(
        obtener_estado=lambda: {
            "detener_solicitado": False,
            "score": 92,
            "ram_mb": 285.0,
            "estado_camara": "ACTIVA",
            "app_conflicto": "",
            "alerta_activa": True,
            "mensaje_alerta": "Cervical adelantada — Alinea espalda",
            "cluster": "Baja fatiga",
            "angulo_cervical": 30.5,
            "desnivel_hombros": 2.1,
            "modo_demo": False,
            "camara_indice": 0,
        },
        on_calibrar=lambda: None,
        on_cambiar_camara=lambda: None,
        on_alternar_pausa=lambda: None,
        on_alternar_demo=lambda: None,
        on_evaluar_kmeans=lambda: None,
    )
    accion = overlay.ejecutar(auto_cerrar_ms=900)
    assert accion == "salir"
    assert overlay._toast_visible is True, "El Toast flotante de Vista 2 no se mostró correctamente"
    print(" -> Interfaz Tkinter (Selector inicial + Vista 2 Overlay Toast visible): [APROBADA]")

    # Verificar que los historiales acotados nunca excedan maxlen=300 (420 frames ejecutados!)
    assert len(asistente.historial_cervical) <= 300
    assert len(asistente.historial_hombros) <= 300
    assert len(asistente.historial_score) <= 300
    print(f" -> Vista 1 sin espejo (Avatar 720x460): [APROBADA] ({fps_v1:.1f} FPS sintéticos)")
    print(f" -> Vista 2 Background sin render OpenCV: [APROBADA] ({fps_v2:.1f} FPS sintéticos)")
    print(f" -> Buffers acotados anti-fugas (<=300 tras {asistente.total_frames} frames): [APROBADA]")

    # Liberación controlada
    asistente.cerrar()
    ram_post_cierre = monitor.liberar_memoria_temporal()
    registrar_punto("Post-cierre")

    resumen = monitor.obtener_resumen()
    print(f" -> Backend de medición : {resumen['backend_medicion']}")
    print(f" -> RAM Base (pre-IA)   : {m_antes.ram_actual_mb:.2f} MB")
    print(f" -> RAM Tras Carga IA   : {m_carga.ram_actual_mb:.2f} MB")
    print(f" -> RAM Prom Vista 1    : {sum(ram_v1)/len(ram_v1):.2f} MB")
    print(f" -> RAM Prom Vista 2    : {sum(ram_v2)/len(ram_v2):.2f} MB")
    print(f" -> RAM Prom BlazePose  : {sum(ram_mp)/len(ram_mp):.2f} MB")
    print(f" -> RAM Pico Proceso    : {resumen['ram_pico_proceso_mb']:.2f} MB")
    print(f" -> RAM Post-cierre     : {ram_post_cierre:.2f} MB")
    print(
        f" -> Estabilidad memoria : {'SIN FUGAS DETECTADAS' if resumen['memoria_estable_sin_fugas'] else 'REVISAR'} "
        f"(deriva: {resumen['deriva_memoria_mb']:+.2f} MB)"
    )

    # Generar gráfica de evidencia de consumo de RAM si matplotlib está instalado
    grafica_generada = _generar_grafica_ram(serie_fases)

    return {
        "ram_base_pre_ia_mb": m_antes.ram_actual_mb,
        "ram_post_carga_ia_mb": m_carga.ram_actual_mb,
        "ram_promedio_vista_1_mb": round(sum(ram_v1) / len(ram_v1), 2),
        "ram_promedio_vista_2_mb": round(sum(ram_v2) / len(ram_v2), 2),
        "ram_promedio_blazepose_mb": round(sum(ram_mp) / len(ram_mp), 2),
        "ram_post_cierre_mb": ram_post_cierre,
        "fps_vista_1_render": round(fps_v1, 1),
        "fps_vista_2_sin_render": round(fps_v2, 1),
        "resumen_monitor": resumen,
        "grafica_evidencia": grafica_generada,
    }


def _generar_grafica_ram(serie: List[Dict[str, Any]]) -> str:
    """Genera una gráfica profesional del consumo de RAM a lo largo de las fases."""
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidencia_consumo_ram.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [p["t_seg"] for p in serie]
        ys_ws = [p["ram_actual_mb"] for p in serie]
        ys_priv = [p["ram_privada_mb"] for p in serie]

        plt.figure(figsize=(9, 4.5), dpi=130)
        plt.plot(xs, ys_ws, label="RAM Working Set (MB)", color="#0284c7", linewidth=2.2)
        plt.plot(xs, ys_priv, label="RAM Privada Comprometida (MB)", color="#16a34a", linestyle="--", linewidth=1.6)
        plt.axhline(y=450, color="#ef4444", linestyle=":", linewidth=1.3, label="Umbral máximo objetivo (450 MB)")

        plt.title("Validación de Consumo de Memoria RAM — Asistente Ergonómico IA v3.0", fontsize=11, fontweight="bold")
        plt.xlabel("Tiempo de ejecución (segundos)")
        plt.ylabel("Memoria RAM (MB)")
        plt.grid(True, alpha=0.28)
        plt.legend(loc="best", fontsize=8.5)
        plt.tight_layout()
        plt.savefig(ruta)
        plt.close()
        print(f" -> Gráfica generada    : {os.path.basename(ruta)}")
        return os.path.basename(ruta)
    except Exception as e:
        print(f" -> [AVISO] No se pudo generar gráfica matplotlib: {e}")
        return ""


class _FakeLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _construir_landmarks(nariz, o_izq, o_der, h_izq, h_der):
    import mediapipe as mp

    lm_enum = mp.solutions.pose.PoseLandmark
    arr = [_FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    arr[lm_enum.NOSE.value] = _FakeLandmark(*nariz)
    arr[lm_enum.LEFT_EAR.value] = _FakeLandmark(*o_izq)
    arr[lm_enum.RIGHT_EAR.value] = _FakeLandmark(*o_der)
    arr[lm_enum.LEFT_SHOULDER.value] = _FakeLandmark(*h_izq)
    arr[lm_enum.RIGHT_SHOULDER.value] = _FakeLandmark(*h_der)
    return arr


def validar_postura_frontal_y_perfil() -> Dict[str, Any]:
    print("\n[5/5] VALIDACIÓN BIOMECÁNICA FRONTAL vs PERFIL")
    print("-" * 65)
    from asistente_ergonomico import AsistenteErgonomico

    asist = AsistenteErgonomico(model_complexity=0, mostrar_video_camara=True)

    def evaluar_caso(lm2d, lm3d, reiniciar_ema=True):
        if reiniciar_ema:
            asist._ema_inicializado = False
        m = asist.extraer_metricas_postura(lm2d, lm3d, aspect_ratio=0.75)
        ang, homb, dist, pen, motivos = asist.evaluar_metricas_postura(m)
        return m["vista_corporal"], round(ang, 1), round(homb, 1), pen, motivos

    # Fijar referencia frontal erguida (calibración)
    lm_f_ok = _construir_landmarks(
        nariz=(0.50, 0.44, 0.0),
        o_izq=(0.57, 0.43, 0.0),
        o_der=(0.43, 0.43, 0.0),
        h_izq=(0.68, 0.70, 0.0),
        h_der=(0.32, 0.70, 0.0),
    )
    w_f_ok = _construir_landmarks(
        nariz=(0.0, -0.22, -0.10),
        o_izq=(0.08, -0.21, -0.05),
        o_der=(-0.08, -0.21, -0.05),
        h_izq=(0.18, 0.0, 0.0),
        h_der=(-0.18, 0.0, 0.0),
    )
    m_base = asist.extraer_metricas_postura(lm_f_ok, w_f_ok, aspect_ratio=0.75)
    asist.baseline_ratio_cuello = m_base["ratio_cuello"]
    asist.baseline_ratio_nariz = m_base["ratio_nariz"]
    asist.baseline_pitch_nariz = m_base["pitch_nariz"]
    asist.baseline_distancia = m_base["dist_hombros"]
    asist.baseline_3d_sagital = m_base["ang_3d_sagital"]
    asist.calibrado = True

    resultados = {}

    # 1. Frontal erguido
    v, ang, h, pen, mot = evaluar_caso(lm_f_ok, w_f_ok)
    assert v == "FRONTAL" and pen == 0, f"Frontal erguido falló: {v}, {ang}, {mot}"
    resultados["frontal_erguido"] = {"vista": v, "angulo": ang, "penalizacion": pen, "ok": True}
    print(f" -> Frontal erguido            : vista={v} | cervical={ang}° | pen={pen} [APROBADA]")

    # 2. Frontal encorvado (cabeza baja hacia los hombros)
    lm_f_enc = _construir_landmarks(
        nariz=(0.50, 0.54, 0.0),
        o_izq=(0.57, 0.52, 0.0),
        o_der=(0.43, 0.52, 0.0),
        h_izq=(0.68, 0.70, 0.0),
        h_der=(0.32, 0.70, 0.0),
    )
    v, ang, h, pen, mot = evaluar_caso(lm_f_enc, w_f_ok)
    assert v == "FRONTAL" and "Cervical adelantada" in mot and ang > 28.0
    resultados["frontal_encorvado"] = {"vista": v, "angulo": ang, "motivos": mot, "ok": True}
    print(f" -> Frontal encorvado (2D)     : vista={v} | cervical={ang}° | {mot} [APROBADA]")

    # 3. Frontal mirando abajo (text neck / pitch nariz)
    lm_f_pitch = _construir_landmarks(
        nariz=(0.50, 0.52, 0.0),
        o_izq=(0.57, 0.44, 0.0),
        o_der=(0.43, 0.44, 0.0),
        h_izq=(0.68, 0.70, 0.0),
        h_der=(0.32, 0.70, 0.0),
    )
    v, ang, h, pen, mot = evaluar_caso(lm_f_pitch, w_f_ok)
    assert v == "FRONTAL" and "Cervical adelantada" in mot and ang > 28.0
    resultados["frontal_mirando_abajo"] = {"vista": v, "angulo": ang, "motivos": mot, "ok": True}
    print(f" -> Frontal text-neck (pitch)  : vista={v} | cervical={ang}° | {mot} [APROBADA]")

    # 4. Frontal cabeza adelantada hacia pantalla (profundidad 3D Z)
    w_f_z = _construir_landmarks(
        nariz=(0.0, -0.20, -0.22),
        o_izq=(0.08, -0.20, -0.15),
        o_der=(-0.08, -0.20, -0.15),
        h_izq=(0.18, 0.0, 0.0),
        h_der=(-0.18, 0.0, 0.0),
    )
    v, ang, h, pen, mot = evaluar_caso(lm_f_ok, w_f_z)
    assert v == "FRONTAL" and "Cervical adelantada" in mot and ang > 28.0
    resultados["frontal_adelantada_3d"] = {"vista": v, "angulo": ang, "motivos": mot, "ok": True}
    print(f" -> Frontal adelantada (3D Z)  : vista={v} | cervical={ang}° | {mot} [APROBADA]")

    # 5. Frontal muy cerca a la pantalla
    lm_f_cerca = _construir_landmarks(
        nariz=(0.50, 0.36, 0.0),
        o_izq=(0.59, 0.35, 0.0),
        o_der=(0.41, 0.35, 0.0),
        h_izq=(0.75, 0.70, 0.0),
        h_der=(0.25, 0.70, 0.0),
    )
    v, ang, h, pen, mot = evaluar_caso(lm_f_cerca, w_f_ok)
    assert v == "FRONTAL" and "Muy cerca a pantalla" in mot
    resultados["frontal_muy_cerca"] = {"vista": v, "motivos": mot, "ok": True}
    print(f" -> Frontal muy cerca          : vista={v} | {mot} [APROBADA]")

    # 6. Frontal hombros desnivelados
    lm_f_homb = _construir_landmarks(
        nariz=(0.50, 0.44, 0.0),
        o_izq=(0.57, 0.43, 0.0),
        o_der=(0.43, 0.43, 0.0),
        h_izq=(0.68, 0.65, 0.0),
        h_der=(0.32, 0.76, 0.0),
    )
    v, ang, h, pen, mot = evaluar_caso(lm_f_homb, w_f_ok)
    assert v == "FRONTAL" and "Hombros desnivelados" in mot
    resultados["frontal_hombros"] = {"vista": v, "hombros_deg": h, "motivos": mot, "ok": True}
    print(f" -> Frontal hombros desniv.    : vista={v} | hombros={h}° | {mot} [APROBADA]")

    # 7. Perfil erguido vs Perfil cabeza adelantada
    lm_p_ok = _construir_landmarks(
        nariz=(0.58, 0.43, 0.0),
        o_izq=(0.52, 0.43, 0.0),
        o_der=(0.50, 0.43, 0.0),
        h_izq=(0.48, 0.70, 0.0),
        h_der=(0.50, 0.70, 0.0),
    )
    lm_p_mal = _construir_landmarks(
        nariz=(0.70, 0.48, 0.0),
        o_izq=(0.64, 0.48, 0.0),
        o_der=(0.62, 0.48, 0.0),
        h_izq=(0.48, 0.70, 0.0),
        h_der=(0.50, 0.70, 0.0),
    )
    asist.baseline_cervical = 12.0
    v1, ang1, _, pen1, _ = evaluar_caso(lm_p_ok, None)
    v2, ang2, _, pen2, mot2 = evaluar_caso(lm_p_mal, None)
    assert v1 == "PERFIL" and pen1 == 0
    assert v2 == "PERFIL" and "Cervical adelantada" in mot2 and ang2 > 28.0
    resultados["perfil_erguido_vs_riesgo"] = {
        "vista": v2,
        "angulo_erguido": ang1,
        "angulo_riesgo": ang2,
        "ok": True,
    }
    print(f" -> Perfil erguido vs riesgo   : vista={v2} | {ang1}° -> {ang2}° ({mot2}) [APROBADA]")

    asist.cerrar()
    return resultados


def main():
    print("=" * 72)
    print(" VALIDACIÓN INTEGRAL DE CORRECCIONES DOCENTES - POSTUFIX v3.0")
    print("=" * 72)

    reporte = {
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "1_multicamara": validar_multicamara(),
        "2_manejo_conflictos": validar_manejo_conflictos(),
        "3_y_4_vistas_y_recursos_ram": validar_interfaces_y_ram(),
        "5_postura_frontal_y_perfil": validar_postura_frontal_y_perfil(),
    }

    ruta_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reporte_validacion_recursos.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(" TODAS LAS VALIDACIONES FINALIZARON EXITOSAMENTE")
    print(f" Reporte exportado a: {os.path.basename(ruta_json)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
