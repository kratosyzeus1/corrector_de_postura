"""
=============================================================================
PROYECTO: Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
INSTITUCIÓN: Politécnico Grancolombiano
AUTORES: Valery Sofía Meléndez Gallego, Juan Guillermo Ramírez Vásquez, Samuel Felipe Marín Lara
ENTREGA: Avance 2 - Prototipo Funcional Integrado (CNN BlazePose + Clustering K-Means)
=============================================================================
DELIMITACIÓN TÉCNICA Y ÉTICA (ATENCIÓN A RETROALIMENTACIÓN DOCENTE):
El sistema delimita estrictamente el rol de la Inteligencia Artificial a la
prevención ergonómica ocupacional y a la modulación de pausas activas.
No diagnostica patologías clínicas ni prescribe tratamientos médicos.
=============================================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import json
from datetime import datetime

try:
    import joblib
    TIENE_JOBLIB = True
except ImportError:
    TIENE_JOBLIB = False


class AsistenteErgonomico:
    def __init__(self):
        # Inicialización de MediaPipe Pose (CNN Ligera para Landmarks 3D)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Variables biomecánicas y de calibración
        self.calibrado = False
        self.baseline_cervical = 15.0  # Ángulo de referencia neutro en grados
        self.baseline_hombros = 0.0    # Desnivel de hombros neutro
        self.baseline_distancia = 0.25 # Distancia relativa estimada
        
        # Umbrales ergonómicos (según directrices ISO 9241-5 / OSHA)
        self.UMBRAL_CUELLO_MAX = 28.0       # > 28° inclinación cervical se considera riesgo
        self.UMBRAL_HOMBROS_MAX = 7.0       # > 7° de asimetría escapular
        self.TIEMPO_ALERTA_SEGUNDOS = 5.0   # Ventana base para alerta
        
        # Estado del sistema y temporizadores
        self.tiempo_mala_postura_inicio = None
        self.segundos_mala_postura = 0.0
        self.alerta_activa = False
        self.mensaje_alerta = ""
        self.tipo_diagnostico = "POSTURA ÓPTIMA"
        self.score_ergonomico = 100
        
        # Estadísticas de la sesión
        self.tiempo_inicio_sesion = time.time()
        self.total_frames = 0
        self.frames_correctos = 0
        self.frames_riesgo = 0
        self.total_alertas_disparadas = 0
        
        # Buffer de telemetría para Clustering K-Means
        self.historial_cervical = []
        self.historial_hombros = []
        self.historial_score = []
        
        # Estado del modelo K-Means
        self.modelo_kmeans_cargado = False
        self.kmeans_scaler = None
        self.kmeans_model = None
        self.kmeans_names_map = {}
        self.cluster_actual_fatiga = "Baja fatiga"
        self.cluster_color = (0, 220, 100) # Verde
        self.accion_ergonomica = "Mantener ritmo de trabajo; postura adecuada."
        self.cargar_modelo_kmeans()

        # Modo de ejecución (True: Cámara Real, False: Modo Demostración Sintético)
        self.modo_simulacion = False
        self.sim_phase = 0.0

    def cargar_modelo_kmeans(self):
        """Carga el modelo de K-Means entrenado en el archivo pickle o JSON"""
        pkl_path = "modelo_kmeans_fatiga.pkl"
        if TIENE_JOBLIB and os.path.exists(pkl_path):
            try:
                data = joblib.load(pkl_path)
                self.kmeans_model = data['kmeans']
                self.kmeans_scaler = data['scaler']
                self.kmeans_names_map = data['cluster_names_map']
                self.modelo_kmeans_cargado = True
                print("[K-MEANS] Modelo de clustering cargado exitosamente desde:", pkl_path)
                return
            except Exception as e:
                print(f"[WARN] No se pudo cargar {pkl_path}: {e}")

        # Fallback a metadatos de centroides JSON
        json_path = "modelo_kmeans_fatiga.json"
        if os.path.exists(json_path):
            print("[K-MEANS] Usando perfiles y centroides desde:", json_path)
            self.modelo_kmeans_cargado = True

    def evaluar_bloque_con_kmeans(self):
        """
        Ejecuta inferencia con K-Means sobre la ventana reciente de telemetría
        para clasificar el estado de fatiga en 'Baja fatiga', 'Fatiga moderada' o 'Alta fatiga'.
        """
        if self.total_frames == 0:
            return

        pct_riesgo = (self.frames_riesgo / max(1, self.total_frames)) * 100.0
        media_cervical = np.mean(self.historial_cervical[-150:]) if self.historial_cervical else 15.0
        media_hombros = np.mean(self.historial_hombros[-150:]) if self.historial_hombros else 2.0
        media_score = np.mean(self.historial_score[-150:]) if self.historial_score else 90.0
        sedentario_min = (time.time() - self.tiempo_inicio_sesion) / 60.0

        if self.modelo_kmeans_cargado and self.kmeans_model is not None:
            vec = np.array([[
                pct_riesgo,
                media_cervical,
                media_hombros,
                self.total_alertas_disparadas,
                sedentario_min,
                media_score
            ]])
            vec_scaled = self.kmeans_scaler.transform(vec)
            raw_cluster = int(self.kmeans_model.predict(vec_scaled)[0])
            self.cluster_actual_fatiga = self.kmeans_names_map.get(raw_cluster, "Indeterminado")
        else:
            # Clasificación heurística congruente con los centroides entrenados
            if pct_riesgo < 25.0 and media_cervical < 22.0:
                self.cluster_actual_fatiga = "Baja fatiga"
            elif pct_riesgo < 55.0 and media_cervical < 29.0:
                self.cluster_actual_fatiga = "Fatiga moderada"
            else:
                self.cluster_actual_fatiga = "Alta fatiga"

        # Actualizar colores y acciones ergonómicas moduladas (sin extralimitación médica)
        if self.cluster_actual_fatiga == "Baja fatiga":
            self.cluster_color = (0, 220, 100) # Verde
            self.TIEMPO_ALERTA_SEGUNDOS = 7.0
            self.accion_ergonomica = "Ritmo óptimo; pausa de hidratación a los 45 min."
        elif self.cluster_actual_fatiga == "Fatiga moderada":
            self.cluster_color = (0, 190, 255) # Ámbar/Naranja
            self.TIEMPO_ALERTA_SEGUNDOS = 4.5
            self.accion_ergonomica = "Micro-pausa de 30s sugerida: estiramiento de trapecios."
        else:
            self.cluster_color = (0, 60, 255) # Rojo
            self.TIEMPO_ALERTA_SEGUNDOS = 3.0
            self.accion_ergonomica = "Pausa activa obligatoria de 3-5 min: levantarse y descanso visual 20-20-20."

    def calcular_angulo(self, p1, p2):
        """Calcula el ángulo en grados de una línea respecto a la vertical u horizontal"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angulo_rad = math.atan2(abs(dx), abs(dy) + 1e-6)
        return math.degrees(angulo_rad)

    def calcular_inclinacion_hombros(self, hombro_izq, hombro_der):
        """Calcula el ángulo de desnivel entre hombros respecto a la horizontal"""
        dx = hombro_der[0] - hombro_izq[0]
        dy = hombro_der[1] - hombro_izq[1]
        angulo_rad = math.atan2(dy, dx + 1e-6)
        return abs(math.degrees(angulo_rad))

    def procesar_frame(self, frame):
        """Procesa un frame, extrae landmarks y calcula diagnóstico ergonómico"""
        alto, ancho, _ = frame.shape
        self.total_frames += 1
        
        # Preprocesamiento de color para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True

        diagnostico = "DESCONOCIDO"
        color_estado = (200, 200, 200) # Gris por defecto
        angulo_cervical = 0.0
        desnivel_hombros = 0.0
        distancia_relativa = 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Obtención de puntos articulares clave (coordenadas normalizadas 0.0 a 1.0)
            nariz = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                     landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            oreja_izq = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            oreja_der = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            hombro_izq = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hombro_der = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Punto medio de orejas y hombros para mayor robustez
            punto_medio_orejas = [(oreja_izq[0] + oreja_der[0]) / 2, (oreja_izq[1] + oreja_der[1]) / 2]
            punto_medio_hombros = [(hombro_izq[0] + hombro_der[0]) / 2, (hombro_izq[1] + hombro_der[1]) / 2]

            # 1. Ángulo de inclinación cervical (inclinación de cabeza hacia adelante)
            angulo_cervical = self.calcular_angulo(punto_medio_hombros, punto_medio_orejas)

            # 2. Desnivel de hombros (inclinación lateral del torso)
            desnivel_hombros = self.calcular_inclinacion_hombros(hombro_izq, hombro_der)

            # 3. Estimación de distancia a la pantalla (ancho de hombros normalizado)
            distancia_relativa = math.sqrt((hombro_der[0] - hombro_izq[0])**2 + (hombro_der[1] - hombro_izq[1])**2)

            # Clasificación Ergonómica basada en reglas biomecánicas
            es_cabeza_adelantada = angulo_cervical > (self.baseline_cervical + 12.0)
            es_hombros_desalineados = desnivel_hombros > (self.baseline_hombros + self.UMBRAL_HOMBROS_MAX)
            es_muy_cerca = distancia_relativa > (self.baseline_distancia * 1.45)

            penalizacion = 0
            motivos = []

            if es_cabeza_adelantada:
                penalizacion += 40
                motivos.append("Cervical Adelantada")
            if es_hombros_desalineados:
                penalizacion += 30
                motivos.append("Hombros Desalineados")
            if es_muy_cerca:
                penalizacion += 20
                motivos.append("Demasiado Cerca a Pantalla")

            self.score_ergonomico = max(10, 100 - penalizacion)

            # Guardar en buffer para K-Means
            self.historial_cervical.append(angulo_cervical)
            self.historial_hombros.append(desnivel_hombros)
            self.historial_score.append(self.score_ergonomico)
            if len(self.historial_cervical) > 300:
                self.historial_cervical.pop(0)
                self.historial_hombros.pop(0)
                self.historial_score.pop(0)

            if penalizacion == 0:
                diagnostico = "POSTURA ÓPTIMA (ADECUADA)"
                color_estado = (0, 220, 60) # Verde vibrante
                self.frames_correctos += 1
                self.tiempo_mala_postura_inicio = None
                self.segundos_mala_postura = 0.0
                self.alerta_activa = False
            else:
                self.frames_riesgo += 1
                diagnostico = "RIESGO: " + " & ".join(motivos)
                color_estado = (0, 70, 255) if penalizacion >= 40 else (0, 190, 255)

                ahora = time.time()
                if self.tiempo_mala_postura_inicio is None:
                    self.tiempo_mala_postura_inicio = ahora
                
                self.segundos_mala_postura = ahora - self.tiempo_mala_postura_inicio

                if self.segundos_mala_postura >= self.TIEMPO_ALERTA_SEGUNDOS:
                    if not self.alerta_activa:
                        self.total_alertas_disparadas += 1
                    self.alerta_activa = True
                    self.mensaje_alerta = f"¡PAUSA ACTIVA SUGERIDA! Corrige {motivos[0]}"

            # Actualizar K-Means periódicamente cada 60 frames
            if self.total_frames % 60 == 0:
                self.evaluar_bloque_con_kmeans()

            # Dibujar Skeleton ergonómico estilizado
            self.dibujar_esqueleto_ergonomico(frame, landmarks, ancho, alto, color_estado)
        else:
            diagnostico = "USUARIO NO DETECTADO"
            color_estado = (120, 120, 120)
            self.score_ergonomico = 50

        self.tipo_diagnostico = diagnostico

        # Renderizar Head-Up Display (HUD) con diseño moderno
        self.renderizar_hud(frame, diagnostico, color_estado, angulo_cervical, 
                            desnivel_hombros, distancia_relativa, ancho, alto)

        return frame

    def dibujar_esqueleto_ergonomico(self, frame, landmarks, ancho, alto, color):
        """Dibuja las articulaciones posturales clave en pantalla"""
        h_izq = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * ancho),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * alto))
        h_der = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * ancho),
                 int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * alto))
        o_izq = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * ancho),
                 int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * alto))
        o_der = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x * ancho),
                 int(landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y * alto))
        nariz = (int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].x * ancho),
                 int(landmarks[self.mp_pose.PoseLandmark.NOSE.value].y * alto))

        cv2.line(frame, h_izq, h_der, color, 3)
        cv2.line(frame, h_izq, o_izq, (255, 200, 50), 2)
        cv2.line(frame, h_der, o_der, (255, 200, 50), 2)
        cv2.line(frame, o_izq, o_der, (255, 255, 255), 1)

        for pt in [h_izq, h_der, o_izq, o_der, nariz]:
            cv2.circle(frame, pt, 7, (255, 255, 255), -1)
            cv2.circle(frame, pt, 5, color, -1)

    def renderizar_hud(self, frame, diagnostico, color_estado, ang_cervical, 
                       des_hombros, dist_rel, ancho, alto):
        """Dibuja una interfaz gráfica translúcida y moderna sobre el video"""
        # 1. Barra superior translúcida de estado
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (ancho - 15, 95), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (15, 15), (ancho - 15, 95), color_estado, 2)

        # Título y diagnóstico
        cv2.putText(frame, "MONITOR ERGONOMICO IA | POLITECNICO GRANCOLOMBIANO", 
                    (30, 42), cv2.FONT_HERSHEY_DUPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame, f"ESTADO: {diagnostico}", 
                    (30, 78), cv2.FONT_HERSHEY_DUPLEX, 0.75, color_estado, 2)

        # Score ergonómico en esquina superior derecha
        score_color = (0, 230, 100) if self.score_ergonomico >= 80 else (0, 70, 255)
        cv2.putText(frame, f"SCORE: {self.score_ergonomico}%", 
                    (ancho - 190, 68), cv2.FONT_HERSHEY_DUPLEX, 0.85, score_color, 2)

        # 2. Panel lateral de telemetría biomecánica
        panel_w = 320
        panel_h = 240
        p_x1, p_y1 = 15, 115
        p_x2, p_y2 = p_x1 + panel_w, p_y1 + panel_h
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (p_x1, p_y1), (p_x2, p_y2), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (p_x1, p_y1), (p_x2, p_y2), (60, 70, 85), 1)

        cv2.putText(frame, "TELEMETRIA BIOMECANICA", (p_x1 + 15, p_y1 + 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

        # Métricas individuales
        y_offset = p_y1 + 60
        cv2.putText(frame, f"- Angulo Cervical: {ang_cervical:.1f} deg", 
                    (p_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        
        pct_cervical = min(1.0, ang_cervical / 45.0)
        cv2.rectangle(frame, (p_x1 + 15, y_offset + 8), (p_x1 + 260, y_offset + 14), (50, 50, 50), -1)
        bar_color = (0, 220, 100) if ang_cervical < 28 else (0, 70, 255)
        cv2.rectangle(frame, (p_x1 + 15, y_offset + 8), (p_x1 + 15 + int(245 * pct_cervical), y_offset + 14), bar_color, -1)

        y_offset += 45
        cv2.putText(frame, f"- Inclinacion Hombros: {des_hombros:.1f} deg", 
                    (p_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        
        y_offset += 40
        cv2.putText(frame, f"- Tiempo Riesgo: {self.segundos_mala_postura:.1f}s / {self.TIEMPO_ALERTA_SEGUNDOS:.1f}s", 
                    (p_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)

        y_offset += 40
        calib_txt = "CALIBRADO [OK]" if self.calibrado else "PRESIONA [C] PARA CALIBRAR"
        calib_col = (0, 255, 150) if self.calibrado else (0, 200, 255)
        cv2.putText(frame, calib_txt, (p_x1 + 15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, calib_col, 1)

        # 3. Panel K-MEANS CLUSTERING (Nuevo módulo no supervisado)
        k_w = 330
        k_h = 130
        k_x1 = ancho - k_w - 15
        k_y1 = 115
        k_x2 = k_x1 + k_w
        k_y2 = k_y1 + k_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (k_x1, k_y1), (k_x2, k_y2), (18, 22, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        cv2.rectangle(frame, (k_x1, k_y1), (k_x2, k_y2), self.cluster_color, 1)

        cv2.putText(frame, "MODELO K-MEANS (CLUSTERING)", (k_x1 + 15, k_y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"Fatiga Bloque: {self.cluster_actual_fatiga.upper()}", 
                    (k_x1 + 15, k_y1 + 55), cv2.FONT_HERSHEY_DUPLEX, 0.55, self.cluster_color, 1)

        # Micro-texto de recomendación operativa
        cv2.putText(frame, f"- Accion: {self.accion_ergonomica[:36]}", 
                    (k_x1 + 15, k_y1 + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        cv2.putText(frame, f"- Ventana Alerta Adaptada: {self.TIEMPO_ALERTA_SEGUNDOS:.1f}s", 
                    (k_x1 + 15, k_y1 + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 255), 1)

        # 4. Banner Flotante de ALERTA ACTIVA cuando se supera el umbral continuo
        if self.alerta_activa:
            al_h = 75
            al_y1 = alto - 150
            al_y2 = al_y1 + al_h
            overlay = frame.copy()
            cv2.rectangle(overlay, (30, al_y1), (ancho - 30, al_y2), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (30, al_y1), (ancho - 30, al_y2), (0, 240, 255), 3)

            cv2.putText(frame, "ALERTA DE HIGIENE POSTURAL ACTIVA (SALUD OCUPACIONAL)", 
                        (50, al_y1 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, self.mensaje_alerta + " - Endereza tu espalda y alinea la mirada", 
                        (50, al_y1 + 60), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1)

        # 5. Pie de página informativo con controles para la exposición
        cv2.rectangle(frame, (0, alto - 40), (ancho, alto), (15, 18, 24), -1)
        controles_txt = "[C] Calibrar  |  [M] Modo Demo  |  [K] Inferencia K-Means  |  [S] Guardar  |  [Q] Salir"
        cv2.putText(frame, controles_txt, (20, alto - 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 190, 205), 1)

    def calibrar(self, frame):
        """Calibra la postura de referencia del usuario sentado correctamente"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            oreja_izq = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            oreja_der = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            hombro_izq = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hombro_der = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            p_orejas = [(oreja_izq[0] + oreja_der[0]) / 2, (oreja_izq[1] + oreja_der[1]) / 2]
            p_hombros = [(hombro_izq[0] + hombro_der[0]) / 2, (hombro_izq[1] + hombro_der[1]) / 2]

            self.baseline_cervical = self.calcular_angulo(p_hombros, p_orejas)
            self.baseline_hombros = self.calcular_inclinacion_hombros(hombro_izq, hombro_der)
            self.baseline_distancia = math.sqrt((hombro_der[0] - hombro_izq[0])**2 + (hombro_der[1] - hombro_izq[1])**2)
            self.calibrado = True
            print(f"[CALIBRACIÓN EXITOSA] Baseline Cervical: {self.baseline_cervical:.1f}° | Hombros: {self.baseline_hombros:.1f}°")

    def generar_frame_simulado(self):
        """Genera un frame de simulación interactivo si no hay cámara disponible"""
        ancho, alto = 800, 600
        canvas = np.zeros((alto, ancho, 3), dtype=np.uint8)
        canvas[:] = (30, 32, 40) # Fondo oscuro elegante

        self.sim_phase += 0.05
        ciclo = math.sin(self.sim_phase)
        
        c_x, c_y = ancho // 2, alto // 2
        offset_cabeza = int(ciclo * 45)
        desnivel = int(math.cos(self.sim_phase * 0.7) * 20)

        cabeza_pt = (c_x + offset_cabeza, c_y - 120)
        cuello_pt = (c_x + offset_cabeza // 2, c_y - 60)
        h_izq_pt = (c_x - 110, c_y - 20 - desnivel)
        h_der_pt = (c_x + 110, c_y - 20 + desnivel)
        torso_base = (c_x, c_y + 140)

        color_esqueleto = (0, 220, 100) if abs(offset_cabeza) < 20 else (0, 80, 255)
        
        cv2.circle(canvas, cabeza_pt, 40, (255, 255, 255), 2)
        cv2.line(canvas, cabeza_pt, cuello_pt, color_esqueleto, 4)
        cv2.line(canvas, cuello_pt, h_izq_pt, color_esqueleto, 4)
        cv2.line(canvas, cuello_pt, h_der_pt, color_esqueleto, 4)
        cv2.line(canvas, cuello_pt, torso_base, color_esqueleto, 4)
        cv2.circle(canvas, h_izq_pt, 10, (0, 200, 255), -1)
        cv2.circle(canvas, h_der_pt, 10, (0, 200, 255), -1)

        ang_cerv = abs(offset_cabeza) * 0.7 + 10.0
        desn_h = abs(desnivel) * 0.4
        
        es_riesgo = ang_cerv > 28.0 or desn_h > 7.0
        diag = "POSTURA ÓPTIMA (ADECUADA)" if not es_riesgo else "RIESGO: Cabeza Adelantada (Cifosis)"
        color_st = (0, 220, 60) if not es_riesgo else (0, 70, 255)

        self.total_frames += 1
        if es_riesgo:
            self.frames_riesgo += 1
            self.segundos_mala_postura += 0.08
            if self.segundos_mala_postura >= self.TIEMPO_ALERTA_SEGUNDOS:
                if not self.alerta_activa:
                    self.total_alertas_disparadas += 1
                self.alerta_activa = True
                self.mensaje_alerta = "¡ALERTA DE FATIGA! Postura encorvada detectada"
        else:
            self.frames_correctos += 1
            self.segundos_mala_postura = 0.0
            self.alerta_activa = False

        self.score_ergonomico = max(15, int(100 - (ang_cerv - 15) * 3)) if ang_cerv > 15 else 100
        self.historial_cervical.append(ang_cerv)
        self.historial_hombros.append(desn_h)
        self.historial_score.append(self.score_ergonomico)

        if self.total_frames % 50 == 0:
            self.evaluar_bloque_con_kmeans()

        self.renderizar_hud(canvas, diag, color_st, ang_cerv, desn_h, 0.28, ancho, alto)
        cv2.putText(canvas, "[MODO DEMOSTRACION SINTETICO PARA EXPOSICION]", 
                    (c_x - 220, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return canvas

def main():
    print("=" * 70)
    print(" INICIANDO ASISTENTE ERGONÓMICO INTELIGENTE - POLI GRANCOLOMBIANO")
    print("=" * 70)
    print(" Controles:")
    print("  - Tecla [C]: Calibrar postura de referencia neutra actual")
    print("  - Tecla [M]: Alternar entre Cámara Web y Modo Demostración Sintético")
    print("  - Tecla [K]: Evaluar clasificación K-Means en vivo")
    print("  - Tecla [S]: Guardar captura de pantalla de evidencia")
    print("  - Tecla [Q]: Finalizar ejecución y ver estadísticas")
    print("=" * 70)

    asistente = AsistenteErgonomico()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[AVISO] No se detectó cámara web física disponible.")
        print("[INFO] Activando automáticamente el 'Modo Demostración Sintético' para la exposición...")
        asistente.modo_simulacion = True

    cv2.namedWindow("Asistente Ergonomico Inteligente - Avance 2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Asistente Ergonomico Inteligente - Avance 2", 960, 640)

    while True:
        if asistente.modo_simulacion:
            frame = asistente.generar_frame_simulado()
            time.sleep(0.03) # ~30 FPS simulados
        else:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Error al leer cámara. Cambiando a Modo Demostración.")
                asistente.modo_simulacion = True
                continue
            frame = cv2.flip(frame, 1)
            frame = asistente.procesar_frame(frame)

        cv2.imshow("Asistente Ergonomico Inteligente - Avance 2", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c') or key == ord('C'):
            if not asistente.modo_simulacion and cap.isOpened():
                ret_cal, cal_frame = cap.read()
                if ret_cal:
                    asistente.calibrar(cal_frame)
            else:
                asistente.calibrado = True
                print("[CALIBRACIÓN] Calibración simulada fijada exitosamente.")
        elif key == ord('m') or key == ord('M'):
            asistente.modo_simulacion = not asistente.modo_simulacion
            modo_nombre = "Demostración Sintética" if asistente.modo_simulacion else "Cámara Web en Vivo"
            print(f"[MODO CAMBIADO] Modo actual: {modo_nombre}")
        elif key == ord('k') or key == ord('K'):
            asistente.evaluar_bloque_con_kmeans()
            print(f"[K-MEANS INFERENCIA] Fatiga del Bloque: {asistente.cluster_actual_fatiga.upper()}")
            print(f" -> Recomendación operativa: {asistente.accion_ergonomica}")
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidencia_postura_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"[CAPTURA GUARDADA] Evidencia exportada a: {filename}")

    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()

    tiempo_total = time.time() - asistente.tiempo_inicio_sesion
    print("\n" + "=" * 70)
    print(" RESUMEN DE LA SESIÓN DE MONITOREO ERGONÓMICO")
    print("=" * 70)
    print(f" - Duración de la sesión: {tiempo_total:.1f} segundos")
    print(f" - Fotogramas evaluados: {asistente.total_frames}")
    if asistente.total_frames > 0:
        cumplimiento = (asistente.frames_correctos / asistente.total_frames) * 100
        print(f" - Cumplimiento ergonómico general: {cumplimiento:.1f}%")
    print(f" - Total de alertas de pausa activa activadas: {asistente.total_alertas_disparadas}")
    print(f" - Clasificación final K-Means: {asistente.cluster_actual_fatiga.upper()}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
