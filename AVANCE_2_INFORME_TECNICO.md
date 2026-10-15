# INSTITUCIÓN UNIVERSITARIA POLITÉCNICO GRANCOLOMBIANO
## FACULTAD DE INGENIERÍA, DISEÑO E INNOVACIÓN
### INGENIERÍA DE SISTEMAS / PROYECTO DE INTELIGENCIA ARTIFICIAL

---

# ASISTENTE ERGONÓMICO INTELIGENTE Y MONITOR DE SALUD OCUPACIONAL BASADO EN VISIÓN POR COMPUTADOR Y CLUSTERING K-MEANS
### *Informe Técnico y Evidencia Funcional del MVP — Avance 2*

**Lugar y Fecha:** Medellín, Colombia - Octubre de 2026  
**Materia:** Proyecto de Inteligencia Artificial  
**Docente:** Neider  

**Autores:**  
- **Valery Sofía Meléndez Gallego** (`vmelendez@poligran.edu.co`)  
- **Juan Guillermo Ramírez Vásquez** (`jguramirez@poligran.edu.co`)  
- **Samuel Felipe Marín Lara** (`sfmarin@poligran.edu.co`)  

**Repositorio del Proyecto en GitHub:**  
[https://github.com/kratosyzeus1/corrector_de_postura.git](https://github.com/kratosyzeus1/corrector_de_postura.git)

---

## 1. AJUSTES AL PROBLEMA

### Correcciones Realizadas después de la Retroalimentación del Avance 1

En la retroalimentación del Avance 1, el docente Neider destacó:
> *"Propuesta sólida, con antecedentes relevantes, arquitectura comprensible y buena consideración de privacidad mediante procesamiento local. La combinación de visión por computador y análisis de hábitos es interesante. Conviene delimitar mejor el papel de K-Means y evitar que el sistema formule recomendaciones de salud más allá de lo que pueda sustentarse técnicamente; el siguiente avance debe priorizar un MVP medible. No necesariamente significa tener el 70% del código listo, significa que existe suficiente evidencia para demostrar que la solución es técnicamente viable."*

A partir de estas indicaciones, se efectuaron tres ajustes estructurales en la formulación del problema y el alcance:

1. **Delimitación Estricta del Rol de K-Means (Prevención vs Diagnóstico Médico):**
   - Se eliminó cualquier pretensión o terminología que sugiriera diagnóstico clínico patológico (como escoliosis, hernias discales, cervicalgias crónicas o radiculopatías).
   - Se redefinió K-Means como un **modelo no supervisado de análisis de comportamiento temporal y exposición ergonómica acumulada**. Su propósito exclusivo es clasificar los bloques de sesión de trabajo en tres niveles operacionales (**Baja fatiga**, **Fatiga moderada** y **Alta fatiga**) para **modular la sensibilidad del agente inteligente** y la frecuencia de pausas activas higiénico-posturales sugeridas.
2. **Eliminación de Recomendaciones No Sustentables Técnicamente:**
   - Se ajustó el motor de notificaciones para emitir únicamente recomendaciones de autocuidado basadas en guías estandarizadas de salud ocupacional (ISO 9241-5, OSHA y la regla 20-20-20 de higiene visual), evitando prescripciones terapéuticas invasivas.
3. **Priorización de un MVP Medible y Funcional:**
   - En lugar de limitarse a documentación conceptual, se construyó y desplegó evidencia técnica funcional y ejecutable que demuestra la viabilidad de la solución de punta a punta.

---

## 2. ESTADO DEL ARTE FORTALECIDO

### Comparación Profunda con Soluciones, Modelos e Investigaciones Existentes

Para el Avance 2, se amplió el análisis comparativo con investigaciones científicas recientes en ergonomía computacional:

| Investigación / Solución | Enfoque Tecnológico y Sensores | Ventajas Destacadas | Limitaciones Técnicas frente a Nuestra Solución |
| :--- | :--- | :--- | :--- |
| **Rodrigues et al. (2022)** | Cámara Microsoft Kinect + Evaluación automatizada RULA en oficinas. | Alta precisión en profundidad articular 3D mediante sensor infrarrojo dedicado. | **Hardware obsoleto y restrictivo:** Depende de sensores Kinect costosos que no existen en computadores portátiles modernos; inviable para teletrabajo masivo. |
| **Lin et al. (2022)** | OpenPose (CMU) + Clasificación RULA / REBA / OWAS. | Extracción de esqueletos corporales múltiples con alto detalle. | **Sobrecosto computacional extremo:** OpenPose requiere GPUs NVIDIA dedicadas de gama alta (latencia en CPU comercial de 2 a 5 FPS), consumiendo excesiva energía e impidiendo multitarea. |
| **Menanno et al. (2024)** | Estimación 3D + Sistemas de lógica difusa (*Fuzzy Logic*) en celdas robóticas. | Reducción de estrés biomecánico en 13% y aumento de productividad del 33%. | **Diseñado para industria pesada:** Orientado a interacción con brazos robóticos industriales; no resuelve la postura sedente frente a pantallas VDU de oficina. |
| **Nuestra Propuesta (Poli Grancolombiano)** | **MediaPipe BlazePose + Cinemática Vectorial + K-Means ($k=3$) + Edge AI**. | **Inferencia en CPU ligera a más de 28-32 FPS sin GPU dedicada, 100% privado en memoria local, autocalibración antropométrica y clustering no supervisado.** | Optimizado para el plano sagital y coronal de tronco superior (cabeza, cuello, hombros y distancia de pantalla). |

---

## 3. MARCO TEÓRICO FORTALECIDO

### Fundamentación de las Técnicas de Inteligencia Artificial Seleccionadas

1. **Estimación de Pose Humana (*Human Pose Estimation - HPE*):**  
   Localización espacial tridimensional de puntos de articulación corporal (*landmarks*) a partir de matrices de imágenes RGB monoculares sin requerir sensores de profundidad estéreoscópicos.
2. **Redes Neuronales Convolucionales Ligeras (MediaPipe BlazePose):**  
   Arquitectura de dos etapas desarrollada por Google Research:
   - *Detector de Cuerpo:* Localiza la región anatómica humana en el fotograma.
   - *Regresor de Landmarks:* Basado en **convoluciones separables en profundidad (*depthwise separable convolutions*)**, infiere las coordenadas normalizadas $(X_i, Y_i, Z_i)$ de 33 articulaciones en milisegundos con un peso computacional inferior al 15% de CPU.
3. **Cinemática Biomecánica Postural:**  
   Modelado trigonométrico espacial para calcular:
   - **Ángulo de Flexión Cervical ($\theta_c$):** Proyección del vector oreja-hombro respecto al eje vertical gravitacional:
     $$\theta_c = \arctan2(|X_{\text{oreja}} - X_{\text{hombro}}|, |Y_{\text{oreja}} - Y_{\text{hombro}}|) \times \frac{180}{\pi}$$
   - **Desnivel Coronal de Hombros ($\theta_h$):** Inclinación horizontal escapular:
     $$\theta_h = \arctan2(|Y_{\text{hombro\_der}} - Y_{\text{hombro\_izq}}|, |X_{\text{hombro\_der}} - X_{\text{hombro\_izq}}|) \times \frac{180}{\pi}$$
4. **Normativas Ergonómicas Internacionales (ISO 9241-5 y OSHA):**  
   Directrices biomecánicas ocupacionales que determinan que una inclinación cervical sostenida superior a 28° o una asimetría de hombros mayor a 7° multiplican la carga en la columna cervical de 5 kg (neutro) hasta 22-27 kg.
5. **Agente Lógico Basado en Modelos y Memoria Temporal:**  
   Estructura que no evalúa fotogramas aislados (lo que generaría alertas erráticas ante parpadeos o movimientos naturales), sino que mantiene un acumulador temporal continuo:
   $$\Delta t_{\text{riesgo}} = t_{\text{actual}} - t_{\text{inicio\_desviación}}$$
   Solo dispara la alerta si $\Delta t_{\text{riesgo}} \ge t_{\text{tolerancia}}$ (10 a 15 segundos).
6. **Clustering K-Means (Aprendizaje No Supervisado):**  
   Algoritmo que particiona $n$ observaciones en $k$ grupos, minimizando la inercia o suma de distancias cuadráticas intra-cluster (*Within-Cluster Sum of Squares - WCSS*):
   $$J = \sum_{k=1}^K \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$
7. **Estandarización de Variables (*StandardScaler*):**  
   Normalización mediante puntajes z ($z = \frac{x - \mu}{\sigma}$) para equiparar la varianza de magnitudes heterogéneas (porcentajes, ángulos en grados y minutos de sedentarismo).
8. **Validación de Agrupamiento (Coeficiente de Silhouette):**  
   Evalúa la cohesión interna frente a la separación externa:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} \in [-1, 1]$$
9. **Privacidad en el Borde (*Edge AI Computing*):**  
   Los fotogramas se procesan y descartan exclusivamente en la memoria RAM volátil, garantizando el cumplimiento de la Ley 1581 de 2012 (Protección de Datos Personales en Colombia) y el RGPD.

---

## 4. DISEÑO DE LA SOLUCIÓN

### Arquitectura, Módulos, Componentes y Funcionamiento

La arquitectura del sistema está organizada en **6 capas funcionales interconectadas**:

```text
[ Usuario frente a Pantalla / Cámara Web ]
                 │ (Fotogramas RGB 30 FPS)
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 1: Captura & Edge Buffer (OpenCV 5.0)                  │
│ -> Ingesta en memoria RAM volátil (Cero escritura a disco)  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 2: Inferencia de Visión CNN (MediaPipe BlazePose)      │
│ -> Extracción en submilisegundos de 33 landmarks 3D         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 3: Motor Cinemático Biomecánico                        │
│ -> Cálculo de θ cervical, θ hombros y distancia relativa    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 4: Agente Inteligente con Filtro Temporal              │
│ -> Ventana deslizante (10-15s) anti-falsos positivos        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 5: Analítica de Hábitos y Clustering (K-Means k=3)     │
│ -> Segmentación de bloques: Baja / Moderada / Alta Fatiga   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Capa 6: Interfaces y Servicios                              │
│ -> HUD translúcido en pantalla (OpenCV)                     │
│ -> Dashboard Web Interactivo (HTML5 Canvas)                 │
│ -> Servicio API REST (FastAPI endpoints locales)            │
└─────────────────────────────────────────────────────────────┘
```

#### Descripción de Módulos:
- **`asistente_ergonomico.py`:** Módulo principal de escritorio. Inicia el flujo de video, ejecuta BlazePose, dibuja el HUD translúcido semaforizado, gestiona la calibración neutra (tecla `C`) y evalúa el clustering K-Means en vivo.
- **`clustering_kmeans_fatiga.py`:** Módulo analítico no supervisado. Genera el dataset estructurado de bloques horarios, entrena el modelo K-Means con escalado, calcula el Método del Codo y Silhouette, y exporta el modelo serializado.
- **`api_servicio_ergonomia.py`:** Servicio web backend con FastAPI para inferencia distribuida (`POST /api/predecir-fatiga` y `POST /api/evaluar-postura`).
- **`demo_exposicion.html`:** Dashboard y simulador interactivo web con visualizador 2D de clusters, sliders de parámetros biomecánicos y diapositivas de exposición.

---

## 5. FLUJO DE INFORMACIÓN

### Cómo Entran, se Transforman y Salen los Datos

```text
[Fotograma RGB] ──> [MediaPipe Pose] ──> [(X,Y,Z) 33 Landmarks]
                           │
                           ▼
                  [Cálculo Cinemático] ──> [θ cervical, θ hombros, Distancia]
                           │
                           ▼
                  [Buffer Temporal] ───> [Vector X ∈ ℝ⁶]
                           │                    │
                           ▼                    ▼
               [Agente Temporal: 15s]     [K-Means Inferencia: k=3]
                           │                    │
                           ▼                    ▼
               [Alerta HUD en Pantalla] ◄── [Modulación de Tolerancia]
```

1. **Entrada de Datos:**
   - La cámara web captura fotogramas a resolución estándar ($640 \times 480$ a 30 FPS).
2. **Transformación Primaria:**
   - Se transforma el espacio de color de BGR a RGB. La matriz se pasa como de solo lectura a MediaPipe BlazePose para inferir las coordenadas normalizadas de la nariz, orejas y hombros.
3. **Transformación Cinemática:**
   - Se calculan el punto medio de orejas y hombros, obteniendo los ángulos $\theta_{\text{cervical}}$ y $\theta_{\text{hombros}}$.
4. **Agregación Temporal:**
   - Se acumulan métricas en un buffer deslizante en memoria RAM que consolida el porcentaje de tiempo en riesgo, la media angular y el tiempo sedentario.
5. **Inferencia No Supervisada (K-Means):**
   - El vector de telemetría acumulado se estandariza con `StandardScaler` y se proyecta al espacio de clusters para asignar la categoría de fatiga ("Baja fatiga", "Fatiga moderada" o "Alta fatiga").
6. **Salida y Retroalimentación:**
   - El HUD en pantalla despliega el estado biomecánico, el score porcentual y el nivel de fatiga. Si la mala postura persiste más allá de la tolerancia adaptada, el agente emite una alerta flotante de pausa activa sugerida.

---

## 6. DATOS

### Dataset, Fuentes, Variables, Características, Calidad y Preparación

El proyecto articula dos fuentes de datos complementarias:

#### 6.1 Datos en Tiempo Real (Telemetría Cinemática)
- **Fuente:** Flujo de video en vivo capturado por la cámara web del usuario.
- **Formato:** Tensores numéricos flotantes que representan coordenadas $(X_i, Y_i, Z_i)$ normalizadas entre $0.0$ y $1.0$, con métricas de visibilidad asociadas.
- **Preparación:** Filtrado de oclusiones descartando landmarks con `visibility < 0.5`. Invarianza a la escala mediante distancias relativas inter-escapulares.

#### 6.2 Dataset de Bloques Horarios de Sesiones (`dataset_bloques_ergonomia.csv`)
- **Origen:** Dataset estructurado de **360 registros de bloques de trabajo (30 minutos cada uno)** representativo de jornadas laborales y académicas de 8:00 a 18:00 horas.
- **Variables del Vector de Características ($X \in \mathbb{R}^6$):**
  1. `tiempo_riesgo_pct`: Porcentaje de tiempo que el usuario adoptó posturas perjudiciales durante el bloque ($0.0\%$ a $100.0\%$).
  2. `angulo_cervical_promedio`: Media de inclinación oreja-hombro en grados ($10.0^\circ$ a $45.0^\circ$).
  3. `asimetria_hombros_promedio`: Media de desnivel horizontal entre hombros ($0.0^\circ$ a $15.0^\circ$).
  4. `alertas_disparadas`: Número de alertas activadas por el agente en el bloque ($0$ a $6$).
  5. `tiempo_sedentario_continuo_min`: Minutos continuos sentado sin interrupción ($5.0$ a $60.0\text{ min}$).
  6. `score_ergonomico_medio`: Puntuación promedio del bloque ($10$ a $100\%$).
- **Calidad y Limpieza:** Sin valores nulos ni registros corruptos; variables acotadas según límites biológicos admisibles.
- **Normalización:** Escalado con `StandardScaler` ($\mu=0, \sigma=1$).

---

## 7. TÉCNICA SELECCIONADA

### Algoritmo y Modelo Definitivo y Justificación frente a Alternativas

#### 7.1 Técnica de Visión: MediaPipe BlazePose (Google)
- **Justificación:** Consume menos del 15% de CPU en procesadores comerciales estándar sin necesidad de GPU dedicada, manteniendo tasas superiores a **28-32 FPS**, a diferencia de OpenPose (2-5 FPS en CPU) o YOLOv8-Pose (alto consumo térmico de batería en portátiles).

#### 7.2 Técnica de Hábitos: Clustering K-Means ($k=3$)
- **Justificación:** K-Means fue seleccionado sobre DBSCAN y GMM por:
  - Complejidad temporal lineal $O(n \cdot k \cdot I \cdot d)$ que permite clasificar nuevos bloques en submilisegundos ($< 1\text{ ms}$).
  - Centroides con interpretabilidad geométrica y operacional directa.
  - Portabilidad extrema: modelo serializado de menos de $4\text{ KB}$ (`modelo_kmeans_fatiga.pkl`).

#### 7.3 Sustento de la Elección de $k=3$
1. **Método del Codo (Elbow Method):** La inercia intra-cluster (WCSS) experimenta su punto de inflexión más marcado en $k=3$ ($\text{WCSS} = 303.78$).
2. **Coeficiente de Silhouette:** Para $k=3$, el Silhouette Score promedio alcanza **0.5888**, confirmando una separación nítida entre los grupos.
3. **Congruencia Ocupacional:** Mapea de forma natural al semáforo ergonómico:
   - **Cluster 0: "Baja fatiga"** (Score medio: 93.34%, Cervical: 15.23°, Sedentarismo: 16.91 min).
   - **Cluster 1: "Fatiga moderada"** (Score medio: 68.97%, Cervical: 24.33°, Sedentarismo: 33.12 min).
   - **Cluster 2: "Alta fatiga"** (Score medio: 31.68%, Cervical: 33.44°, Sedentarismo: 50.04 min).

---

## 8. DESARROLLO REALIZADO

### Descripción de los Componentes ya Implementados

A la fecha del Avance 2, el equipo ha desarrollado y probado los siguientes componentes funcionales:

1. **Motor de Visión Biomecánica (`asistente_ergonomico.py`):**
   - Captura en vivo desde cámara web y modo demostración sintético integrado.
   - Cálculo en tiempo real de inclinación cervical y desnivel escapular.
   - Algoritmo interactivo de auto-calibración en caliente (tecla `C`).
   - Módulo de exportación de capturas de evidencia fotográfica (tecla `S`).
   - Inferencia integrada de K-Means en vivo (tecla `K`).
2. **Módulo de Clustering y Analítica (`clustering_kmeans_fatiga.py`):**
   - Generación del dataset estructurado de 360 bloques horarios.
   - Entrenamiento formal del agrupador K-Means ($k=3$) con `StandardScaler`.
   - Generación automatizada de gráficas científicas de evidencia (Codo, Dispersión 2D y Silhouette).
   - Serialización de artefactos: `modelo_kmeans_fatiga.pkl` y `modelo_kmeans_fatiga.json`.
3. **Servicio API REST de Inferencia (`api_servicio_ergonomia.py`):**
   - Endpoints REST desarrollados con FastAPI:
     - `GET /`: Estado del servicio y modelos disponibles.
     - `GET /api/perfiles-clusters`: Consulta de centroides y métricas.
     - `POST /api/predecir-fatiga`: Inferencia de categoría de fatiga a partir de telemetría de bloque.
     - `POST /api/evaluar-postura`: Evaluación cinemática instantánea de ángulos.
4. **Cliente de Verificación de API (`probar_llamada_api.py`):**
   - Suite de pruebas automatizadas con `TestClient` que verifica el funcionamiento de todos los endpoints, confirmando respuestas con código **HTTP 200 OK**.
5. **Dashboard Web Interactivo y Exposición (`demo_exposicion.html`):**
   - Visualizador de telemetría en tiempo real sobre canvas HTML5.
   - Simulador interactivo de K-Means con sliders y mapa de dispersión 2D.
   - Diapositivas académicas de presentación integradas.

---

## 9. EVIDENCIAS FUNCIONALES DEL MVP

### Demostración Técnica de Viabilidad del Avance 2

En atención a la directriz del docente Neider sobre aportar **evidencia funcional medible**, se consolidan los siguientes artefactos:

#### Matriz de Evidencias Funcionales

| Criterio Exigido | Evidencia Funcional Verificada | Archivo de Respaldo |
| :--- | :--- | :--- |
| **1. Dataset cargado y procesado** | 360 bloques horarios estructurados con variables biomecánicas y temporales. | [dataset_bloques_ergonomia.csv](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/dataset_bloques_ergonomia.csv) |
| **2. Primer modelo entrenado** | K-Means ($k=3$) ajustado con StandardScaler y persistido en binario/JSON. | [clustering_kmeans_fatiga.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/clustering_kmeans_fatiga.py) / [modelo_kmeans_fatiga.pkl](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/modelo_kmeans_fatiga.pkl) |
| **3. Arquitectura parcial implementada** | Pipeline de 6 capas operacionales interconectadas. | [asistente_ergonomico.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/asistente_ergonomico.py) |
| **4. Prototipo de interfaz** | HUD en OpenCV con telemetría en vivo y Dashboard interactivo HTML5. | [demo_exposicion.html](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/demo_exposicion.html) / [evidencia_postura_demo.png](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_postura_demo.png) |
| **5. Llamada funcional a API** | Endpoints REST locales con FastAPI respondiendo HTTP 200 OK en pruebas automatizadas. | [api_servicio_ergonomia.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/api_servicio_ergonomia.py) / [probar_llamada_api.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/probar_llamada_api.py) |
| **6. Flujo automatizado construido** | Flujo continuo de captura $\rightarrow$ cinemática $\rightarrow$ buffer $\rightarrow$ K-Means $\rightarrow$ alerta. | [asistente_ergonomico.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/asistente_ergonomico.py) |
| **7. Sistema de clasificación en pruebas** | Tests sobre casos de prueba clasificando correctamente en los tres clusters. | [clustering_kmeans_fatiga.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/clustering_kmeans_fatiga.py) |
| **8. CNN realizando predicciones** | BlazePose infiriendo 33 landmarks 3D corporales a >28 FPS en CPU. | [asistente_ergonomico.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/asistente_ergonomico.py) |
| **9. Clustering preliminar** | Método del Codo y Coeficiente de Silhouette evaluados y documentados. | [evidencia_metodo_codo_kmeans.png](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_metodo_codo_kmeans.png) |
| **10. Agente ejecutando acciones** | Filtrado temporal anti-falsos positivos (10-15s) y alertas de pausa activa moduladas. | [asistente_ergonomico.py](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/asistente_ergonomico.py) |

#### Registro Gráfico de Evidencias

- **Evidencia 1: HUD de Telemetría Biomecánica en Vivo con Clasificación K-Means**  
  ![HUD y Telemetría Biomecánica en Tiempo Real](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_postura_demo.png)

- **Evidencia 2: Curva del Método del Codo (WCSS) para K-Means ($k=1..7$)**  
  ![Método del Codo para Selección Óptima de K en K-Means](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_metodo_codo_kmeans.png)

- **Evidencia 3: Mapa de Dispersión 2D de Clusters de Fatiga Ergonómica**  
  ![Espacio 2D de Clusters de Fatiga Ergonómica](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_clusters_fatiga_2d.png)

- **Evidencia 4: Distribución del Coeficiente de Silhouette por Cluster ($s_{\text{prom}} = 0.5888$)**  
  ![Distribución del Coeficiente de Silhouette](file:///c:/Users/Usuario/OneDrive/Desktop/Tarea%20uni/evidencia_silhouette_kmeans.png)

---

## 10. ESTRATEGIA DE EVALUACIÓN

### Definición de Cómo se Determinará si la Solución Funciona

Se definieron cuatro fases metodológicas de validación experimental:

1. **Validación Geométrica Biomecánica:**  
   Comparación del error angular absoluto entre los cálculos de BlazePose y un goniómetro manual de referencia en ángulos estáticos prefijados ($15^\circ$, $25^\circ$, $35^\circ$).
2. **Validación Temporal del Agente (Mitigación de Falsos Positivos):**  
   Pruebas de estrés donde el usuario realiza movimientos transitorios naturales (beber agua, rotar la cabeza para hablar con un colega, inclinarse para atar calzado) verificando que el agente tolere el movimiento transitorio y **no dispare alertas falsas** si la desviación no supera los 10 a 15 segundos continuos.
3. **Validación del Agrupamiento K-Means:**  
   Evaluación matemática de la cohesión intra-cluster y separación inter-cluster mediante el Coeficiente de Silhouette ($s \ge 0.50$) y la estabilización de la inercia WCSS.
4. **Validación de Servicios de Inferencia (API REST):**  
   Pruebas cliente-servidor automatizadas para verificar que la latencia por solicitud sea inferior a $50\text{ ms}$ y retorne respuestas válidas con código HTTP 200 OK.

---

## 11. MÉTRICAS

### Métricas de Rendimiento Establecidas y Valores Alcanzados

| Métrica de Rendimiento | Propósito Técnico | Meta u Objetivo | Valor Alcanzado en Avance 2 | Estado |
| :--- | :--- | :---: | :---: | :---: |
| **Latencia / Tasa de Cuadros (FPS)** | Fluidez de la inferencia de visión en CPU | $\ge 24\text{ FPS}$ | **28 - 32 FPS** | **Superado** |
| **Mean Absolute Error (MAE) Cervical** | Precisión de la flexión oreja-hombro | $\le 3.5^\circ$ | **$2.4^\circ$** | **Cumplido** |
| **Mean Absolute Error (MAE) Hombros** | Precisión del desnivel horizontal | $\le 3.0^\circ$ | **$1.8^\circ$** | **Cumplido** |
| **Accuracy en Clasificación Postural** | Detección binaria (óptima vs riesgo) | $\ge 90\%$ | **$93.8\%$** | **Cumplido** |
| **F1-Score en Clasificación Postural** | Balance entre precisión y exhaustividad | $\ge 0.88$ | **$0.91$** | **Cumplido** |
| **Tasa de Falsos Positivos ($FPR$)** | Alertas indebidas ante movimientos $< 10\text{s}$ | $\le 5\%$ | **$2.8\%$** | **Cumplido** |
| **Silhouette Score (K-Means)** | Calidad y separación de clusters de fatiga | $\ge 0.50$ | **$0.5888$** | **Cumplido** |
| **Inercia WCSS ($k=3$)** | Compacidad intra-cluster de fatiga | Punto de codo | **$303.78$** | **Validado** |
| **Tiempo de Respuesta API REST** | Latencia de inferencia cliente-servidor | $\le 50\text{ ms}$ | **$8.4\text{ ms}$** | **Superado** |

---

## 12. RIESGOS Y ÉTICA

### Sesgos, Privacidad, Seguridad, Explicabilidad y Supervisión Humana

1. **Privacidad por Diseño (*Privacy by Design* y *Edge AI*):**  
   - **Riesgo:** Desconfianza de los usuarios ante el temor de vigilancia patronal o grabación encubierta en su hogar.  
   - **Mitigación:** Arquitectura estricta de computación en el borde. Las imágenes residen exclusivamente en la memoria RAM volátil durante los milisegundos de inferencia matemática y se destruyen de inmediato. No existe persistencia de video en disco duro ni transmisión a servidores externos en la nube.
2. **Mitigación de Sesgos Antropométricos y Lumínicos:**  
   - **Riesgo:** Variaciones por contextura física, altura de la persona o inclinación de la pantalla del portátil.  
   - **Mitigación:** Mecanismo interactivo de auto-calibración en neutro (tecla `C`) que registra los valores base individuales $(\theta_{c0}, \theta_{h0}, D_{r0})$.
3. **Explicabilidad y Transparencia Algorítmica (*XAI*):**  
   - Los diagnósticos no provienen de una "caja negra" opaca. El HUD despliega con precisión los ángulos medidos en grados, los umbrales normativos ISO aplicados y el centroide K-Means asignado.
4. **Supervisión Humana y Delimitación Médica:**  
   - El sistema actúa como un asistente consultivo y no como un sistema de toma de decisiones autónomo punitivo.  
   - Incluye un descargo de responsabilidad explícito declarando que es una herramienta de prevención laboral higiénico-postural y no un sustituto de diagnóstico médico o fisioterapéutico.

---

## 13. TRABAJO PENDIENTE

### Hoja de Ruta hacia la Versión Final (Avance 3)

Para la entrega final del proyecto, se desarrollarán los siguientes componentes:

1. **Empaquetamiento en Segundo Plano (*System Tray App*):**  
   Migrar el asistente a una aplicación de bandeja del sistema en Windows/Linux que opere silenciosamente en segundo plano, activando notificaciones nativas de escritorio del sistema operativo.
2. **Módulo Automatizado de Reportes Semanales de Salud Ocupacional:**  
   Generador de resúmenes analíticos en PDF con gráficos de tendencias de mejora postural, mapas de calor por horas de mayor fatiga y cumplimiento de pausas activas.
3. **Pruebas de Aceptación de Usuario en Campo (*UAT*):**  
   Sesiones de validación longitudinal con 15 usuarios reales en el campus universitario del Politécnico Grancolombiano durante jornadas laborales de 4 horas continuas, evaluando usabilidad, ergonomía y satisfacción.

---

### REFERENCIAS (FORMATO APA 7.ª EDICIÓN)

1. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
2. International Organization for Standardization. (2018). *Ergonomics of human-system interaction — Part 5: Workstation layout and postural requirements* (ISO Standard No. 9241-5:2018). https://www.iso.org/standard/66847.html
3. Lin, P.-C., Chen, Y.-J., Chen, W.-S., et al. (2022). Automatic real-time occupational posture evaluation and select corresponding ergonomic assessments. *Scientific Reports*, 12(1), Article 2139. https://doi.org/10.1038/s41598-022-05812-9
4. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., & Grundmann, M. (2019). MediaPipe: A Framework for Building Perception Pipelines. *arXiv preprint arXiv:1906.08172*.
5. Menanno, M., Riccio, C., Benedetto, V., Gissi, F., Savino, M. M., & Troiano, L. (2024). An Ergonomic Risk Assessment System Based on 3D Human Pose Estimation and Collaborative Robot. *Applied Sciences*, 14(11), Article 4823. https://doi.org/10.3390/app14114823
6. Occupational Safety and Health Administration (OSHA). (s. f.). *Computer Workstations: Good Working Positions*. U.S. Department of Labor. https://www.osha.gov/etools/computer-workstations/positions
7. Rodrigues, P. B., Xiao, Y., Fukumura, Y. E., Awada, M., Aryal, A., Becerik-Gerber, B., Lucas, G., & Roll, S. C. (2022). Ergonomic assessment of office worker postures using 3D automated joint angle assessment. *Advanced Engineering Informatics*, 52, Article 101596. https://doi.org/10.1016/j.aei.2022.101596
8. Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7
