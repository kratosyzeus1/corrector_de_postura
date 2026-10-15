# Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
### *Intelligent Ergonomic Assistant and Occupational Health Monitor Based on Computer Vision and K-Means Clustering*

**Institución:** Institución Universitaria Politécnico Grancolombiano  
**Facultad:** Facultad de Ingeniería, Diseño e Innovación  
**Materia:** Proyecto de Inteligencia Artificial (Avance 2)  
**Lugar y Fecha:** Medellín, Colombia - Octubre de 2026  

**Autores:**  
- **Valery Sofía Meléndez Gallego** (`vmelendez@poligran.edu.co`)  
- **Juan Guillermo Ramírez Vásquez** (`jguramirez@poligran.edu.co`)  
- **Samuel Felipe Marín Lara** (`sfmarin@poligran.edu.co`)  

---

## 📌 Descripción del Proyecto

El **Asistente Ergonómico Inteligente** es una solución de salud ocupacional preventiva basada en **Visión por Computador (Edge AI)** y **Aprendizaje No Supervisado (Clustering K-Means)**. 

El sistema utiliza la cámara web convencional del usuario para monitorizar en tiempo real la postura frente al computador, calculando ángulos cervicales y desalineación de hombros, mitigando falsos positivos mediante ventanas temporales y agrupando los bloques horarios de jornada en categorías de **"Baja fatiga"**, **"Fatiga moderada"** y **"Alta fatiga"** para modular la frecuencia y tipo de pausas activas sugeridas.

> 🛡️ **Delimitación Técnica y Ética:**  
> El sistema está delimitado exclusivamente a la prevención ergonómica laboral y a la modulación de pausas activas. **No constituye un dispositivo de diagnóstico médico ni prescribe tratamientos clínicos.**

---

## 🏗️ Arquitectura y Tecnologías

1. **Visión por Computador (Edge AI):** Google MediaPipe BlazePose para la estimación 3D de 33 articulaciones corporales en tiempo real (>25 FPS en CPU).
2. **Cinemática Biomecánica:** Cálculo trigonométrico de flexión cervical y simetría escapular según directrices ISO 9241-5 y OSHA.
3. **Agente Lógico con Memoria Temporal:** Tolerancia de 10 a 15 segundos para filtrar movimientos transitorios (beber agua, tomar apuntes) y evitar alertas invasivas.
4. **Machine Learning No Supervisado (K-Means):** Clasificación de bloques de jornada (30 a 60 min) en 3 perfiles de fatiga ($k=3$), evaluado con Silhouette Score (**0.5888**) y Método del Codo.
5. **Servicio API REST (FastAPI):** Endpoints para inferencia distribuida y evaluación instantánea de postura.
6. **Privacidad por Diseño:** Procesamiento 100% en memoria RAM volátil local; ninguna imagen ni video se almacena en disco ni se transmite a internet.

---

## 📂 Estructura del Repositorio

```text
├── asistente_ergonomico.py         # Módulo principal de visión por computador y HUD en tiempo real
├── .py                             # Wrapper de ejecución rápida del asistente
├── clustering_kmeans_fatiga.py     # Script de entrenamiento, evaluación del codo y métricas de K-Means
├── dataset_bloques_ergonomia.csv   # Dataset estructurado de 360 bloques horarios de sesiones
├── modelo_kmeans_fatiga.pkl        # Modelo serializado K-Means entrenado (scikit-learn)
├── modelo_kmeans_fatiga.json       # Metadatos, centroides e interpretabilidad de clusters
├── api_servicio_ergonomia.py       # API REST con FastAPI para inferencia de fatiga y postura
├── probar_llamada_api.py           # Cliente de prueba automatizado de la API REST (HTTP 200 OK)
├── demo_exposicion.html            # Dashboard web interactivo con simulador y diapositivas
├── DOCUMENTO_ENTREGA_AVANCE_2.md   # Documento académico formal de la entrega (Avance 2)
├── evidencia_postura_demo.png      # Captura de pantalla funcional del HUD en ejecución
├── evidencia_metodo_codo_kmeans.png# Gráfica del método del codo (WCSS)
├── evidencia_clusters_fatiga_2d.png# Mapa de dispersión 2D de clusters de fatiga
├── evidencia_silhouette_kmeans.png # Gráfica de distribución del Coeficiente de Silhouette
├── .gitignore                      # Exclusiones de Git (caché, entornos virtuales)
└── README.md                       # Documentación del repositorio
```

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd "Tarea uni"
```

### 2. Instalar dependencias
```bash
pip install opencv-python mediapipe numpy scikit-learn pandas matplotlib fastapi uvicorn httpx
```

### 3. Ejecutar el Asistente en Tiempo Real (Webcam / Demostración)
```bash
python asistente_ergonomico.py
```
**Controles del Teclado:**
- `[C]`: Calibrar postura neutra de referencia actual.
- `[M]`: Alternar entre Cámara Web en vivo y Modo Demostración Sintético.
- `[K]`: Ejecutar inferencia K-Means sobre la telemetría acumulada.
- `[S]`: Guardar captura de pantalla de evidencia.
- `[Q]`: Finalizar ejecución y ver estadísticas de la sesión.

### 4. Probar el Servicio API REST
```bash
python probar_llamada_api.py
```

### 5. Reentrenar el Modelo K-Means
```bash
python clustering_kmeans_fatiga.py
```

### 6. Abrir el Dashboard Interactivo de Exposición
Abre el archivo `demo_exposicion.html` directamente en cualquier navegador web (Google Chrome, Edge, Firefox).

---

## 📊 Métricas de Desempeño Alcanzadas (Avance 2)

- **Tasa de inferencia:** 28 - 32 FPS en CPU comercial sin GPU dedicada.
- **Error Absoluto Medio Angular (MAE):** $2.4^\circ$ cervical / $1.8^\circ$ hombros.
- **Coeficiente de Silhouette (K-Means $k=3$):** **0.5888** (separación nítida de clusters).
- **Inercia WCSS:** 303.78 (punto de inflexión del codo óptimo).
- **Tasa de falsos positivos en movimientos breves:** $< 3\%$.
- **Tiempo de respuesta de inferencia API:** $< 10\text{ ms}$.
