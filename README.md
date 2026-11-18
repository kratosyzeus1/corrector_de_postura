# Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
### *Intelligent Ergonomic Assistant and Occupational Health Monitor Based on Computer Vision and K-Means Clustering*
### Holi

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
├── asistente_ergonomico.py         # Módulo principal: visión BlazePose (Frontal/Perfil), 2 vistas y K-Means
├── gestor_camara.py                # Detección multi-cámara y manejo anti-conflicto con videollamadas (Windows)
├── interfaz_overlay.py             # Selector inicial multi-cámara + Vista 2 transparente de solo notificaciones
├── monitor_recursos.py             # Monitor nativo en tiempo real de RAM (Working Set/Privada/Pico) y CPU
├── validar_sistema.py              # Script de validación integral (multi-cámara, conflictos, RAM y frontal/perfil)
├── .py                             # Wrapper de ejecución rápida del asistente
├── clustering_kmeans_fatiga.py     # Script de entrenamiento, evaluación del codo y métricas de K-Means
├── dataset_bloques_ergonomia.csv   # Dataset estructurado de 360 bloques horarios de sesiones
├── modelo_kmeans_fatiga.pkl        # Modelo serializado K-Means entrenado (scikit-learn)
├── modelo_kmeans_fatiga.json       # Metadatos, centroides e interpretabilidad de clusters
├── api_servicio_ergonomia.py       # API REST con FastAPI para inferencia de fatiga y postura
├── probar_llamada_api.py           # Cliente de prueba automatizado de la API REST (HTTP 200 OK)
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

### 3. Ejecutar el Asistente en Tiempo Real (Multi-Cámara / 2 Vistas / Anti-Conflicto)
```bash
python asistente_ergonomico.py
```
Al iniciar, se abre un selector compacto para elegir la cámara (cuando hay más de una), la vista inicial y la protección de videollamadas. También acepta opciones por línea de comandos:
```bash
python asistente_ergonomico.py --listar-camaras
python asistente_ergonomico.py --camara 1 --vista overlay
python asistente_ergonomico.py --vista compacta --demo
```

**Dos Vistas No Disruptivas:**
- **Vista 2 (`overlay` — Recomendada):** Ejecución en segundo plano sobre la pantalla normal del usuario (sin verse en cámara). Solo muestra una mini-píldora flotante semitransparente y notificaciones discretas tipo *Toast* cuando se requiere corregir la postura o cuando se cede la cámara a una videollamada.
- **Vista 1 (`compacta`):** Muestra la imagen en vivo de la cámara con esqueleto ergonómico ligero, paneles compactos, mensajes/notificaciones pequeños (con doble buffer y suavizado anti-parpadeo) y lectura en vivo de RAM. Con `[F]` también puede alternarse a modo solo avatar.

**Controles del Teclado (en Vista 1):**
- `[T]` o `[TAB]`: Alternar a Vista 2 (Overlay / Segundo plano).
- `[V]`: Cambiar a la siguiente cámara disponible en caliente.
- `[P]`: Pausar y liberar manualmente la cámara (o reanudar).
- `[F]`: Alternar fondo entre Cámara en vivo (defecto) y Avatar sin espejo.
- `[C]`: Calibrar postura neutra de referencia actual.
- `[M]`: Alternar entre Cámara Web en vivo y Modo Demostración Sintético.
- `[K]`: Ejecutar inferencia K-Means sobre la telemetría acumulada.
- `[S]`: Guardar captura de pantalla de evidencia.
- `[Q]`: Finalizar ejecución y ver resumen de sesión (incluye métricas de RAM y conflictos).

### 4. Validar Multi-Cámara, Conflictos de Videollamada y Consumo de RAM
```bash
python validar_sistema.py
```
Ejecuta las pruebas automatizadas de los 4 ajustes solicitados, exporta `reporte_validacion_recursos.json` y genera la gráfica `evidencia_consumo_ram.png`.

### 5. Probar el Servicio API REST
```bash
python probar_llamada_api.py
```

### 6. Reentrenar el Modelo K-Means
```bash
python clustering_kmeans_fatiga.py
```

---

## 📊 Métricas de Desempeño Alcanzadas (Avance 2)

- **Tasa de inferencia:** 28 - 32 FPS en CPU comercial sin GPU dedicada.
- **Error Absoluto Medio Angular (MAE):** $2.4^\circ$ cervical / $1.8^\circ$ hombros.
- **Coeficiente de Silhouette (K-Means $k=3$):** **0.5888** (separación nítida de clusters).
- **Inercia WCSS:** 303.78 (punto de inflexión del codo óptimo).
- **Tasa de falsos positivos en movimientos breves:** $< 3\%$.
- **Tiempo de respuesta de inferencia API:** $< 10\text{ ms}$.
