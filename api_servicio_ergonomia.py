"""
=============================================================================
PROYECTO: Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
INSTITUCIÓN: Politécnico Grancolombiano
ENTREGA: Avance 2 - Servicio de Inferencia API REST para Evidencia Funcional
=============================================================================
Este servicio web proporciona endpoints REST para:
1. Inferencia del modelo no supervisado K-Means (Clustering de fatiga en bloques horarios).
2. Evaluación cinemática de coordenadas articulares en tiempo real.
3. Consulta de telemetría y metadatos de modelos de IA locales.
=============================================================================
"""

import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List

# Inicializar aplicación FastAPI
app = FastAPI(
    title="API de Inferencia - Asistente Ergonómico Inteligente",
    description="Servicio backend para evaluación postural biomecánica y clustering K-Means de fatiga.",
    version="2.0.0"
)

# Rutas de artefactos
MODEL_PKL = "modelo_kmeans_fatiga.pkl"
MODEL_JSON = "modelo_kmeans_fatiga.json"

# Carga de modelo K-Means
modelo_cargado = None
metadatos_json = {}

if os.path.exists(MODEL_PKL):
    try:
        modelo_cargado = joblib.load(MODEL_PKL)
    except Exception as e:
        print(f"[WARN] Error cargando {MODEL_PKL}: {e}")

if os.path.exists(MODEL_JSON):
    try:
        with open(MODEL_JSON, 'r', encoding='utf-8') as f:
            metadatos_json = json.load(f)
    except Exception as e:
        print(f"[WARN] Error cargando {MODEL_JSON}: {e}")


# Esquemas de datos Pydantic
class BloqueTelemetriaInput(BaseModel):
    tiempo_riesgo_pct: float = Field(..., ge=0.0, le=100.0, description="% de tiempo en mala postura durante el bloque")
    angulo_cervical_promedio: float = Field(..., ge=0.0, le=60.0, description="Ángulo medio de flexión cervical en grados")
    asimetria_hombros_promedio: float = Field(..., ge=0.0, le=30.0, description="Ángulo medio de asimetría escapular en grados")
    alertas_disparadas: int = Field(..., ge=0, description="Número de alertas activadas por el agente en el bloque")
    tiempo_sedentario_continuo_min: float = Field(..., ge=0.0, description="Minutos continuos sentado sin descanso")
    score_ergonomico_medio: float = Field(..., ge=0.0, le=100.0, description="Score postural acumulado promedio")


class PosturaInstantaneaInput(BaseModel):
    angulo_cervical: float
    desnivel_hombros: float
    distancia_relativa: float = 0.25


@app.get("/")
def raiz_estado():
    return {
        "sistema": "Asistente Ergonómico Inteligente - Poli Grancolombiano",
        "fase": "Avance 2 - Evidencia Funcional de Arquitectura y Modelado",
        "estado": "ACTIVO_Y_OPERACIONAL",
        "modelos_disponibles": {
            "vision_artificial": "MediaPipe BlazePose (CNN Ligera en Edge)",
            "clustering_habitos": "K-Means (k=3, No Supervisado)",
            "metricas": metadatos_json.get("metricas_evaluacion", {})
        },
        "delimitacion_tecnica": (
            "Sistema enfocado en prevención ergonómica laboral y modulación de pausas activas. "
            "No emite diagnósticos médicos ni prescripciones clínicas."
        )
    }


@app.get("/api/perfiles-clusters")
def obtener_perfiles_clusters():
    if not modelo_cargado:
        raise HTTPException(status_code=503, detail="Modelo K-Means aún no disponible en el servidor.")
    return {
        "algoritmo": "K-Means Clustering",
        "silhouette_score": round(modelo_cargado.get('silhouette_avg', 0.0), 4),
        "perfiles": modelo_cargado.get('cluster_profiles', {})
    }


@app.post("/api/predecir-fatiga")
def predecir_categoria_fatiga(bloque: BloqueTelemetriaInput):
    """
    Clasifica un bloque de trabajo (ej. 30 minutos) en una de las tres categorías de fatiga
    mediante el modelo no supervisado K-Means previamente entrenado.
    """
    if not modelo_cargado:
        raise HTTPException(status_code=503, detail="Modelo K-Means no cargado.")

    scaler = modelo_cargado['scaler']
    kmeans = modelo_cargado['kmeans']
    names_map = modelo_cargado['cluster_names_map']
    profiles = modelo_cargado['cluster_profiles']

    # Ensamblar vector
    vec = np.array([[
        bloque.tiempo_riesgo_pct,
        bloque.angulo_cervical_promedio,
        bloque.asimetria_hombros_promedio,
        bloque.alertas_disparadas,
        bloque.tiempo_sedentario_continuo_min,
        bloque.score_ergonomico_medio
    ]])

    vec_scaled = scaler.transform(vec)
    raw_cluster = int(kmeans.predict(vec_scaled)[0])
    categoria_semantica = names_map.get(raw_cluster, "Indeterminado")
    perfil = profiles.get(categoria_semantica, {})

    # Calcular distancia al centroide más cercano
    centroide = kmeans.cluster_centers_[raw_cluster]
    distancia_euclidiana = float(np.linalg.norm(vec_scaled - centroide))

    return {
        "status": "success",
        "bloque_clasificado": {
            "cluster_raw_id": raw_cluster,
            "categoria_fatiga": categoria_semantica,
            "distancia_al_centroide": round(distancia_euclidiana, 4),
            "recomendacion_operativa": perfil.get("accion_recomendada", ""),
            "metricas_referencia_cluster": {
                "score_esperado": perfil.get("score_medio"),
                "cervical_esperada_deg": perfil.get("cervical_media_deg"),
                "sedentario_medio_min": perfil.get("sedentario_min_medio")
            }
        }
    }


@app.post("/api/evaluar-postura")
def evaluar_postura_instantanea(p: PosturaInstantaneaInput):
    """
    Evalúa cinemáticamente una medición angular y aplica las reglas biomecánicas ISO 9241-5.
    """
    motivos = []
    penalizacion = 0

    if p.angulo_cervical > 28.0:
        motivos.append("Inclinación Cervical Crítica (Cifosis)")
        penalizacion += 40
    if p.desnivel_hombros > 7.0:
        motivos.append("Asimetría Escapular Marcada")
        penalizacion += 30
    if p.distancia_relativa > 0.36:
        motivos.append("Proximidad Excesiva al Monitor")
        penalizacion += 20

    score = max(10, 100 - penalizacion)
    estado = "POSTURA ÓPTIMA" if penalizacion == 0 else "RIESGO POSTURAL"

    return {
        "estado": estado,
        "score_ergonomico": score,
        "desviaciones_detectadas": motivos,
        "requiere_atencion": penalizacion >= 40
    }
