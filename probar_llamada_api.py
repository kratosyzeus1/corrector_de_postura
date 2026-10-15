"""
=============================================================================
PROYECTO: Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
INSTITUCIÓN: Politécnico Grancolombiano
ENTREGA: Avance 2 - Script de Verificación de Llamada Funcional a API REST
=============================================================================
Demuestra la invocación funcional cliente-servidor a los endpoints de IA,
cumpliendo el requerimiento del docente de evidenciar llamadas a API operativas.
=============================================================================
"""

import json
from fastapi.testclient import TestClient
from api_servicio_ergonomia import app

def ejecutar_pruebas_api():
    print("=" * 70)
    print(" VERIFICACIÓN DE LLAMADA FUNCIONAL A API REST ERGONÓMICA")
    print(" ASISTENTE ERGONÓMICO INTELIGENTE | AVANCE 2 (POLI)")
    print("=" * 70)

    client = TestClient(app)

    # 1. Llamada GET a la raíz del servicio
    print("\n[TEST 1] Invocando endpoint: GET / (Estado del servicio y modelos)")
    res_root = client.get("/")
    print(f" -> Código HTTP de respuesta: {res_root.status_code}")
    print(f" -> Contenido JSON recibido:\n{json.dumps(res_root.json(), indent=2, ensure_ascii=False)}")
    assert res_root.status_code == 200, "Error en endpoint raíz"

    # 2. Llamada GET a perfiles de clusters
    print("\n[TEST 2] Invocando endpoint: GET /api/perfiles-clusters (Metadatos K-Means)")
    res_clusters = client.get("/api/perfiles-clusters")
    print(f" -> Código HTTP: {res_clusters.status_code}")
    data_clusters = res_clusters.json()
    print(f" -> Silhouette Score del modelo: {data_clusters.get('silhouette_score')}")
    print(f" -> Categorías disponibles: {list(data_clusters.get('perfiles', {}).keys())}")
    assert res_clusters.status_code == 200, "Error en endpoint perfiles"

    # 3. Llamada POST de Inferencia K-Means (Simulación de Bloque de Alta Fatiga)
    print("\n[TEST 3] Invocando endpoint: POST /api/predecir-fatiga (Caso: Jornada Intensa)")
    payload_fatiga = {
        "tiempo_riesgo_pct": 78.5,
        "angulo_cervical_promedio": 35.2,
        "asimetria_hombros_promedio": 8.7,
        "alertas_disparadas": 5,
        "tiempo_sedentario_continuo_min": 52.0,
        "score_ergonomico_medio": 32.0
    }
    res_fatiga = client.post("/api/predecir-fatiga", json=payload_fatiga)
    print(f" -> Código HTTP: {res_fatiga.status_code}")
    print(f" -> Payload enviado: {payload_fatiga}")
    print(f" -> Respuesta de Inferencia K-Means:\n{json.dumps(res_fatiga.json(), indent=2, ensure_ascii=False)}")
    assert res_fatiga.status_code == 200, "Error en predicción K-Means"
    assert res_fatiga.json()["bloque_clasificado"]["categoria_fatiga"] == "Alta fatiga"

    # 4. Llamada POST de Evaluación Cinemática Instantánea
    print("\n[TEST 4] Invocando endpoint: POST /api/evaluar-postura (Telemetría de Pose)")
    payload_postura = {
        "angulo_cervical": 15.4,
        "desnivel_hombros": 1.8,
        "distancia_relativa": 0.26
    }
    res_postura = client.post("/api/evaluar-postura", json=payload_postura)
    print(f" -> Código HTTP: {res_postura.status_code}")
    print(f" -> Respuesta:\n{json.dumps(res_postura.json(), indent=2, ensure_ascii=False)}")
    assert res_postura.status_code == 200

    print("\n" + "=" * 70)
    print(" TODAS LAS LLAMADAS A LA API RESPONDIERON CON ÉXITO (HTTP 200 OK)")
    print(" EVIDENCIA FUNCIONAL DE API Y ARQUITECTURA DISTRIBUIDA VERIFICADA")
    print("=" * 70)

if __name__ == "__main__":
    ejecutar_pruebas_api()
