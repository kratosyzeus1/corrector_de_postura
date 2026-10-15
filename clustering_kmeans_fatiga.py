"""
=============================================================================
PROYECTO: Asistente Ergonómico Inteligente y Monitor de Salud Ocupacional
INSTITUCIÓN: Politécnico Grancolombiano
AUTORES: Valery Sofía Meléndez Gallego, Juan Guillermo Ramírez Vásquez, Samuel Felipe Marín Lara
ENTREGA: Avance 2 - Módulo de Aprendizaje No Supervisado (Clustering K-Means)
=============================================================================
DELIMITACIÓN TÉCNICA Y ÉTICA (ATENCIÓN A RETROALIMENTACIÓN DOCENTE):
Este módulo clasifica bloques horarios de trabajo en tres niveles operacionales
de exposición biomecánica ("Baja fatiga", "Fatiga moderada", "Alta fatiga").
NO constituye un diagnóstico médico clínico ni formula recomendaciones terapéuticas.
Su objetivo exclusivo es la modulación no supervisada de la frecuencia y tipo
de sugerencias de pausas activas preventivas en el entorno laboral/académico.
=============================================================================
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Modo headless sin GUI para servidor/terminal
import matplotlib.pyplot as plt

try:
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, silhouette_samples
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[INFO] Bibliotecas avanzadas en proceso de carga...")


def generar_dataset_bloques_ergonomia(n_samples=360, random_state=42):
    """
    Genera un conjunto de datos estructurado y representativo de bloques horarios (30 min)
    de uso de computador en jornadas laborales y académicas.
    
    Variables biomecánicas y de comportamiento:
    - hora_jornada: Hora del día (8 a 18)
    - tiempo_riesgo_pct: Porcentaje de tiempo en postura deficiente dentro del bloque (0-100%)
    - angulo_cervical_promedio: Media de inclinación oreja-hombro (grados)
    - asimetria_hombros_promedio: Media de desnivel escapular (grados)
    - alertas_disparadas: Número de alertas activadas por el agente temporal (0 a 7)
    - tiempo_sedentario_continuo_min: Minutos continuos sin descanso (5 a 55 min)
    - score_ergonomico_medio: Score acumulado del bloque (0 a 100)
    """
    np.random.seed(random_state)
    
    # 1. Subpoblación de "Baja fatiga" (~40% de los bloques: inicio de jornada o post-almuerzo)
    n1 = int(n_samples * 0.40)
    riesgo_pct_1 = np.random.uniform(2.0, 18.0, n1)
    cervical_1 = np.random.normal(15.0, 2.5, n1)
    hombros_1 = np.random.normal(2.0, 1.0, n1)
    alertas_1 = np.random.choice([0, 1], size=n1, p=[0.85, 0.15])
    sedentario_1 = np.random.uniform(8.0, 25.0, n1)
    score_1 = np.clip(100 - (riesgo_pct_1 * 0.7) - np.random.normal(0, 2, n1), 85, 100)
    horas_1 = np.random.choice([8, 9, 10, 14, 15], size=n1)

    # 2. Subpoblación de "Fatiga moderada" (~35% de los bloques: mitad de jornada)
    n2 = int(n_samples * 0.35)
    riesgo_pct_2 = np.random.uniform(22.0, 48.0, n2)
    cervical_2 = np.random.normal(24.0, 3.2, n2)
    hombros_2 = np.random.normal(5.5, 1.5, n2)
    alertas_2 = np.random.choice([1, 2, 3], size=n2, p=[0.5, 0.35, 0.15])
    sedentario_2 = np.random.uniform(25.0, 42.0, n2)
    score_2 = np.clip(100 - (riesgo_pct_2 * 0.85) - np.random.normal(0, 3, n2), 58, 82)
    horas_2 = np.random.choice([10, 11, 12, 15, 16], size=n2)

    # 3. Subpoblación de "Alta fatiga" (~25% de los bloques: final de jornada o sobrecarga)
    n3 = n_samples - n1 - n2
    riesgo_pct_3 = np.random.uniform(52.0, 92.0, n3)
    cervical_3 = np.random.normal(33.0, 4.0, n3)
    hombros_3 = np.random.normal(8.5, 2.0, n3)
    alertas_3 = np.random.choice([3, 4, 5, 6], size=n3, p=[0.3, 0.4, 0.2, 0.1])
    sedentario_3 = np.random.uniform(40.0, 58.0, n3)
    score_3 = np.clip(100 - (riesgo_pct_3 * 0.95) - np.random.normal(0, 4, n3), 20, 55)
    horas_3 = np.random.choice([11, 12, 16, 17, 18], size=n3)

    # Concatenación y ensamble de DataFrame
    data = {
        'bloque_id': [f"BLK_{i+1:04d}" for i in range(n_samples)],
        'hora_jornada': np.concatenate([horas_1, horas_2, horas_3]),
        'tiempo_riesgo_pct': np.round(np.concatenate([riesgo_pct_1, riesgo_pct_2, riesgo_pct_3]), 2),
        'angulo_cervical_promedio': np.round(np.concatenate([cervical_1, cervical_2, cervical_3]), 2),
        'asimetria_hombros_promedio': np.round(np.concatenate([hombros_1, hombros_2, hombros_3]), 2),
        'alertas_disparadas': np.concatenate([alertas_1, alertas_2, alertas_3]),
        'tiempo_sedentario_continuo_min': np.round(np.concatenate([sedentario_1, sedentario_2, sedentario_3]), 1),
        'score_ergonomico_medio': np.round(np.concatenate([score_1, score_2, score_3]), 1)
    }

    df = pd.DataFrame(data)
    # Mezcla pseudoaleatoria
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


class ModeloKMeansFatiga:
    def __init__(self, k=3):
        self.k = k
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=k, init='k-means++', n_init=15, max_iter=300, random_state=42)
        self.features = [
            'tiempo_riesgo_pct',
            'angulo_cervical_promedio',
            'asimetria_hombros_promedio',
            'alertas_disparadas',
            'tiempo_sedentario_continuo_min',
            'score_ergonomico_medio'
        ]
        self.cluster_names_map = {}
        self.cluster_profiles = {}
        self.wcss_list = []
        self.silhouette_avg = 0.0

    def ajustar_y_entrenar(self, df):
        """Entrena el agrupador no supervisado y clasifica los centroides por severidad de fatiga"""
        X = df[self.features].values
        X_scaled = self.scaler.fit_transform(X)

        # 1. Cálculo de curva del codo (Elbow Method) para sustentar la elección de K=3
        self.wcss_list = []
        for k_val in range(1, 8):
            km_temp = KMeans(n_clusters=k_val, init='k-means++', n_init=10, random_state=42)
            km_temp.fit(X_scaled)
            self.wcss_list.append(km_temp.inertia_)

        # 2. Ajuste definitivo con K=3
        self.kmeans.fit(X_scaled)
        labels = self.kmeans.labels_
        self.silhouette_avg = float(silhouette_score(X_scaled, labels))

        # 3. Mapeo semántico de clusters (ordenar por score promedio o riesgo de menor a mayor fatiga)
        df_eval = df.copy()
        df_eval['cluster_raw'] = labels
        agrupados = df_eval.groupby('cluster_raw')['score_ergonomico_medio'].mean().sort_values(ascending=False)
        
        # El cluster con mayor score medio = "Baja fatiga"
        # El intermedio = "Fatiga moderada"
        # El menor score medio = "Alta fatiga"
        orden_clusters = agrupados.index.tolist()
        self.cluster_names_map = {
            orden_clusters[0]: "Baja fatiga",
            orden_clusters[1]: "Fatiga moderada",
            orden_clusters[2]: "Alta fatiga"
        }

        # Generar perfiles resumidos de cada cluster
        for raw_id, label_semantico in self.cluster_names_map.items():
            subset = df_eval[df_eval['cluster_raw'] == raw_id]
            self.cluster_profiles[label_semantico] = {
                'tamano_muestra': int(len(subset)),
                'porcentaje': round(len(subset) / len(df) * 100, 1),
                'score_medio': round(float(subset['score_ergonomico_medio'].mean()), 2),
                'cervical_media_deg': round(float(subset['angulo_cervical_promedio'].mean()), 2),
                'tiempo_riesgo_pct_medio': round(float(subset['tiempo_riesgo_pct'].mean()), 2),
                'alertas_medias': round(float(subset['alertas_disparadas'].mean()), 2),
                'sedentario_min_medio': round(float(subset['tiempo_sedentario_continuo_min'].mean()), 2),
                'accion_recomendada': self._obtener_accion_operativa(label_semantico)
            }

        return labels

    def _obtener_accion_operativa(self, nivel_fatiga):
        """Acción ergonómica modulada sin extralimitación médica"""
        if nivel_fatiga == "Baja fatiga":
            return "Mantener ritmo de trabajo; programar recordatorio de hidratación a los 45 minutos."
        elif nivel_fatiga == "Fatiga moderada":
            return "Micro-pausa sugerida de 30 segundos: estiramiento suave de trapecios y parpadeo consciente."
        else:
            return "Pausa activa obligatoria de 3 a 5 minutos: levantarse de la silla, rotación de hombros y descanso visual 20-20-20."

    def predecir_bloque(self, datos_dict):
        """Clasifica un nuevo vector de telemetría de 30 min en su categoría de fatiga"""
        vec = np.array([[
            datos_dict['tiempo_riesgo_pct'],
            datos_dict['angulo_cervical_promedio'],
            datos_dict['asimetria_hombros_promedio'],
            datos_dict['alertas_disparadas'],
            datos_dict['tiempo_sedentario_continuo_min'],
            datos_dict['score_ergonomico_medio']
        ]])
        vec_scaled = self.scaler.transform(vec)
        raw_cluster = int(self.kmeans.predict(vec_scaled)[0])
        nombre_cluster = self.cluster_names_map.get(raw_cluster, "Indeterminado")
        perfil = self.cluster_profiles.get(nombre_cluster, {})
        return {
            'cluster_id': raw_cluster,
            'categoria_fatiga': nombre_cluster,
            'accion_ergonomica': self._obtener_accion_operativa(nombre_cluster),
            'perfil_referencia': perfil
        }

    def generar_graficos_evidencia(self, df, labels, out_dir="."):
        """Genera y guarda en disco las figuras de evidencia analítica para la entrega"""
        plt.style.use('dark_background')
        df_plot = df.copy()
        df_plot['categoria'] = [self.cluster_names_map[l] for l in labels]

        # -------------------------------------------------------------
        # 1. Gráfica de la Curva del Codo (Elbow Method)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        k_range = list(range(1, 8))
        ax.plot(k_range, self.wcss_list, 'o-', color='#00d4ff', linewidth=2.5, markersize=8, label='WCSS (Inercia)')
        ax.axvline(x=3, color='#ff3d00', linestyle='--', linewidth=2, label='K Seleccionado Óptimo (k=3)')
        ax.set_title('Método del Codo para Selección Óptima de Clusters (K-Means)', fontsize=13, fontweight='bold', pad=12, color='#ffffff')
        ax.set_xlabel('Número de Clusters (k)', fontsize=11, color='#cbd5e1')
        ax.set_ylabel('Inercia WCSS (Suma de Distancias Cuadráticas)', fontsize=11, color='#cbd5e1')
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.legend(frameon=True, facecolor='#1e2838', edgecolor='#334155')
        plt.tight_layout()
        path_codo = os.path.join(out_dir, 'evidencia_metodo_codo_kmeans.png')
        plt.savefig(path_codo)
        plt.close()
        print(f"[GRÁFICO] Guardado: {path_codo}")

        # -------------------------------------------------------------
        # 2. Gráfica de Dispersión 2D de Clusters de Fatiga
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        colores = {
            'Baja fatiga': '#00e676',       # Verde
            'Fatiga moderada': '#ffb300',   # Ámbar
            'Alta fatiga': '#ff3d00'        # Rojo
        }

        for cat, col in colores.items():
            subset = df_plot[df_plot['categoria'] == cat]
            ax.scatter(subset['tiempo_riesgo_pct'], subset['angulo_cervical_promedio'],
                       c=col, label=f"{cat} (n={len(subset)})", alpha=0.75, edgecolors='none', s=45)

        ax.set_title('Clustering K-Means (k=3): Clasificación de Bloques de Fatiga Ergonómica',
                     fontsize=13, fontweight='bold', pad=12, color='#ffffff')
        ax.set_xlabel('% Tiempo en Postura de Riesgo en el Bloque (30 min)', fontsize=11, color='#cbd5e1')
        ax.set_ylabel('Ángulo de Inclinación Cervical Promedio (°)', fontsize=11, color='#cbd5e1')
        ax.axhline(y=28.0, color='#ff6b81', linestyle=':', label='Umbral ISO 9241-5 (28°)')
        ax.grid(True, linestyle=':', alpha=0.35)
        ax.legend(frameon=True, facecolor='#1e2838', edgecolor='#334155', loc='upper left')
        plt.tight_layout()
        path_scatter = os.path.join(out_dir, 'evidencia_clusters_fatiga_2d.png')
        plt.savefig(path_scatter)
        plt.close()
        print(f"[GRÁFICO] Guardado: {path_scatter}")

        # -------------------------------------------------------------
        # 3. Gráfica de Distribución de Silhouette
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        X_scaled = self.scaler.transform(df[self.features].values)
        sample_silhouette_values = silhouette_samples(X_scaled, labels)
        
        y_lower = 10
        for i, (raw_id, cat_name) in enumerate(self.cluster_names_map.items()):
            ith_cluster_values = sample_silhouette_values[labels == raw_id]
            ith_cluster_values.sort()
            size_cluster_i = ith_cluster_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = colores.get(cat_name, '#00d4ff')
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, cat_name, color='#ffffff', fontsize=9, fontweight='bold')
            y_lower = y_upper + 10

        ax.axvline(x=self.silhouette_avg, color='#ffffff', linestyle='--', label=f'Silhouette Promedio: {self.silhouette_avg:.3f}')
        ax.set_title(f'Coeficiente de Silhouette por Cluster (Media Global = {self.silhouette_avg:.3f})',
                     fontsize=12, fontweight='bold', pad=10, color='#ffffff')
        ax.set_xlabel('Valor del Coeficiente de Silhouette', fontsize=10, color='#cbd5e1')
        ax.set_ylabel('Muestras por Grupo', fontsize=10, color='#cbd5e1')
        ax.set_yticks([])
        ax.legend(frameon=True, facecolor='#1e2838', edgecolor='#334155')
        ax.grid(True, linestyle=':', alpha=0.3)
        plt.tight_layout()
        path_sil = os.path.join(out_dir, 'evidencia_silhouette_kmeans.png')
        plt.savefig(path_sil)
        plt.close()
        print(f"[GRÁFICO] Guardado: {path_sil}")

    def guardar_modelo(self, filepath_pkl="modelo_kmeans_fatiga.pkl", filepath_json="modelo_kmeans_fatiga.json"):
        """Exporta el modelo entrenado y metadatos en formatos portables"""
        # Guardar binario con scikit-learn
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'features': self.features,
            'cluster_names_map': self.cluster_names_map,
            'cluster_profiles': self.cluster_profiles,
            'silhouette_avg': self.silhouette_avg
        }, filepath_pkl)
        print(f"[MODELO] Exportado binario a: {filepath_pkl}")

        # Guardar resumen legible en JSON para integración con Frontend/API
        resumen = {
            'algoritmo': 'K-Means (Clustering No Supervisado)',
            'k_clusters': self.k,
            'metricas_evaluacion': {
                'silhouette_score_promedio': round(self.silhouette_avg, 4),
                'inercia_final_wcss': round(float(self.kmeans.inertia_), 2)
            },
            'variables_modelo': self.features,
            'perfiles_clusters': self.cluster_profiles
        }
        with open(filepath_json, 'w', encoding='utf-8') as f:
            json.dump(resumen, f, indent=2, ensure_ascii=False)
        print(f"[MODELO] Exportado JSON de metadatos a: {filepath_json}")


def ejecutar_pipeline_completo():
    print("=" * 70)
    print(" PIPELINE DE ENTRENAMIENTO Y VALIDACIÓN - CLUSTERING K-MEANS")
    print(" ASISTENTE ERGONÓMICO INTELIGENTE | AVANCE 2 (POLI GRANCOLOMBIANO)")
    print("=" * 70)

    # 1. Generación y almacenamiento del Dataset
    dataset_path = "dataset_bloques_ergonomia.csv"
    print(f"\n[PASO 1] Generando dataset procesado de sesiones de trabajo...")
    df = generar_dataset_bloques_ergonomia(n_samples=360, random_state=42)
    df.to_csv(dataset_path, index=False, encoding='utf-8')
    print(f" -> Dataset guardado exitosamente en: {dataset_path} ({len(df)} registros)")

    # 2. Entrenamiento del Modelo
    print(f"\n[PASO 2] Entrenando K-Means (k=3) y escalando con StandardScaler...")
    modelo = ModeloKMeansFatiga(k=3)
    labels = modelo.ajustar_y_entrenar(df)
    print(f" -> Coeficiente de Silhouette alcanzado: {modelo.silhouette_avg:.4f}")

    # 3. Reporte de Centroides y Perfiles
    print(f"\n[PASO 3] Perfiles de fatiga obtenidos mediante agrupamiento no supervisado:")
    for cat, prof in modelo.cluster_profiles.items():
        print(f"  • {cat.upper()}:")
        print(f"     - Muestras: {prof['tamano_muestra']} ({prof['porcentaje']}%)")
        print(f"     - Score Ergonómico Medio: {prof['score_medio']}%")
        print(f"     - Ángulo Cervical Medio: {prof['cervical_media_deg']}°")
        print(f"     - Tiempo en Riesgo Medio: {prof['tiempo_riesgo_pct_medio']}% del bloque")
        print(f"     - Sedentarismo Continuo Medio: {prof['sedentario_min_medio']} minutos")
        print(f"     - Acción Ergonómica: {prof['accion_recomendada']}")

    # 4. Generación de Gráficos de Evidencia
    print(f"\n[PASO 4] Generando figuras de evidencia experimental...")
    modelo.generar_graficos_evidencia(df, labels)

    # 5. Persistencia del Modelo
    print(f"\n[PASO 5] Guardando artefactos del modelo...")
    modelo.guardar_modelo()

    # 6. Prueba de Predicción con Nuevos Casos (Validación de Inferencia)
    print(f"\n[PASO 6] Test de inferencia sobre bloques horarios de prueba:")
    caso_prueba_1 = {
        'tiempo_riesgo_pct': 8.5,
        'angulo_cervical_promedio': 16.2,
        'asimetria_hombros_promedio': 2.1,
        'alertas_disparadas': 0,
        'tiempo_sedentario_continuo_min': 15.0,
        'score_ergonomico_medio': 94.0
    }
    pred_1 = modelo.predecir_bloque(caso_prueba_1)
    print(f"  * Caso 1 (Usuario alineado y activo): -> {pred_1['categoria_fatiga'].upper()}")
    print(f"    Acción: {pred_1['accion_ergonomica']}")

    caso_prueba_2 = {
        'tiempo_riesgo_pct': 72.0,
        'angulo_cervical_promedio': 34.5,
        'asimetria_hombros_promedio': 9.2,
        'alertas_disparadas': 4,
        'tiempo_sedentario_continuo_min': 50.0,
        'score_ergonomico_medio': 38.0
    }
    pred_2 = modelo.predecir_bloque(caso_prueba_2)
    print(f"  * Caso 2 (Usuario encorvado con sedentarismo alto): -> {pred_2['categoria_fatiga'].upper()}")
    print(f"    Acción: {pred_2['accion_ergonomica']}")

    print("\n" + "=" * 70)
    print(" PIPELINE DE K-MEANS EJECUTADO Y VALIDADO CON ÉXITO")
    print("=" * 70)

if __name__ == "__main__":
    ejecutar_pipeline_completo()
