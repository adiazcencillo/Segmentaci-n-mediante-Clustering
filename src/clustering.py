import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def caso_de_uso_1(file_path, price_percentile=0.6, output_dir="output/clusters"):
    """
    Realiza clustering sobre alojamientos cuyo precio normalizado esté por encima del percentil dado.
    Args:
        file_path (str): Ruta al archivo CSV con los datos normalizados.
        price_percentile (float): Percentil superior del precio a filtrar (por defecto, 60%).
        output_dir (str): Carpeta donde se guardarán los gráficos.
    """
    # 1. Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # 2. Cargar datos normalizados
    print("Cargando datos normalizados...")
    data = pd.read_csv(file_path, sep=';')

    # 3. Calcular el percentil del precio
    price_threshold = data['Price avg'].quantile(price_percentile)
    print(f"Seleccionando alojamientos con precio normalizado mayor a {price_threshold:.2f} (percentil {price_percentile * 100:.0f}%)")

    # 4. Filtrar alojamientos por precio
    filtered_data = data[data['Price avg'] > price_threshold]
    print(f"Alojamientos seleccionados: {filtered_data.shape[0]}")

    # 5. Seleccionar variables relevantes para clustering
    features = filtered_data[['Price avg', 'Distance', 'Rating']]

    # 6. Aplicar K-means para k de 2 a 10 y calcular métricas
    silhouette_scores = []
    calinski_scores = []

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)
        labels = kmeans.labels_

        # Calcular métricas
        silhouette = silhouette_score(features, labels)
        calinski = calinski_harabasz_score(features, labels)

        silhouette_scores.append(silhouette)
        calinski_scores.append(calinski)

        print(f"K={k}: Silhouette Score={silhouette:.3f}, Calinski-Harabasz Index={calinski:.3f}")

    # 7. Visualizar métricas y guardar gráficos
    metrics_path = os.path.join(output_dir, "metrics_plot.png")
    visualize_metrics(range(2, 11), silhouette_scores, calinski_scores, metrics_path)

    # 8. Generar gráfica de Silhouette para el mejor K y guardar gráfico
    best_k = np.argmax(silhouette_scores) + 2
    print(f"Mejor K según Silhouette Score: {best_k}")
    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(features)
    silhouette_path = os.path.join(output_dir, f"silhouette_plot_k{best_k}.png")
    plot_silhouette(features, kmeans, silhouette_path)

def visualize_metrics(k_range, silhouette_scores, calinski_scores, output_path):
    """Visualiza y guarda las métricas de Silhouette y Calinski-Harabasz."""
    plt.figure(figsize=(12, 6))

    # Silhouette Score
    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o', label="Silhouette Score")
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score por K")
    plt.grid(True)

    # Calinski-Harabasz Index
    plt.subplot(1, 2, 2)
    plt.plot(k_range, calinski_scores, marker='o', label="Calinski-Harabasz Index", color='orange')
    plt.xlabel("Número de clusters (K)")
    plt.ylabel("Calinski-Harabasz Index")
    plt.title("Calinski-Harabasz Index por K")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)  # Guardar gráfico
    print(f"Métricas guardadas en {output_path}")
    plt.close()

def plot_silhouette(features, kmeans, output_path):
    """Genera y guarda un gráfico de Silhouette para los clusters."""
    labels = kmeans.labels_
    n_clusters = len(set(labels))
    silhouette_vals = silhouette_samples(features, labels)

    y_lower = 10
    plt.figure(figsize=(10, 6))
    for i in range(n_clusters):
        ith_cluster_silhouette_values = silhouette_vals[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # Espacio entre clusters

    plt.title(f"Gráfico de Silhouette para {n_clusters} Clusters")
    plt.xlabel("Coeficiente de Silhouette")
    plt.ylabel("Índice de Clusters")
    plt.axvline(x=silhouette_vals.mean(), color="red", linestyle="--")
    plt.savefig(output_path)  # Guardar gráfico
    print(f"Gráfico de Silhouette guardado en {output_path}")
    plt.close()

if __name__ == "__main__":
    file_path = "data/processed/alojamientos_booking_Granada_2024_cleaned.csv"
    caso_de_uso_1(file_path, price_percentile=0.6)



