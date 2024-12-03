import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Cargar datos desde un archivo CSV."""
    try:
        data = pd.read_csv(file_path ,sep= ';', encoding="iso-8859-1")
        print(f"Datos cargados correctamente desde {file_path}")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def clean_data(data):
    """Limpieza básica del dataset, incluyendo eliminación de duplicados, manejo de valores faltantes y eliminación de columnas con demasiados faltantes."""
    # Eliminar duplicados
    data = data.drop_duplicates()

    # Identificar columnas con más del 60% de valores faltantes
    missing_percentage = data.isnull().mean() * 100
    columns_to_drop = missing_percentage[missing_percentage > 60].index
    print(f"Eliminando columnas con más del 60% de valores faltantes: {columns_to_drop.tolist()}")
    data = data.drop(columns=columns_to_drop)

    # Comprobar valores faltantes restantes
    missing_data = data.isnull().sum()
    print("\nValores faltantes por columna después de eliminar columnas con demasiados faltantes:")
    print(missing_data)

    # Rellenar valores faltantes para variables numéricas usando KNN
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    if data[numeric_columns].isnull().sum().sum() > 0:  # Verificar si hay valores faltantes en columnas numéricas
        print("\nRellenando valores faltantes en variables numéricas con KNN...")
        knn_imputer = KNNImputer(n_neighbors=5)
        data[numeric_columns] = knn_imputer.fit_transform(data[numeric_columns])
        print("Valores numéricos faltantes rellenados.")

    # Rellenar valores faltantes en variables categóricas con la moda
    categorical_columns = data.select_dtypes(include=['object']).columns
    if data[categorical_columns].isnull().sum().sum() > 0:  # Verificar si hay valores faltantes en columnas categóricas
        print("\nRellenando valores faltantes en variables categóricas con la moda...")
        most_frequent_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_columns] = most_frequent_imputer.fit_transform(data[categorical_columns])
        print("Valores categóricos faltantes rellenados.")

    print("\nLimpieza completada.")
    return data


def normalize_data(data):
    """Normalizar todas las columnas numéricas del dataset."""
    # Seleccionamos solo las columnas numéricas
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()

    # Normalizamos las columnas numéricas
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print(f"Columnas numéricas normalizadas: {numeric_columns.tolist()}")

    return data

def save_processed_data(data, output_path):
    """Guardar los datos procesados en un archivo CSV."""
    data.to_csv(output_path, index=False, sep=';')
    print(f"Datos procesados guardados en {output_path}")

