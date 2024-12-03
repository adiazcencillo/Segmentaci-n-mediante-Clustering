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

def clean_data(data):
    """Limpieza básica del dataset, incluyendo eliminación de duplicados y rellenado de valores faltantes con KNN."""
    # Eliminar duplicados
    data = data.drop_duplicates()
    
    # Comprobamos si hay valores faltantes en el dataset
    missing_data = data.isnull().sum()
    print("Valores faltantes por columna:")
    print(missing_data)

    # Rellenar valores faltantes usando KNN Imputer
    if missing_data.sum() > 0:
        print("Rellenando valores faltantes usando KNN...")
        imputer = KNNImputer(n_neighbors=5)
        data_imputed = imputer.fit_transform(data.select_dtypes(include=['float64', 'int64']))  # Solo numéricas
        data[data.select_dtypes(include=['float64', 'int64']).columns] = data_imputed
        print("Valores faltantes rellenados.")
    else:
        print("No hay valores faltantes en los datos.")
    
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
    data.to_csv(output_path, index=False)
    print(f"Datos procesados guardados en {output_path}")

