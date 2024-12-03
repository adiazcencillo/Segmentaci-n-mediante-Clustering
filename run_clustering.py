from src.data_preparation import *

def main():
    # Cargar datos
    raw_data_path = "data/raw/alojamientos_booking_Granada_2024.csv"
    processed_data_path = "data/processed/alojamientos_booking_Granada_2024_cleaned.csv"
    data = load_data(raw_data_path)

    if data is not None:
        # Limpiar datos
        data = clean_data(data)

        # Normalizar todas las columnas numéricas
        data = normalize_data(data)

        # Guardar los datos procesados
        save_processed_data(data, processed_data_path)

if __name__ == "__main__":
    main()
