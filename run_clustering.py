from src.data_preparation import *
from src.clustering import * 

def process_data(raw_path, processed_path):

    data = load_data(raw_path)

    if data is not None:
        # Limpiar datos
        data = clean_data(data)

        # Normalizar todas las columnas numéricas
        data = normalize_data(data)

        # Guardar los datos procesados
        save_processed_data(data, processed_path)


def main():
    raw_data_aloj = "data/raw/alojamientos_booking_Granada_2024.csv"
    processed_data_aloj = "data/processed/alojamientos_booking_Granada_2024_cleaned.csv"
    raw_data_booking = "data/raw/booking_Granada_2024.csv"
    processed_data_booking = "data/processed/booking_Granada_2024_cleaned.csv"
    # Cargar datos
    process_data(raw_data_aloj, processed_data_aloj)
    process_data(raw_data_booking, processed_data_booking)


    caso_de_uso_1(processed_data_aloj)
    



if __name__ == "__main__":
    main()
