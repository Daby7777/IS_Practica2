import pandas as pd
from datetime import datetime


class Lectura:
    """
    Clase encargada de la carga y transformación de los datos crudos (ETL).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def leerCSV(self):
        print(f"[DataLoader] Cargando datos desde {self.filepath}...")
        try:
            # Cargar CSV con delimitador de punto y coma
            df = pd.read_csv(self.filepath, sep=';')

            # Convertir columna de tiempo a datetime
            df['tiempo'] = pd.to_datetime(df['tiempo'], format='%d/%m/%Y %H:%M')

            # 1. Separar status (target) de voltajes (features)
            df_status = df[df['medida'] == 'status'][['tiempo', 'valor']].rename(columns={'valor': 'status'})
            df_voltages = df[df['medida'].str.contains('voltage')]

            # 2. Pivotar voltajes
            # Índice: tiempo, Columnas: medida + canal, Valores: valor
            df_pivot = df_voltages.pivot_table(
                index='tiempo',
                columns=['medida', 'canal'],
                values='valor'
            )

            # Aplanar los nombres de las columnas
            df_pivot.columns = [f'{col[0]}_{col[1]}' for col in df_pivot.columns]

            # 3. Unir con status
            dataset = df_pivot.join(df_status.set_index('tiempo'), how='inner')

            # Limpieza final
            dataset.dropna(inplace=True)

            self.data = dataset
            print(f"[DataLoader] Datos procesados. Muestras totales: {len(self.data)}")

            return self.data

        except Exception as e:
            print(f"[Error] Fallo en la carga de datos: {e}")
            return None
