from src.lectura import Lectura
from src.algorithim import RandomForestStrategy
from src.detectorIncidencia import DetectorIncidencia
from moduloVisualizacion import ConsoleLogger, Visualizar


class SystemController:
    def __init__(self):
        self.observers = []
        self.data_loader = None
        # Ya no necesitamos un diccionario de resultados comparativos

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, message):
        for observer in self.observers:
            observer.update(message)

    def run(self):
        self.notify("Iniciando Sistema de Detección con Random Forest...")

        # 1. Carga de Datos
        # Instanciamos la clase Lectura (antes DataLoader)
        self.data_loader = Lectura('../Data/Dataset-CV.csv')
        data = self.data_loader.leerCSV()

        if data is None:
            self.notify("Error: No se pudieron cargar los datos.")
            return

        # 2. CONFIGURACIÓN DE ESTRATEGIA (Solo Random Forest)
        # Instanciamos directamente la estrategia deseada
        strategy = RandomForestStrategy()

        # Inyectamos la estrategia en el contexto (DetectorIncidencia)
        predictor = DetectorIncidencia(strategy)

        # 3. EJECUCIÓN DEL PIPELINE
        self.notify("\n--- Ejecutando Modelo Random Forest ---")

        # Ejecutamos la detección (entrenamiento y test)
        metrics = predictor.detectarIncidencia(data)

        # 4. RESULTADOS FINALES
        # Notificamos los resultados a los observadores (Consola y Gráficas)
        self.notify(metrics)

        self.notify("\n" + "*" * 50)
        self.notify("Proceso finalizado con éxito.")
        self.notify("*" * 50)


if __name__ == "__main__":
    app = SystemController()

    # Añadimos los observadores
    app.attach(ConsoleLogger())
    app.attach(Visualizar())

    # Ejecutamos
    app.run()
