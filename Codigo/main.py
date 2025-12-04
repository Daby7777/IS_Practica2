from lectura import Lectura
from algorithim import RandomForestStrategy, SVMStrategy
from detectorIncidencia import DetectorIncidencia
from moduloVisualizacion import ConsoleLogger, Visualizar


class SystemController:
    def __init__(self):
        self.observers = []
        self.data_loader = None
        self.results = {}

    def attach(self, observer):
        self.observers.append(observer)

    def notify(self, message):
        for observer in self.observers:
            observer.update(message)

    def run(self):
        self.notify("Iniciando Proceso de Selección de Modelo de Incidencias...")

        # 1. Carga de Datos
        self.data_loader = Lectura('Dataset-CV.csv')
        data = self.data_loader.leerCSV()

        if data is None:
            return

        # 2. SELECCIÓN DE ALGORITMO (Requisito ii)
        # Vamos a probar dos estrategias para ver cuál predice mejor los tipos
        strategies = [
            ("Random Forest", RandomForestStrategy()),
            ("SVM", SVMStrategy())
        ]

        best_score = 0
        best_model_name = ""

        for name, strategy in strategies:
            self.notify(f"\n--- Probando Estrategia: {name} ---")
            predictor = DetectorIncidencia(strategy)

            # Ejecuta el pipeline
            metrics = predictor.detectarIncidencia(data)

            # Notificamos resultados parciales
            self.notify(metrics)

            # Guardamos el mejor
            if metrics['accuracy'] > best_score:
                best_score = metrics['accuracy']
                best_model_name = name

        # 3. Conclusión Final
        self.notify("\n" + "*" * 50)
        self.notify(f"CONCLUSIÓN DE SELECCIÓN:")
        self.notify(f"El algoritmo más adecuado para estos datos es: {best_model_name}")
        self.notify(f"Precisión alcanzada: {best_score:.2%}")
        self.notify("*" * 50)


if __name__ == "__main__":
    app = SystemController()
    app.attach(ConsoleLogger())
    app.attach(Visualizar())
    app.run()
