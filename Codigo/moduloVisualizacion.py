import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from datetime import datetime


class Observer(ABC):
    @abstractmethod
    def update(self, message):
        pass


class ConsoleLogger(Observer):
    def update(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        if not isinstance(message, dict):
            print(f"[{timestamp}] LOG: {message}")


class Visualizar(Observer):
    def update(self, result_data):
        if isinstance(result_data, dict) and "accuracy" in result_data:
            self._print_text_report(result_data)
            self._plot_confusion_matrix(result_data['confusion_matrix'])

    def _print_text_report(self, data):
        print("\n" + "=" * 50)
        print("    INFORME DE CLASIFICACIÓN DE INCIDENCIAS")
        print("=" * 50)
        print(f"Precisión (Accuracy): {data['accuracy']:.2%}")
        print("\nDetalle por Tipo de Incidencia:")
        print(data['report'])

    def _plot_confusion_matrix(self, matrix):
        plt.figure(figsize=(8, 6))

        # Etiquetas actualizadas al contexto de "Tipos de Incidencia"
        # Asumiendo que 0 y 1 son los dos tipos que menciona el enunciado
        labels = ['Incidencia Tipo 0', 'Incidencia Tipo 1']

        sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds',  # Cambio a Rojo (alerta/incidencia)
                    xticklabels=labels,
                    yticklabels=labels)

        plt.title('Matriz de Confusión: Predicción de Tipos de Incidencia')
        plt.ylabel('Tipo Real')
        plt.xlabel('Tipo Predicho por el Modelo')

        plt.tight_layout()
        plt.savefig('resultado_incidencias.png')
        print(f"[Visualización] Gráfica guardada como 'resultado_incidencias.png'")
        plt.show()
