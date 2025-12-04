import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from datetime import datetime
import os  # <--- Necesario para crear carpetas


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
        labels = ['Incidencia Tipo 0', 'Incidencia Tipo 1']

        sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds',
                    xticklabels=labels,
                    yticklabels=labels)

        plt.title('Matriz de Confusión: Predicción de Tipos de Incidencia')
        plt.ylabel('Tipo Real')
        plt.xlabel('Tipo Predicho por el Modelo')

        plt.tight_layout()

        # --- CORRECCIÓN DEL ERROR ---
        # 1. Definir nombre de la carpeta (sin barra al principio para que sea relativa)
        folder_name = 'Images'

        # 2. Crear la carpeta si no existe
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"[Sistema] Carpeta '{folder_name}' creada.")

        # 3. Construir la ruta de forma segura para cualquier sistema operativo
        filepath = os.path.join(folder_name, 'resultado_incidencias.png')

        # 4. Guardar
        plt.savefig(filepath)
        print(f"[Visualización] Gráfica guardada correctamente en: {filepath}")

        # Mostrar ventana (bloqueante, cerrar para continuar si es necesario)
        plt.show()
