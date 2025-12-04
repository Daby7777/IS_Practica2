from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class MLStrategy(ABC):
    """
    Interfaz Strategy. Define el contrato para cualquier algoritmo de predicción de incidencias.
    """
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

class RandomForestStrategy(MLStrategy):
    """
    Opción A: Random Forest.
    Bueno para manejar ruido y relaciones no lineales en los voltajes.
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X, y):
        print("[Modelo RF] Entrenando Random Forest para clasificación de incidencias...")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class SVMStrategy(MLStrategy):
    """
    Opción B: Support Vector Machine.
    Algoritmo alternativo para comparar y cumplir el requisito de 'Selección'.
    """
    def __init__(self):
        # Kernel 'rbf' es estándar para datos complejos
        self.model = SVC(kernel='rbf', random_state=42)

    def train(self, X, y):
        print("[Modelo SVM] Entrenando Support Vector Machine...")
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
