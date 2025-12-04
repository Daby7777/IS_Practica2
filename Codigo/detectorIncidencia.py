from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from algorithim import MLStrategy


class DetectorIncidencia:
    """
    Contexto principal que ejecuta el pipeline de predicción.
    """

    def __init__(self, strategy: MLStrategy):
        self.strategy = strategy
        self.X_test = None
        self.y_test = None
        self.predictions = None

    def detectarIncidencia(self, dataset):
        # Separar Features (X) y Target (y)
        X = dataset.drop('status', axis=1)
        y = dataset['status']

        # División 80/20 según requisitos
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # Entrenamiento
        self.strategy.train(X_train, y_train)

        # Predicción sobre conjunto de test
        self.predictions = self.strategy.predict(self.X_test)

        return self.evaluate()

    def evaluate(self):
        acc = accuracy_score(self.y_test, self.predictions)
        report = classification_report(self.y_test, self.predictions)
        matrix = confusion_matrix(self.y_test, self.predictions)
        return {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": matrix
        }
