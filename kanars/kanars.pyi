class NeuralNet:
    """
    Represents a Neural Network.
    """
    def __init__(self) -> None: ...
    def model(self, layers: list[int]) -> NeuralNet:
        """
        Initializes the model with specific layers.
        """

    def train(
        self,
        xis: list[list[float]],
        y_target: list[float],
        nb_iter: int,
        learning_rate: float,
        debug: bool,
    ) -> NeuralNet:
        """
        Trains the Neural Network: forward, backward, gradient descent, parameters update.
        """

    def predict(self, xis: list[list[float]]) -> list[float]:
        """
        Predicts the Xs input.
        Must be called after the train method.
        """

    def summary(self) -> NeuralNet:
        """
        Shows the Neural Network model.
        """

    def draw(self, filename: str) -> NeuralNet:
        """
        Draws in a file the model using dot.
        To export in svg, use the command: dot -Tsvg filename.dot > filename.svg
        """

def hello() -> None:
    """
    Says "hello" in duck language
    """
