from pytest import approx
import kanars as ks
import matplotlib.pyplot as plt
import sklearn.datasets as skd
import numpy as np


def make_y_minus_plus_1(y: np.ndarray) -> np.ndarray:
    # y is 0 then -1 else 1
    return y * 2 - 1


def plot_decision_boundary(
    model: ks.NeuralNet, x_data: np.ndarray, y_target: np.ndarray, filename: str
) -> None:
    # Set min and max values and give it some padding
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    h = 0.25

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]

    # Predict the function value for the whole grid
    scores = model.predict(Xmesh)
    Z = np.array([s > 0 for s in scores]).reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)  # type: ignore
    plt.scatter(x_data[:, 0], x_data[:, 1], c=y_target, cmap=plt.cm.Spectral)  # type: ignore
    plt.savefig(filename)
    plt.close()


def plot_3d(x_data: np.ndarray, filename: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    colors = np.zeros((len(x_data), 3))
    colors[x_data[:, 2] >= 0, :] = [0, 0, 1]  # Blue if >= 0
    colors[x_data[:, 2] < 0, :] = [1, 0, 0]  # Red otherwise

    ax.scatter(x_data[:, 0], x_data[:, 1], x_data[:, 2], c=colors)
    plt.savefig(filename)
    plt.close()


def test_hello() -> None:
    # Check no crash
    ks.hello()


def test_summary_draw() -> None:
    x_data = [[3.0, 2.0], [-2.0, -1.0]]
    y_target = [1.0, -1.0]

    nn = ks.NeuralNet()

    (
        nn.model(layers=[2, 2, 1]).train(
            x_data, y_target, nb_iter=5, learning_rate=0.01, debug=False
        )
    )

    # Check no crash
    nn.summary()
    nn.draw("test_kanars.dot")


def test_basic() -> None:
    x_data = [[3.0, 2.0], [-2.0, -1.0]]
    y_target = [1.0, -1.0]
    x_pred = [[3.5, 2.5], [-2.5, -1.5]]

    nn = ks.NeuralNet()

    y_pred = (
        nn.model(layers=[2, 2, 1])
        .train(x_data, y_target, nb_iter=5, learning_rate=0.01, debug=False)
        .predict(x_pred)
    )

    assert y_pred == [approx(0.6015727), approx(-0.5651625)]


def test_linear() -> None:
    x_data, y_target = skd.make_classification(
        n_samples=20,
        n_features=2,
        n_redundant=0,
        n_informative=1,
        n_clusters_per_class=1,
        random_state=12,
    )
    y_target = make_y_minus_plus_1(y_target)

    nn = ks.NeuralNet()

    nn.model(layers=[2, 2, 1]).train(
        x_data, y_target, nb_iter=100, learning_rate=0.01, debug=False
    )

    # Check no crash
    plot_decision_boundary(nn, x_data, y_target, "test_kanars_linear.png")


def test_moons() -> None:
    x_data, y_target = skd.make_moons(n_samples=100, noise=0.1, random_state=1)
    y_target = make_y_minus_plus_1(y_target)

    nn = ks.NeuralNet()

    nn.model(layers=[2, 4, 4, 1]).train(
        x_data, y_target, nb_iter=1000, learning_rate=0.01, debug=False
    )

    # Check no crash
    plot_decision_boundary(nn, x_data, y_target, "test_kanars_moons.png")


def test_circle() -> None:
    x_data, y_target = skd.make_circles(
        n_samples=100, factor=0.5, noise=0.1, random_state=3
    )
    y_target = make_y_minus_plus_1(y_target)

    nn = ks.NeuralNet()
    nn.model(layers=[2, 4, 4, 1]).train(
        x_data, y_target, nb_iter=1000, learning_rate=0.01, debug=False
    )

    # Check no crash
    plot_decision_boundary(nn, x_data, y_target, "test_kanars_circle.png")


def test_3d() -> None:
    x_data, y_target = skd.make_swiss_roll(n_samples=200, random_state=50)
    y_target = [-1 if x < 0 else 1 for x in x_data[:, 2]]

    nn = ks.NeuralNet()
    nn.model(layers=[3, 8, 4, 1]).train(
        x_data, y_target, nb_iter=500, learning_rate=0.001, debug=False
    )

    y_pred = nn.predict([[5, 3, -1], [-2, -4, 1]])

    assert y_pred[0] < -0.95
    assert y_pred[1] > 0.95

    plot_3d(x_data, "test_kanars_3d.png")
