from sklearn.datasets import make_blobs
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from hdbscan.prediction import approximate_predict


if __name__ == "__main__":
    data = make_blobs(10000)[0]
    c = HDBSCAN(prediction_data=True)
    c = c.fit(data)
    c = approximate_predict(c, data)
    for i in c: print(i.shape)

    plt.scatter(data[:,0], data[:,1], c=c.labels*c.probabilities)
    plt.show()