from collections import Counter

from sklearn.manifold import TSNE as skTSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.neighbors import RadiusNeighborsClassifier
import os
from MulticoreTSNE import MulticoreTSNE as TSNE
from tsne_animate import tsneAnimate
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist
from hdbscan import HDBSCAN
from hdbscan.prediction import approximate_predict

import umap

TSNE_METRIC = "cosine"
PCA_TSNE_METRIC = "cosine"
TSNE_PERPLEXITY = 30
TSNE_EXAGGERATION = 30
TSNE_LEARNING_RATE = 100
# set ARR_UPTO to None to get whole array
ARR_UPTO = 5_000_000
ARR_UPTO = 125_000

def load_tsne(
        arr,
        name,
        perplexity=30,
        exaggeration=12,
        learning_rate=200,
        metric="cosine",
):
    """
    Load the TSNE embedding if present, else generate it from the given array.

    :param name:
    :param arr:
    :param perplexity:
    :param exaggeration:
    :param learning_rate:
    :param metric:
    :return:
    """
    tsne_file = f"{name} (metric {metric}, perp {perplexity}, lr {learning_rate}, exag {exaggeration}) tsne"
    try:
        tsne = np.load(f"../wiki/embeddings/{tsne_file}.npy")
        print(f"Found file: {tsne_file}")
    except FileNotFoundError:
        print("File not found.  Running TSNE on array of shape", arr.shape)
        tsne_model = TSNE(
            perplexity=perplexity,
            early_exaggeration=exaggeration,
            verbose=5,
            metric=(lambda x: np.power(10, pdist(x, metric="cosine")) - 10),
            # metric="cosine",
            learning_rate=learning_rate,
            n_iter=1000,
            angle=0.5,
            n_jobs=3,
        )
        tsne = tsne_model.fit_transform(arr)
        np.save(f"../wiki/embeddings/{tsne_file}.npy", tsne)

    return tsne

def load_umap(
        arr,
        name="",
        n_neighbors=50,
        metric="cosine",
        n_epochs=None,
        min_dist=0.1,
        spread=1,
):
    """
    Load the UMAP embedding if present, else generate it from the given array.

    :param arr: array of points to embed
    :param name: the word whose vectors this represents (used for naming the file)
    :param n_neighbors: n_neighbors parameter in UMAP
    :param metric: metric parameter in UMAP
    :param n_epochs: n_epochs parameter in UMAP
    :param min_dist: min_dist parameter in UMAP
    :param spread: spread_parameter in UMAP
    :return:
    """
    umap_file = f"{name}, nn={n_neighbors}, metric={metric}, n_epochs={n_epochs}, min_dist={min_dist}, spread={spread} umap"
    try:
        umap_embed = np.load(f"../wiki/embeddings/{umap_file}.npy")
        print(f"Found file: {umap_file}")
    except FileNotFoundError:
        print("File not found.  Running UMAP on array of shape", arr.shape)
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            metric=metric,
            # metric=cospow,
            n_epochs=n_epochs,
            min_dist=min_dist,
            verbose=True,
            spread=spread,
        )
        umap_embed = umap_model.fit_transform(arr)
        np.save(f"../wiki/embeddings/{umap_file}.npy", umap_embed)

    return umap_embed

def hdbscan_plot(
        arr,
        name,
        hdbscan_metric="euclidean",
        min_cluster_size=50,
        tsne_or_umap="TSNE",
        bounds=((), ())
):
    """
    Do HDBSCAN clustering and coloring, filtering out low-frequency clusters.

    :param arr: array of points to cluster and then plot
    :param name: name of the file (used to greate the plot title)
    :param hdbscan_metric: distance metric for HDBSCAN (can leave as euclidean)
    :param min_cluster_size: min_clister_size parameter for hdbscan
    :param tsne_or_umap: "tsne" or "umap" or other embedding algorithm used
        to generate arr; used for plot title and filename.
    :return:
    """
    a = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        prediction_data=True,
    ).fit(arr)


    c_tsne = a.labels_
    counts = Counter(c_tsne)
    print(counts.most_common())
    c_tsne = [i if counts[i] >= 500 else -1 for i in c_tsne]
    enum = dict(map(reversed, enumerate(
        sorted(
            set(c_tsne),
            key=lambda x: counts[x],
            # reverse=True,
        )
    )))
    enum[-1] = -1
    c_tsne = np.array([enum[i] for i in c_tsne])

    # arr = arr[c_tsne != -1]
    # c_tsne = c_tsne[c_tsne != -1]
    # print(np.unique(c_tsne))

    plt.title(f"{name}, unclustered {tsne_or_umap} embedding")
    plt.scatter(arr[:, 0], arr[:, 1], s=1, alpha=0.05)
    if bounds[0]: plt.xlim(bounds[0])
    if bounds[1]: plt.ylim(bounds[1])
    plt.axes("off")
    plt.savefig(f"../out/{name} {tsne_or_umap}.png", dpi=600)
    plt.close()

    plt.title(f"{name}, HDBSCAN clustering\n({len(np.unique(c_tsne))} clusters)")
    plt.scatter(
        arr[:, 0],
        arr[:, 1],
        s=1,
        alpha=0.05,
        c=c_tsne,
        cmap="jet"
    )
    if bounds[0]: plt.xlim(bounds[0])
    if bounds[1]: plt.ylim(bounds[1])
    plt.axes("off")
    plt.savefig(f"../out/{name} {tsne_or_umap} with colors.png", dpi=600)
    plt.close()

    return a.labels_, a.probabilities_

def cospow(x, base=10):
    """
    custom distance metric
    :param x:
    :return:
    """
    return np.power(base, pdist(x, metric="cosine")) - base

if __name__ == "__main__":
    files = [
        i.path
        for i in os.scandir("../wiki/vecs")
        if "kpca" not in i.path
        and "inds" not in i.path
        and "docs" in i.path
    ]
    files = sorted(files, key=os.path.getsize)

    # Dict of word-specific HDBSCAN parameters.
    # Allows for more specific tunings.
    bounds = {
        '../wiki/vecs/too docs.npy':    ((-6,6), (-6,6.5)),
        '../wiki/vecs/next docs.npy':   ((-5,7), (-7,6.5)),
        '../wiki/vecs/both docs.npy':   ((-6,6), (-6,6)),
        '../wiki/vecs/before docs.npy': ((-7,5), (-6.5,7.5)),
        '../wiki/vecs/later docs.npy':  ((-4,9), (-5.5,6)),
        '../wiki/vecs/then docs.npy':   ((-5,5), (-8, 9)),
        '../wiki/vecs/all docs.npy':    ((-7,7), (-4,4)),
        '../wiki/vecs/after docs.npy':  ((-6,6), (-5,7)),
        '../wiki/vecs/also docs.npy':   ((-4.5, 4), (-8.5, 7)),
        '../wiki/vecs/but docs.npy':    ((-5,5), (-6,7)),
        '../wiki/vecs/not docs.npy':    ((-5,5), (-6,7)),
        '../wiki/vecs/or docs.npy':     ((-5,6), (-20,5)),
        '../wiki/vecs/and docs.npy':    ((-5,8), (-5,5)),
    }
    files = [
        '../wiki/vecs/too docs.npy',
        '../wiki/vecs/next docs.npy',
        '../wiki/vecs/both docs.npy',
        '../wiki/vecs/before docs.npy',
        '../wiki/vecs/later docs.npy',
        '../wiki/vecs/then docs.npy',
        '../wiki/vecs/all docs.npy',
        '../wiki/vecs/after docs.npy',
        '../wiki/vecs/also docs.npy',
        '../wiki/vecs/but docs.npy',
        '../wiki/vecs/not docs.npy',
        '../wiki/vecs/or docs.npy',
        '../wiki/vecs/and docs.npy',
    ]
    min_cluster_nums = {
        '../wiki/vecs/too docs.npy':    {"umap":500, "tsne":50},
        '../wiki/vecs/next docs.npy':   {"umap":40, "tsne":43},
        '../wiki/vecs/both docs.npy':   {"umap":50, "tsne":750},
        '../wiki/vecs/before docs.npy': {"umap":50, "tsne":750},
        '../wiki/vecs/later docs.npy':  {"umap":223, "tsne":250},
        '../wiki/vecs/then docs.npy':   {"umap":65, "tsne":150},
        '../wiki/vecs/all docs.npy':    {"umap":80, "tsne":75},
        '../wiki/vecs/after docs.npy':  {"umap":750, "tsne":250},
        '../wiki/vecs/also docs.npy':   {"umap":50, "tsne":150},
        '../wiki/vecs/but docs.npy':    {"umap":47, "tsne":50},
        '../wiki/vecs/not docs.npy':    {"umap":55, "tsne":90},
        '../wiki/vecs/or docs.npy':     {"umap":5, "tsne":1000},
        '../wiki/vecs/and docs.npy':    {"umap":170, "tsne":150},
    }

    for i in files:
        name = str(os.path.split(i)[1].split(".")[0])
        # arr = np.load(i)
        arr = np.arange(10)
        print(f"{i}: array of shape {arr.shape}")
        if arr.shape[0] > ARR_UPTO:
            inds = np.random.choice(np.arange(arr.shape[0]), size=ARR_UPTO, replace=False)
            arr = arr[inds]
            np.save(f"../wiki/vecs/{name} inds.npy", inds)

        # normalize so we can use linear kernels for speed
        # print("Norming...", end="", flush=True)
        # arr = np.nan_to_num(arr / np.linalg.norm(arr, axis=1, keepdims=True))
        # print("done.")

        U = load_umap(
            arr=arr,
            name=name,
            n_neighbors=30,
            metric="cosine",
            n_epochs=None,
            min_dist=0.1,
            spread=1,
        )
        T = load_tsne(
            arr=arr,
            name=name,
            perplexity=100,
            exaggeration=12,
            metric="cosine",
            learning_rate=200,
        )

        U_label, U_cluster = hdbscan_plot(U, name, min_cluster_size=min_cluster_nums[i]["umap"], tsne_or_umap="UMAP", bounds=bounds[i])
        T_label, T_cluster = hdbscan_plot(T, name, min_cluster_size=min_cluster_nums[i]["tsne"], tsne_or_umap="TSNE")
