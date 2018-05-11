import bz2
import functools
from multiprocessing import Pool
import os
import re
from sys import getsizeof

from gensim.models.phrases import Phraser
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import matplotlib.pyplot as plt
from nltk import sent_tokenize, download
import numpy as np
from sklearn.manifold import TSNE
from tqdm import tqdm

from word2vec_generation import wiki_yielder
from wikicorpus import filter_wiki, tokenize

tqdm = functools.partial(tqdm, unit_scale=True)


class EpochLogger(CallbackAny2Vec):
    "Callback to log information about word2vec training"

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

def collect_sentences_worker(text, TOK, phraser):
    if text is None: return None
    if not text.strip(): return None
    if len(tokenize(filter_wiki(text))) <= 100: return None
    return tuple(
        (RES, SENT)
        for SENT in sent_tokenize(filter_wiki(text))
        for RES in set(phraser[tokenize(SENT)]).intersection(TOK)
        if SENT[0] not in {"}", "{", ".", "|", "*", "[", "]", "="}
        and len(phraser[tokenize(SENT)]) >= 11
    )

def collect_sentences(corpus, tokens=()):
    """
    Iterate through the Wikipedia corpus, extract sentences
    containing any of the tokens in the frozenset "tokens",
    and save the sentences (one per line) to their own file
    in ../wiki/sents/{token}.txt.bz2.

    :param corpus: .xml.bz2 file with Wikipedia data in it.
    :param tokens: iterable or array-like of tokens to find (case-insensitive).
    :return: 0 on success
    """
    if not os.path.isdir("../wiki/sents"): os.mkdir("../wiki/sents")
    # open files for writing sentences out to
    files = {
        i:bz2.open(f"../wiki/sents/{i}.txt.bz2", "wt", encoding="utf8")
        for i in tokens
    }
    TOK = set(i.lower() for i in tokens)
    worker = functools.partial(
        collect_sentences_worker,
        TOK=TOK,
        phraser=Phraser.load("../models/phraser")
    )

    with Pool(3, maxtasksperchild=1000) as P, open("NUMLINES", "w+") as FNUM:
        try: c = int(FNUM.read().strip())
        except: c = 0
        wiki = tqdm(wiki_yielder(bz2.open(corpus, "rt", encoding="utf8")))
        for i in P.imap_unordered(worker, wiki, chunksize=5000):
            if i is None:
                del i
                continue
            for j in range(len(i)):
                try: files[i[j][0]].write(i[j][1] + "\n")
                except: pass
            del i
        print("Done.  Closing pool...")
    print("Closing files...")
    for i in files: files[i].close()

def get_doc_vectors_file_worker(file):
    """
    Given a file, process the sentences one line at a time.  Get
    the document vectors, context vectors, and before/after vectors
    for *each instance of the token*.  Save these results to file.

    :param text:
    :param w2v:
    :return:
    """
    w2v = Word2Vec.load("../models/word2vec.w2v")
    phraser = Phraser.load("../models/phraser")
    tok = os.path.split(file)[1].split(".")[0].split("_")[0]
    total = {
        '../wiki/sents\\all_filtered.txt.bz2': 3354211,
        '../wiki/sents\\after_filtered.txt.bz2': 3564024,
        '../wiki/sents\\also_filtered.txt.bz2': 4605826,
        '../wiki/sents\\before_filtered.txt.bz2': 1516974,
        '../wiki/sents\\both_filtered.txt.bz2': 1445147,
        '../wiki/sents\\later_filtered.txt.bz2': 1619093,
        '../wiki/sents\\next_filtered.txt.bz2': 548316,
        '../wiki/sents\\but_filtered.txt.bz2': 5549017,
        '../wiki/sents\\and_filtered.txt.bz2': 9320539,
        '../wiki/sents\\not_filtered.txt.bz2': 6171279,
        '../wiki/sents\\too_filtered.txt.bz2': 484836,
        '../wiki/sents\\then_filtered.txt.bz2': 1999212,
        '../wiki/sents\\or_filtered.txt.bz2': 6673443
    }
    total = total[file]

    if not os.path.isfile(f"../wiki/vecs/{tok} docs.npy"):
        with bz2.open(file, "rt", encoding="utf8") as FIN:
            # document vecs
            vecs = np.array(
                [
                    np.sum([
                        w2v.wv[_]
                        if _ in w2v.wv
                        else np.zeros(300)
                        for _ in phraser[tokenize(LINE)]
                    ],
                    axis=0,
                ) - w2v.wv[tok]
                for LINE in tqdm(FIN, desc=f"\"{tok:<10s}\" document vectors", mininterval=5, position=1, total=total)
            ])
            np.save(f"../wiki/vecs/{tok} docs.npy", vecs.astype(np.float32))
            del vecs

    if not os.path.isfile(f"../wiki/vecs/{tok} context.npy"):
        with bz2.open(file, "rt", encoding="utf8") as FIN:
            # context vecs
            vecs = np.array(
                [
                    np.sum([
                        w2v.wv[_]
                        if _ in w2v.wv
                        else np.zeros(300)
                        for _ in LINE[max(0, LINE.index(tok) - 5):min(LINE.index(tok)+6, len(LINE))]
                        ],
                        axis=0,
                    ) - w2v.wv[tok]
                    for LINE in map(
                        lambda x: phraser[tokenize(x)],
                        tqdm(FIN, desc=f"\"{tok:<10s}\" context vectors", mininterval=5, position=1, total=total)
                    )
                ])
            np.save(f"../wiki/vecs/{tok} context.npy", vecs.astype(np.float32))
            del vecs

    return 0

def vectorize_sentences():
    """
    Use the pre-trained word2vec model and compute four different
    vector sets from the text:
        Vector of the entire document, sans the token
        Vector of the text preceding the token
        Vector of the text following the token
        Vector of the 5-word window (same size used for training around the token

    :param how: str; "avg" to average context vectors, "sum" to add them.
    :return:
    """
    files = [i.path for i in os.scandir("../wiki/sents") if i.path.endswith("_filtered.txt.bz2")]
    with Pool(2) as P:
        for _ in P.imap_unordered(get_doc_vectors_file_worker, sorted(files, key=os.path.getsize)): pass


if __name__ == "__main__":
    collect_sentences(
        "../wiki/enwiki-latest-pages-articles.xml.bz2",
        # "also and too both all or but then after before next later not".split(),
        ["duck", "bass"]
    )
    vectorize_sentences()