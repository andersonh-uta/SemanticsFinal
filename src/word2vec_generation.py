import bz2
import functools
import json
from multiprocessing import Pool
import os
import re
import string
from sys import getsizeof
import xml.etree.ElementTree as etree

from gensim.models.word2vec import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils
from tqdm import tqdm

import wikicorpus as wik

tqdm = functools.partial(tqdm, unit_scale=True)
DUMP = "../wiki/lawiki-latest-pages-articles.xml"

START_TEXT = re.compile(r"^\s+<text .*?>")
END_TEXT = re.compile(r"</text>$")

class WikiCorpus:
    """
    Class for iterating through the Wikipedia corpus.  Only used
    for word2vec.
    """
    def __init__(self, infile):
        self.infile = infile
    def __iter__(self):
        with bz2.open(self.infile, "rt", encoding="utf8") as F:
            yield from (i.split() for i in tqdm(F, desc="Word2Vec Epoch"))


def file_yielder(infile, desc=""):
    """
    Quick utility function to yield over a file.

    :param infile: str; path to a file
    :return:
    """
    if infile[-4:] == ".bz2":
        with bz2.open(infile, "rt", encoding="utf8") as F:
            yield from tqdm(F, desc=desc)
    else:
        with open(infile, "r", encoding="utf8") as F:
            yield from tqdm(F, desc=desc)

def wiki_yielder(infile):
    """
    Yield <page> tags' text out of the Wikipedia corpus.
    :param infile: path to corpus
    :return: yields article texts
    """
    for i in etree.iterparse(infile):
        if i[1].tag[-4:] == "text":
            yield i[1].text
        i[1].clear()

def wiki_cleaner_worker(text):
    """
    Worker function to clean up a single Wiki article.
    Meant to be multiprocessed.

    :param text: wikipedia text to clean
    :return:
    """
    if text is None: return "none"
    return wik.tokenize(wik.filter_wiki(text), 1, 15, True)

def clean_wikipedia_mp(dumpfile, outfile, minlen=50):
    """
    Clean the Wikipedia dump using Gensim tools.
    Dump it to a file with one document per line as
    space-separated tokens, with multi-word phrases
    found.

    Multiprocessing version.

    :param dumpfile: .xml file of the Wikipedia dump
    :param outfile: location of the file to dump it to.
        Should end in .bz2 extension.
    :return: 0 on success
    """
    with Pool(3) as P, bz2.open(outfile, "wt", encoding="utf8") as F:
        c = 0
        for i in P.imap(
            wiki_cleaner_worker,
            tqdm(wiki_yielder(bz2.open(dumpfile, "rt", encoding="utf8"))),
            # chunksize=50,
        ):
            if len(i) >= minlen: F.write(" ".join(i) + "\n")

def make_phraser(infile):
    """
    Train the phraser object and save it.
    :param infile: path to xml file with the wikipedia dump
    :return:
    """
    p = Phrases(
        tqdm(
            (i.split() for i in file_yielder(infile)),
            desc="Phrase-finding"
        )
    )
    p = Phraser(p)
    p.save("../models/phraser")

    return 0

def phrasing_worker(text, p):
    """Worker function for multithreaded corpus phrasing"""
    return " ".join(p[text.split()]) + "\n"

def phrase_corpus_mp(infile, outfile, phraserfile):
    """
    Load a trained phraser object and apply it to the extracted
    wikipedia corpus text.

    :param infile: wikipedia xml file
    :param outfile: .bz2 archive file to save phrased text to
    :param phraserfile: gensim phraser file
    :return:
    """
    p = Phraser.load(phraserfile)
    worker = functools.partial(phrasing_worker, p=p)
    with Pool(3) as P, bz2.open(outfile, "wt", encoding="utf8") as F:
        res = P.imap(worker, file_yielder(infile), chunksize=1000)
        for i in tqdm(res, desc="Phrasing"):
            F.write(i)

    return 0

def phrase_corpus(infile, outfile, phraserfile):
    """
    Load a trained phraser object and apply it to the extracted
    wikipedia corpus text.

    :param infile: wikipedia xml file
    :param outfile: .bz2 archive file to save phrased text to
    :param phraserfile: gensim phraser file
    :return:
    """
    p = Phraser.load(phraserfile)
    with bz2.open(outfile, "wt", encoding="utf8") as F:
        for i in tqdm(file_yielder(infile), desc="Phrasing"):
            F.write(" ".join(p[i.split()]) + "\n")

    return 0

def make_word2vec(infile, save_to="../models/word2vec.w2v"):
    """
    Train a Word2Vec algorithm on the wikipedia corpus
    :param infile: path to the coprus, in format of one
        space-separated list of tokens per line
    :return: 0 on success
    """
    wiki = WikiCorpus(infile)
    w2v = Word2Vec(
        wiki,
        size=300,
        sg=0,
        window=5,
        workers=3,
        iter=5,
        min_count=20,
    )
    w2v.delete_temporary_training_data(replace_word_vectors_with_normalized=True)
    w2v.save(save_to)

    return 0

if __name__ == "__main__":
    # clean_wikipedia_mp(
    #     "../wiki/enwiki-latest-pages-articles.xml.bz2",
    #     "../wiki/en.txt.bz2"
    # )
    # make_phraser("../wiki/en.txt.bz2")
    # phrase_corpus_mp(
    #     "../wiki/en.txt.bz2",
    #     "../wiki/en_phrased.txt.bz2",
    #     "../models/phraser"
    # )
    make_word2vec("../wiki/en.txt.bz2")
