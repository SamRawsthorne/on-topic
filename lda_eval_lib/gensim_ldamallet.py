#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2014 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


r"""Python wrapper for `Latent Dirichlet Allocation (LDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation>`_
from `MALLET, the Java topic modelling toolkit <http://mallet.cs.umass.edu/>`_
This module allows both LDA model estimation from a training corpus and inference of topic distribution on new,
unseen documents, using an (optimized version of) collapsed gibbs sampling from MALLET.
Notes
-----
MALLET's LDA training requires :math:`O(corpus\_words)` of memory, keeping the entire corpus in RAM.
If you find yourself running out of memory, either decrease the `workers` constructor parameter,
or use :class:`gensim.models.ldamodel.LdaModel` or :class:`gensim.models.ldamulticore.LdaMulticore`
which needs only :math:`O(1)` memory.
The wrapped model can NOT be updated with new documents for online training -- use
:class:`~gensim.models.ldamodel.LdaModel` or :class:`~gensim.models.ldamulticore.LdaMulticore` for that.
Installation
------------
Use `official guide <http://mallet.cs.umass.edu/download.php>`_ or this one ::
    sudo apt-get install default-jdk
    sudo apt-get install ant
    git clone git@github.com:mimno/Mallet.git
    cd Mallet/
    ant
Examples
--------
.. sourcecode:: pycon
    >>> from gensim.test.utils import common_corpus, common_dictionary
    >>> from gensim.models.wrappers import LdaMallet
    >>>
    >>> path_to_mallet_binary = "/path/to/mallet/binary"
    >>> model = LdaMallet(path_to_mallet_binary, corpus=common_corpus, num_topics=20, id2word=common_dictionary)
    >>> vector = model[common_corpus[0]]  # LDA topics of a documents
"""

import logging
import os
import random
import warnings
import tempfile
import xml.etree.ElementTree as et
import zipfile
from itertools import chain
import numpy
import subprocess

from gensim import utils, matutils
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import revdict

logger = logging.getLogger(__name__)

def silent_check_output(args, shell=False):
    """
    Runs a command in a subprocess, capturing stdout and discarding stderr.
    This ensures MALLET's logs won't appear in Jupyter.
    If you want to discard stdout as well, redirect it to subprocess.DEVNULL.
    """
    process = subprocess.Popen(args, shell=shell,
                               stdout=subprocess.PIPE,   # capture stdout
                               stderr=subprocess.DEVNULL # discard stderr
                              )
    out, err = process.communicate()
    if process.returncode != 0:
        # Optionally raise an error if needed
        raise RuntimeError(f"Command failed with return code {process.returncode}: {args}")
    return out

class LdaMallet(utils.SaveLoad, basemodel.BaseTopicModel):
    """
    Modified MALLET wrapper that:
      - accepts a `beta` parameter (in addition to `alpha`),
      - suppresses MALLET console logs (iterative <LL/token> logs).
    """

    def __init__(self,
                 mallet_path,
                 corpus=None,
                 num_topics=100,
                 alpha=None,
                 use_symmetric_alpha=False,
                 beta=0.01,          # <-- added param
                 id2word=None,
                 workers=4,
                 prefix=None,
                 optimize_interval=0,
                 iterations=1000,
                 topic_threshold=0.0,
                 random_seed=0):
        """
        Parameters
        ----------
        mallet_path : str
            Path to the mallet binary, e.g. '/home/username/mallet-2.0.7/bin/mallet'.
        corpus : iterable of iterable of (int, int), optional
            Collection of texts in BoW format.
        num_topics : int, optional
            Number of topics.
        alpha : float, optional
            Alpha parameter of LDA (scalar for a symmetric prior over document-topic probability).
            Default: 5.0 / num_topics
        use_symmetric_alpha : bool, optional
            If True, force MALLET to use a symmetric alpha (rather than optimizing per-topic).
        beta : float, optional
            Beta (a.k.a. eta) parameter for MALLET's topic-word distributions.
        id2word : gensim.corpora.Dictionary, optional
            Mapping between token ids and words.
        workers : int, optional
            Number of threads that will be used for training.
        prefix : str, optional
            Prefix for produced temporary files.
        optimize_interval : int, optional
            Optimize hyperparameters every `optimize_interval` iterations.
        iterations : int, optional
            Number of training iterations.
        topic_threshold : float, optional
            Threshold of the probability above which we consider a topic.
        random_seed : int, optional
            Random seed to ensure consistent results, if 0 - use system clock.
        """
        if beta == 'auto':
            beta = 0.01
        if alpha == 'auto':
            alpha = None

        self.mallet_path = mallet_path
        self.id2word = id2word
        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        else:
            self.num_terms = 0 if not self.id2word else 1 + max(self.id2word.keys())
        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.num_topics = num_topics
        self.topic_threshold = topic_threshold

        # MALLET's sum(alpha) logic
        if alpha is not None:
            self.alpha = alpha * num_topics
        else:
            mallet_default_sum_alpha = 5.0
            self.alpha = mallet_default_sum_alpha

        self.use_symmetric_alpha = use_symmetric_alpha
        self.beta = beta  # store the new param
        self.workers = workers
        self.optimize_interval = optimize_interval
        self.iterations = iterations
        self.random_seed = random_seed

        if prefix is None:
            rand_prefix = hex(random.randint(0, 0xffffff))[2:] + '_'
            prefix = os.path.join(tempfile.gettempdir(), rand_prefix)
        self.prefix = prefix

        # We keep the name "eta" in code to maintain backward-compat in places, but
        # we actually pass self.beta as MALLET's --beta. The original code used `eta=0.01`.

        self.eta = self.beta

        if corpus is not None:
            self.train(corpus)

    # ---- Filenames for MALLET intermediate data ----

    def finferencer(self):
        return self.prefix + 'inferencer.mallet'

    def ftopickeys(self):
        return self.prefix + 'topickeys.txt'

    def fstate(self):
        return self.prefix + 'state.mallet.gz'

    def fdoctopics(self):
        return self.prefix + 'doctopics.txt'

    def fcorpustxt(self):
        return self.prefix + 'corpus.txt'

    def fcorpusmallet(self):
        return self.prefix + 'corpus.mallet'

    def fwordweights(self):
        return self.prefix + 'wordweights.txt'

    # ---- Converting corpus to MALLET input format ----

    def corpus2mallet(self, corpus, file_like):
        """
        Convert `corpus` to Mallet format and write it to `file_like`.
        Format:
            doc_id 0 token1 token1 token2 token3 ...
        """
        for docno, doc in enumerate(corpus):
            tokens = []
            for tokenid, cnt in doc:
                word = str(tokenid)
                if self.id2word:
                    word = self.id2word.get(tokenid, str(tokenid))
                tokens.extend([word] * int(cnt))
            file_like.write(utils.to_utf8(f"{docno} 0 {' '.join(tokens)}\n"))

    def convert_input(self, corpus, infer=False, serialize_corpus=True):
        """Convert corpus to Mallet format and save it to disk."""
        if serialize_corpus:
            logger.info("serializing temporary corpus to %s", self.fcorpustxt())
            with utils.open(self.fcorpustxt(), 'wb') as fout:
                self.corpus2mallet(corpus, fout)

        # convert the text file above into MALLET's internal format
        if infer:
            cmd = (
                f"{self.mallet_path} import-file --preserve-case --keep-sequence "
                f"--remove-stopwords --token-regex \"\\S+\" --input {self.fcorpustxt()} "
                f"--output {self.fcorpusmallet() + '.infer'} "
                f"--use-pipe-from {self.fcorpusmallet()}"
            )
        else:
            cmd = (
                f"{self.mallet_path} import-file --preserve-case --keep-sequence "
                f"--remove-stopwords --token-regex \"\\S+\" --input {self.fcorpustxt()} "
                f"--output {self.fcorpusmallet()}"
            )
        logger.info("converting temporary corpus to MALLET format with %s", cmd)
        silent_check_output(args=cmd, shell=True)

    # ---- Training ----

    def train(self, corpus):
        """
        Train Mallet LDA with the current hyperparameters and the given `corpus`.
        """
        self.convert_input(corpus, infer=False)
        cmd = (
            f"{self.mallet_path} train-topics "
            f"--input {self.fcorpusmallet()} "
            f"--num-topics {self.num_topics} "
            f"--alpha {self.alpha} "
            f"--beta {self.beta} "            # <-- use self.beta instead of the default 0.01
            f"--use-symmetric-alpha {str(self.use_symmetric_alpha)} "
            f"--optimize-interval {self.optimize_interval} "
            f"--num-threads {self.workers} "
            f"--output-state {self.fstate()} "
            f"--output-doc-topics {self.fdoctopics()} "
            f"--output-topic-keys {self.ftopickeys()} "
            f"--num-iterations {self.iterations} "
            f"--inferencer-filename {self.finferencer()} "
            f"--doc-topics-threshold {self.topic_threshold} "
            f"--random-seed {str(self.random_seed)} "
            f"--show-topics-interval 0"  # turn off MALLET's own console progress
        )
        logger.info("training MALLET LDA with %s", cmd)
        silent_check_output(args=cmd, shell=True)

        # load word-topic counts
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics  # for backward compat

    # ---- Inference on new docs ----

    def __getitem__(self, bow, iterations=100):
        """
        Get LDA topic vector for new doc(s).
        """
        is_corpus, corpus = utils.is_corpus(bow)
        if not is_corpus:
            bow = [bow]  # wrap single doc in a list

        self.convert_input(bow, infer=True)
        cmd = (
            f"{self.mallet_path} infer-topics "
            f"--input {self.fcorpusmallet() + '.infer'} "
            f"--inferencer {self.finferencer()} "
            f"--output-doc-topics {self.fdoctopics() + '.infer'} "
            f"--num-iterations {iterations} "
            f"--doc-topics-threshold {self.topic_threshold} "
            f"--random-seed {str(self.random_seed)}"
        )
        logger.info("inferring topics with MALLET LDA '%s'", cmd)
        silent_check_output(args=cmd, shell=True)
        result = list(self.read_doctopics(self.fdoctopics() + '.infer'))
        return result if is_corpus else result[0]

    # ---- Loading / Parsing outputs ----

    def load_word_topics(self):
        """
        Load words X topics matrix from MALLET's state file.
        """
        logger.info("loading assigned topics from %s", self.fstate())
        word_topics = numpy.zeros((self.num_topics, self.num_terms), dtype=numpy.float64)
        if hasattr(self.id2word, 'token2id'):
            word2id = self.id2word.token2id
        else:
            word2id = revdict(self.id2word)

        with utils.open(self.fstate(), 'rb') as fin:
            # first lines are: <header>, the alpha line, the beta line
            _ = next(fin)  # header
            self.alpha = numpy.fromiter(next(fin).split()[2:], dtype=float)
            _ = next(fin)  # beta line, but we ignore it

            for line in fin:
                line = utils.to_unicode(line)
                doc, source, pos, typeindex, token, topic = line.split(" ")
                if token not in word2id:
                    continue
                tokenid = word2id[token]
                word_topics[int(topic), tokenid] += 1.0
        return word_topics

    def load_document_topics(self):
        """
        Load doc-topic distributions from the main doctopics file.
        """
        return self.read_doctopics(self.fdoctopics())

    def get_topics(self):
        """
        Return the topic-word distributions as a num_topics x num_terms matrix.
        """
        topics = self.word_topics
        return topics / topics.sum(axis=1)[:, None]

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """
        Get the `num_words` most probable words for `num_topics` number of topics.
        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            sort_alpha = self.alpha + 0.0001 * numpy.random.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[: num_topics // 2] + sorted_topics[-num_topics // 2:]
        shown = []
        for i in chosen_topics:
            if formatted:
                topic = self.print_topic(i, topn=num_words)
            else:
                topic = self.show_topic(i, topn=num_words)
            shown.append((i, topic))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic)
        return shown

    def show_topic(self, topicid, topn=10, num_words=None):
        """
        Get `topn` most probable words for the given `topicid`.
        """
        if num_words is not None:
            warnings.warn(
                "The parameter `num_words` is deprecated, will be removed in 4.0.0, use `topn` instead."
            )
            topn = num_words
        if self.word_topics is None:
            logger.warning("Run train or load_word_topics before showing topics.")
        topic = self.word_topics[topicid]
        topic = topic / topic.sum()  # normalize
        bestn = matutils.argsort(topic, topn, reverse=True)
        return [(self.id2word[idx], topic[idx]) for idx in bestn]

    def get_version(self, direc_path):
        """
        Get the version of Mallet by looking inside the zip or pom.xml.
        """
        try:
            archive = zipfile.ZipFile(direc_path, 'r')
            if 'cc/mallet/regression/' not in archive.namelist():
                return '2.0.7'
            else:
                return '2.0.8RC3'
        except Exception:
            xml_path = direc_path.split("bin")[0]
            try:
                doc = et.parse(xml_path + "pom.xml").getroot()
                namespace = doc.tag[:doc.tag.index('}') + 1]
                return doc.find(namespace + 'version').text.split("-")[0]
            except Exception:
                return "Can't parse pom.xml version file"

    def read_doctopics(self, fname, eps=1e-6, renorm=True):
        """
        Parse MALLET's doc-topics output into a list of (topic, weight) pairs per doc.
        """
        mallet_version = self.get_version(self.mallet_path)
        with utils.open(fname, 'rb') as fin:
            for lineno, line in enumerate(fin):
                if lineno == 0 and line.startswith(b"#doc "):
                    continue
                parts = line.split()[2:]  # skip doc id, source
                if len(parts) == 2 * self.num_topics:
                    doc = [
                        (int(id_), float(weight)) for id_, weight in zip(*[iter(parts)] * 2)
                        if abs(float(weight)) > eps
                    ]
                elif len(parts) == self.num_topics and mallet_version != '2.0.7':
                    doc = [(id_, float(weight)) for id_, weight in enumerate(parts) if abs(float(weight)) > eps]
                else:
                    # fallback for older MALLET or tricky format
                    if mallet_version == "2.0.7":
                        count = 0
                        doc = []
                        while count < len(parts):
                            if float(parts[count]) == int(parts[count]):
                                if float(parts[count + 1]) > eps:
                                    doc.append((int(parts[count]), float(parts[count + 1])))
                                count += 2
                            else:
                                if float(parts[count]) - int(parts[count]) > eps:
                                    doc.append((int(parts[count]) % 10, float(parts[count]) - int(parts[count])))
                                count += 1
                    else:
                        raise RuntimeError("invalid doc topics format at line %i in %s" % (lineno + 1, fname))

                if renorm:
                    total_weight = float(sum(weight for _, weight in doc))
                    if total_weight:
                        doc = [(id_, float(weight) / total_weight) for id_, weight in doc]
                yield doc

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Load a previously saved model, with backward compat for `random_seed`.
        """
        model = super(LdaMallet, cls).load(*args, **kwargs)
        if not hasattr(model, 'random_seed'):
            model.random_seed = 0
        return model

def malletmodel2ldamodel(mallet_model, gamma_threshold=0.001, iterations=50):
    """
    Convert this custom LdaMallet to a gensim.models.ldamodel.LdaModel.
    """
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=mallet_model.beta,  # note: MALLET's 'beta' is gensim's 'eta'
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        dtype=numpy.float64
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim