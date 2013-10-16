
import numpy
import gensim

import logging
logger = logging.getLogger('gensim.similarities.simserver')


class SimModel(gensim.utils.SaveLoad):
    """
    A semantic model responsible for translating between plain text and (semantic)
    vectors.

    These vectors can then be indexed/queried for similarity, see the `SimIndex`
    class. Used internally by `SimServer`.
    """
    def __init__(self, fresh_docs, dictionary=None, method=None, params=None):
        """
        Train a model, using `fresh_docs` as training corpus.

        If `dictionary` is not specified, it is computed from the documents.

        `method` is currently one of "tfidf"/"lsi"/"lda".
        """
        # FIXME TODO: use subclassing/injection for different methods, instead of param..
        self.method = method
        if params is None:
            params = {}
        self.params = params
        logger.info("collecting %i document ids" % len(fresh_docs))
        docids = fresh_docs.keys()
        logger.info("creating model from %s documents" % len(docids))
        preprocessed = lambda: (fresh_docs[docid]['tokens'] for docid in docids)

        # create id->word (integer->string) mapping
        logger.info("creating dictionary from %s documents" % len(docids))
        if dictionary is None:
            dictionary = gensim.corpora.Dictionary(preprocessed())
            if len(docids) >= 1000:
                dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=50000)
            else:
                logger.warning("training model on only %i documents; is this intentional?" % len(docids))
                dictionary.filter_extremes(no_below=0, no_above=1.0, keep_n=50000)
        self.dictionary = dictionary

        class IterableCorpus(object):
            def __iter__(self):
                for tokens in preprocessed():
                    yield dictionary.doc2bow(tokens)

            def __len__(self):
                return len(docids)
        corpus = IterableCorpus()

        if method == 'lsi':
            logger.info("training TF-IDF model")
            self.tfidf = gensim.models.TfidfModel(corpus, id2word=self.dictionary)
            logger.info("training LSI model")
            tfidf_corpus = self.tfidf[corpus]
            self.lsi = gensim.models.LsiModel(tfidf_corpus, id2word=self.dictionary, **params)
            self.lsi.projection.u = self.lsi.projection.u.astype(numpy.float32) # use single precision to save mem
            self.num_features = len(self.lsi.projection.s)
        elif method == 'lda_tfidf':
            logger.info("training TF-IDF model")
            self.tfidf = gensim.models.TfidfModel(corpus, id2word=self.dictionary)
            logger.info("training LDA model")
            self.lda = gensim.models.LdaModel(self.tfidf[corpus], id2word=self.dictionary, **params)
            self.num_features = self.lda.num_topics
        elif method == 'lda':
            logger.info("training TF-IDF model")
            self.tfidf = gensim.models.TfidfModel(corpus, id2word=self.dictionary)
            logger.info("training LDA model")
            self.lda = gensim.models.LdaModel(corpus, id2word=self.dictionary, **params)
            self.num_features = self.lda.num_topics
        elif method == 'logentropy':
            logger.info("training a log-entropy model")
            self.logent = gensim.models.LogEntropyModel(corpus, id2word=self.dictionary, **params)
            self.num_features = len(self.dictionary)
        else:
            msg = "unknown semantic method %s" % method
            logger.error(msg)
            raise NotImplementedError(msg)

    def doc2vec(self, doc):
        """Convert a single SimilarityDocument to vector."""
        bow = self.dictionary.doc2bow(doc['tokens'])
        if self.method == 'lsi':
            return self.lsi[self.tfidf[bow]]
        elif self.method == 'lda':
            return self.lda[bow]
        elif self.method == 'lda_tfidf':
            return self.lda[self.tfidf[bow]]
        elif self.method == 'logentropy':
            return self.logent[bow]

    def docs2vecs(self, docs):
        """Convert multiple SimilarityDocuments to vectors (batch version of doc2vec)."""
        bows = (self.dictionary.doc2bow(doc['tokens']) for doc in docs)
        if self.method == 'lsi':
            return self.lsi[self.tfidf[bows]]
        elif self.method == 'lda':
            return self.lda[bows]
        elif self.method == 'lda_tfidf':
            return self.lda[self.tfidf[bows]]
        elif self.method == 'logentropy':
            return self.logent[bows]

    def get_tfidf(self, doc):
        bow = self.dictionary.doc2bow(doc['tokens'])
        if hasattr(self, 'tfidf'):
            tmp = self.tfidf[bow]
            tfidf = [ (self.dictionary[word_id], score) for word_id, score in tmp ]
            return tfidf
        if hasattr(self, 'logent'):
            return self.logent[bow]
        raise ValueError("model must contain either TF-IDF or LogEntropy transformation")

    def close(self):
        """Release important resources manually."""
        pass

    def __str__(self):
        return "SimModel(method=%s, dict=%s)" % (self.method, self.dictionary)
#endclass SimModel
