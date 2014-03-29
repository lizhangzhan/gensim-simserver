from pymongo import MongoClient

class MongoCorpus(object):
    """ A MongoDB container for the corpus to train/index. """
    
    def __init__(self, server='localhost', db='wordizdb4', collection='article'):
        self.connect(server, db, collection)
    
    def connect(self, server, db, collection):
        self.client = MongoClient()
        self.db = self.client[db]
        self.corpus = self.db[collection]
    
    def use_collection(self, collection):
        self.corpus = self.db[collection]
        return str(self.corpus)
        
    def get_all_docs(self):
        it = self.corpus.find()
        return it['result']
    
    def get_doc_by_ids(self, docids):
        for docid in docids:
            doc = self.corpus.find_one({'id' : docid }, {'_id' : 0, 'id' : 1, 'tokens' : 1})
            yield doc
    
    def find(self, *args, **kargs):
        return self.corpus.find(*args, **kargs)
    
    def status(self):
        info = {}
        info['database'] = str(self.db)
        info['corpus'] = str(self.corpus)
        
        return info
    
    def __iter__(self):
        return self.corpus.find({}, {'_id':0, 'id':1, 'tokens':1})
    
class SQLCorpus(object):
    """ Class to train a model on a corpus in a relational db. """
    pass