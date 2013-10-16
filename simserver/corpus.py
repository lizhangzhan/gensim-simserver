from pymongo import MongoClient

class MongoCorpus():
    """ A MongoDB container for the corpus to train/index. Used internally by Simserver. """
    
    def __init__(self, server='localhost', db='wordizdb4', collection='article'):
        self.connect(server, db, collection)
    
    def connect(self, server, db, collection):
        self.client = MongoClient()
        self.db = self.client[db]
        self.corpus = self.db[collection]
    
    def use_collection(self, collection):
        self.corpus = self.db[collection]
        
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