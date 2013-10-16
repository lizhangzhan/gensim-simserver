

TOP_SIMS = 100 # when precomputing similarities, only consider this many "most similar" documents
SHARD_SIZE = 65536 # spill index shards to disk in SHARD_SIZE-ed chunks of documents
DEFAULT_NUM_TOPICS = 400 # use this many topics for topic models unless user specified a value
JOURNAL_MODE = 'OFF' # don't keep journals in sqlite dbs
MONGO_DB_NAME = 'wordizdb2'
COLLECTION_NAME = 'corpus'