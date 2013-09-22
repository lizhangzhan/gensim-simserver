#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU AGPL v3 - http://www.gnu.org/licenses/agpl.html

"""
USAGE: %(program)s DATA_DIRECTORY

    Start a sample similarity server, register it with Pyro and leave it running \
as a daemon.

Example:
    python -m simserver.run_simserver /tmp/server
"""

from __future__ import with_statement

import logging
import os
import sys

import gensim
import simserver

import Pyro4

import rpyc
from rpyc.utils.server import ThreadedServer # or ForkingServer

class SimserverService(rpyc.Service):
    def __init__(self):
        self.service = simserver.SessionServer()

if __name__ == '__main__':
    logging.basicConfig(filename='simserver.log', format='%(asctime)s : %(levelname)s : %(module)s:%(lineno)d : %(funcName)s(%(threadName)s) : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    basename = sys.argv[1]
    
    server = ThreadedServer(SimserverService, port = 12345)
    server.start()

    logging.info("finished running %s" % program)
