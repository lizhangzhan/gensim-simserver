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
    simserver = simserver.SessionServer(basename)
    
    try:
        Pyro4.locateNS()
    except Pyro4.errors.NamingError:
        os.system("python -m Pyro4.naming -n 0.0.0.0 &")
    daemon = Pyro4.Daemon()                 # make a Pyro daemon
    uri = daemon.register(simserver, 'gensim.testserver')   # register the greeting object as a Pyro object
    print uri
    daemon.requestLoop()   

    logging.info("finished running %s" % program)
