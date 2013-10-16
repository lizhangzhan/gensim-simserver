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
import session_server

import Pyro4

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(module)s:%(lineno)d : %(funcName)s(%(threadName)s) : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running %s" % ' '.join(sys.argv))

    program = os.path.basename(sys.argv[0])

    # check and process input arguments
    if len(sys.argv) < 2:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    basename = sys.argv[1]
    simserver = session_server.SessionServer(basename)
    
    
    def getNS():
        try:
            return Pyro4.locateNS()
        except Pyro4.errors.NamingError:
            os.system("python -m Pyro4.naming -n 0.0.0.0 &")
            return Pyro4.locateNS()
    
    with getNS() as ns:
        with Pyro4.Daemon() as daemon:
            # register server for remote access
            servername = 'gensim.simserver'
            uri = daemon.register(simserver, servername)
            ns.remove(servername)
            ns.register(servername, uri)
            logging.info("%s registered with nameserver (URI '%s')" % (servername, uri))
            daemon.requestLoop()  

    logging.info("finished running %s" % program)
