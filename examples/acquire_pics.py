#!/usr/bin/env python
# encoding: utf-8
#
# This file is part of pydc1394.
# 
# pydc1394 is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# pydc1394 is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with pydc1394.  If not, see
# <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2009, 2010 by Holger Rapp <HolgerRapp@gmx.net>
# and the pydc1394 contributors (see README File)


import time
import optparse

from pydc1394 import DC1394Library, Camera
from pydc1394.ui.qt import LiveCameraWin
from pydc1394.cmdline import add_common_options, handle_common_options


def main():
    p = optparse.OptionParser(usage="Usage: %prog [ options ]\n"
      "Acquires 25 pictures and exits.")

    add_common_options(p)

    options, args = p.parse_args()

    l = DC1394Library()
    cam = handle_common_options(options,l)

    if cam:
        print "Starting camera in normal mode. We need to process all pictures!"
        cam.start(interactive=False)

        print "Better be quick now! All images are saved in a Queue object, if we"
        print "are not fast enough in acquisition, the queue will overrun and"
        print "the program will crash."

        print "Acquiring 25 images"
        for idx in range(1,26):
            i = cam.shot()
            print "The last picture %i was acquired at: " % idx, i.timestamp

        cam.stop()

        print "All done."

main()


