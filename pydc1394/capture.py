#!/usr/bin/env python -tt
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


from numpy import *

def caputure(nframes, cam0, cam1 ):

    a1 = empty( (nframes,480,640), dtype="u1")
    a2 = empty( (nframes,480,640), dtype="u1")

    cam0.start(); cam1.start()

    ldiff = 100000000
    while 1:
        k,t1 = cam0.shot()
        k,t2 = cam1.shot()
        diff = abs(t1-t2)
        if diff > ldiff:
            break
        ldiff = diff
        cam0.shot()
    cam1.shot()

    for i in xrange(nframes):
        a1[i],t1 = cam0.shot()
        a2[i],t2 = cam1.shot()
        print t1-t2

    cam0.stop(); cam1.stop()

    return a1, a2

if __name__ == '__main__':
    import sys
    import time

    from camera import Camera, DC1394Library

    def main():
        l = DC1394Library()
        guid = l.enumerate_cameras()[0]['guid']

        cam = Camera(l, guid=guid, mode=(640, 480, "Y8"), framerate=60.)


        n = 1000
        print "Acquiring %i frame ... " % n
        cam.start()
        t = time.time()
        for i in xrange(n):
            i = cam.shot()
        rtime = time.time() - t
        cam.stop()

        fps = n/rtime
        print "fps: %s" % (fps)

    main()


