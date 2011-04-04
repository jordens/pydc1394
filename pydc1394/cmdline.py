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


from __future__ import division

"""
Common cmdline arguments for programs related to cameras
"""

import optparse

from camera2 import Camera, Context

__all__ = [ "add_common_options", "handle_common_options" ]

def add_common_options(p):
    p.set_defaults(fps = None, shutter=None, gain=None, guid=None,
        mode = None, isospeed = 400)

    p.add_option("-l", "--list", action="store_true",
                 help="List all devices on the IEEE Bus")
    p.add_option("-c", "--cam", dest="guid", type="str",
                 help="Use the camera with the given GUID")
    p.add_option("-f", "--fps", dest="fps", type="float",
                 help="Use the given framerate")
    p.add_option("-m", "--mode", dest="mode", type="str",
                 help="Use the given mode (e.g. 640x480_Y8)",metavar="MODE")
    p.add_option("-s", "--shutter", dest="shutter", type="float",
                 help="Set the shutter (integration time) to this amount in ms")
    p.add_option("-g", "--gain", dest="gain", type="float",
                 help="Sets the gain to the given floating point value")
    p.add_option("-i", "--isospeed", dest="isospeed", type="int",
                 help="Choose isospeed [400,800]")

    return p

def handle_common_options(o):
    c = Context()

    if o.list:
    	print "   %s   %s" % ("GUID".center(20), "Unit No".center(20))
	for i, g in c.cameras:
            print "   %s   %s" % (hex(i).center(20), str(g).center(8))

    camera = Camera(context=c, guid=o.guid, mode=o.mode, rate=o.fps,
                    shutter=o.shutter, gain=o.gain, iso_speed=o.isospeed)
    camera.setup()

    return camera
