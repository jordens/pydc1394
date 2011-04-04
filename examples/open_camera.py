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


from pydc1394.camera2 import Camera

print "Opening camera!"
cam0 = Camera()

print "Vendor:", cam0.vendor
print "Model:", cam0.model
print "GUID:", cam0.guid
print "Mode:", cam0.mode
print "Framerate: ", cam0.framerate.value

print "Acquiring one frame!"
cam0.start_capture()
cam0.start_one_shot()
i = cam0.dequeue()
print "Shape:", i.shape
print "Dtype:", i.dtype
i.enqueue()
cam0.stop_one_shot()
cam0.stop_capture()

print "All done!"
