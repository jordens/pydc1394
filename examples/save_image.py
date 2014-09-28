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



import sys
from time import sleep
from pydc1394 import Camera
import Image

print("Opening camera!")
cam0 = Camera()

print("Vendor:", cam0.vendor)
print("Model:", cam0.model)
print("GUID:", cam0.guid)
print("Mode:", cam0.mode)
print("Framerate: ", cam0.rate)
print("Available modes", cam0.modes)
print("Available features", cam0.features)

#those can be automated, the other are manual
try:
    cam0.brightness.mode = 'auto'
    cam0.exposure.mode = 'auto'
    cam0.white_balance.mode = 'auto'
except AttributeError: # thrown if the camera misses one of the features
    pass


for feat in cam0.features:
    print("%s (cam0): %s" % (feat,cam0.__getattribute__(feat).value))

#choose color mode
print(cam0.modes)
cam0.mode = cam0.modes[0]

cam0.start_capture()
cam0.start_one_shot()
matrix = cam0.dequeue()
print("Shape:", matrix.shape)
i = Image.fromarray(matrix.copy())
matrix.enqueue()
cam0.stop_one_shot()
cam0.stop_capture()
i.save("t.bmp")

i.show()
