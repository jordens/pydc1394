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


"""
Display an image using wxPython and PyOpenGL

Example contributed by Peter Liebetraut, modified
by Holger Rapp.
"""
import sys
import optparse


from pydc1394 import DC1394Library, Camera
from pydc1394.ui.wx import LiveCameraDisplay
from pydc1394.cmdline import add_common_options, handle_common_options

import wx
def main():
    p = optparse.OptionParser(usage="Usage: %prog [ options ]\n"
      "This program displays a live image of your camera")

    add_common_options(p)

    options, args = p.parse_args()

    l = DC1394Library()
    cam = handle_common_options(options,l)

    if cam:
        app = wx.PySimpleApp()
        LiveCameraDisplay(cam)
        app.MainLoop()

main()

