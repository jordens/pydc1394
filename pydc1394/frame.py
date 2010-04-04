#!/usr/bin/python
# encoding: utf-8
# Copyright 2010 Robert Jordens <jordens@phys.ethz.ch>
#
# This file is part of pydc1394.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA  02110-1301  USA

from pydc1394._dc1394core import video_mode_vals
from ctypes import ARRAY, c_byte
from numpy import frombuffer, ndarray

class Frame(ndarray):
    frame_id = None
    frames_behind = None
    packet_size = None
    position = None
    packets_per_frame = None
    timestampe = None
    video_mode = None
    data_depth = None
    timestamp = None
    corrupt = None

    @classmethod
    def from_dc1394(cls, frame):
        """
        Convert a dc1394 frame into an Frame instance.

        All metadata are retained as attributes of the resulting image.
        """
        dtyp = ARRAY(c_byte, frame.contents.image_bytes)
        buf = dtyp.from_address(frame.contents.image)
        pixs = frame.contents.size[0]*frame.contents.size[1]
        end = frame.contents.little_endian and "<" or ">"
        typ_str = "%su%i" % (end, frame.contents.image_bytes/pixs)
        img = frombuffer(buf, dtype=typ_str)
        img = img.reshape(frame.contents.size[::-1]).copy().view(cls)
        img.frame_id = frame.contents.id
        img.frames_behind = frame.contents.frames_behind
        img.position = frame.contents.position
        img.packet_size = frame.contents.packet_size
        img.packets_per_frame = frame.contents.packets_per_frame
        img.timestamp = frame.contents.timestamp
        img.video_mode = video_mode_vals[frame.contents.video_mode]
        img.data_depth = frame.contents.data_depth
        return img

