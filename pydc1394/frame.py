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

from pydc1394._dc1394core import video_mode_vals, color_coding_vals, _dll
from ctypes import ARRAY, c_byte
from numpy import frombuffer, ndarray


class Frame(ndarray):
    """
    A frame returned by the :meth:`pydc1394.camera2.Camera.dequeue`.

    All metadata are retained as attributes of the resulting image.

    .. warning::
       This instance references the original frame data in the DMA
       buffer. Call :meth:`pydc1394.camera2.Camera.enqueue` with this
       frame as soon as possible after copying or processing the data or
       else the DMA buffer will starve. Do not touch the original array
       after having enqueued it.
   
    The methodology of subclassing ndarray is according to
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html .
    """

    def __new__(cls, frame): 
        """
        Convert a dc1394 frame into an Frame instance.
        """
        dtyp = ARRAY(c_byte, frame.contents.image_bytes)
        buf = dtyp.from_address(frame.contents.image)
        width, height = frame.contents.size
        pixels = width*height
        endianess = frame.contents.little_endian and "<" or ">"
        typ_string = "%su%i" % (endianess,
                frame.contents.image_bytes/pixels)

        img = ndarray.__new__(cls, shape=(height, width),
                dtype=typ_string, buffer=buf)

        img.frame_id = frame.contents.id
        img.frames_behind = frame.contents.frames_behind
        img.position = frame.contents.position
        img.packet_size = frame.contents.packet_size
        img.packets_per_frame = frame.contents.packets_per_frame
        img.timestamp = frame.contents.timestamp
        img.video_mode = video_mode_vals[frame.contents.video_mode]
        img.data_depth = frame.contents.data_depth
        img.color_coding = color_coding_vals[frame.contents.color_coding]
        img.color_filter = frame.contents.color_filter
        img.yuv_byte_order = frame.contents.yuv_byte_order
        img.stride = frame.contents.stride

        img._frame = frame

        return img

    def __array_finalize__(self, img):
        """
        Finalize the new Image class array.

        If called with an image object, inherit the properties of that image.
        """
        if img == None: return
        for key in ["position", "color_coding", "color_filter",
                    "yuv_byte_order", "stride", "packet_size",
                    "packets_per_frame", "timestamp", "frames_behind",
                    "frame_id", "data_depth", "video_mode", "_frame"]:
            setattr(self, key, getattr(img, key, None))

    def to_rgb(self):
        """
        Convert the image to an RGB image.
        
        Array shape is: (image.shape[0], image.shape[1], 3)
        Uses the dc1394_convert_to_RGB() function for the conversion.
        """
        res = ndarray(3*self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        _dll.dc1394_convert_to_RGB8(inp, res, 
                shape[1], shape[0], self.yuv_byte_order,
                self.color_coding, self.data_depth)
        res.shape = shape[0], shape[1], 3
        return res
    
    def to_mono8(self):
        """
        Convert he image to 8 bit gray scale.

        Uses the dc1394_convert_to_MONO8() funciton
        """
        res = ndarray(self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        _dll.dc1394_convert_to_MONO8(inp, res,
                shape[1], shape[0], self.yuv_byte_order,
                self.color_coding, self.data_depth)
        res.shape = shape
        return res

    def to_yuv422(self):
        """
        Convert he image to YUV422 color format. 

        Uses the dc1394_convert_to_YUV422() function
        """
        res = ndarray(self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        _dll.dc1394_convert_to_YUV422(inp, res,
                shape[1], shape[0], self.yuv_byte_order,
                self.color_coding, self.data_depth)
        # FIXME: untested
        return ndarray(shape=shape, buffer=res.data, dtype='u2')
