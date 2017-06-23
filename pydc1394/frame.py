# -*- coding: utf-8 -*-
#
# Copyright 2010 Robert Jordens <robert@joerdens.org>
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

from __future__ import (print_function, unicode_literals, division,
        absolute_import)

from ctypes import ARRAY, c_byte
from numpy import ndarray

from .dc1394 import *


__all__ = ["Frame"]


class Frame(ndarray):
    """
    A frame returned by the :meth:`pydc1394.camera2.Camera.dequeue`.

    All metadata are retained as attributes of the resulting image.

    .. warning::
       This instance references the original frame data in the DMA
       buffer. Call :meth:`pydc1394.camera2.Camera.enqueue` with this
       frame as soon as possible after copying or processing the data or
       else the DMA buffer will starve. Do not touch the original array
       data after having enqueued it.
   
    The methodology of subclassing ndarray is according to
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html .
    """

    def __new__(cls, camera, frame): 
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
        # save camera and frame for enqueue()
        img._frame = frame
        img._cam = camera
        return img

    def __array_finalize__(self, img):
        """
        Finalize the new Image class array.

        If called with an image object, inherit the properties of that image.
        """
        if img is None:
            return
        # do not inherit _frame and _cam since we also get called on copy()
        # and should not hold references to the frame in this case
        for key in ["position", "color_coding", "color_filter",
                    "yuv_byte_order", "stride", "packet_size",
                    "packets_per_frame", "timestamp", "frames_behind",
                    "frame_id", "data_depth", "video_mode"]:
            setattr(self, key, getattr(img, key, None))

    def enqueue(self):
        """
        Returns a frame to the ring buffer once it has been used.

        This method is also called implicitly on ``del``.

        Only call this method on the original frame obtained from
        :meth:`pydc1394.camera2.Camera.dequeue` and not on its views,
        new-from-templates or copies. Otheriwse an AttributeError will
        be raised.
        """
        if not hasattr(self, "_frame"): # or self.base is not None:
            raise AttributeError("can only enqueue the original frame")
        if self._frame is not None:
            dll.dc1394_capture_enqueue(self._cam, self._frame)
            self._frame = None
            self._cam = None

    # from contextlib iport closing
    # with closing(camera.dequeue()) as im:
    #   do stuff with im
    close = enqueue

    def __del__(self):
        try:
            self.enqueue()
        except AttributeError:
            pass

    @property
    def corrupt(self):
        """
        Is this frame corrupt?

        Returns ``True`` if the given frame has been detected to be
        corrupt (missing data, corrupted data, overrun buffer, etc.) and
        ``False`` otherwise.  
        
        .. note::
           Certain types of corruption may go undetected in which case
           ``False`` will be returned erroneously.  The ability to
           detect corruption also varies between platforms.
        
        .. note::
           Corrupt frames still need to be enqueued with :meth:`enqueue`
           when no longer needed by the user.
        """
        return bool(dll.dc1394_capture_is_frame_corrupt(
                    self._cam, self._frame))
   
    def to_rgb(self):
        """
        Convert the image to an RGB image.
        
        Array shape is: (image.shape[0], image.shape[1], 3)
        Uses the dc1394_convert_to_RGB() function for the conversion.
        """
        res = ndarray(3*self.size, dtype='u1')
        shape = self.shape
        inp = ndarray(shape=len(self.data), buffer=self.data, dtype='u1')
        dll.dc1394_convert_to_RGB8(inp, res, 
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
        dll.dc1394_convert_to_MONO8(inp, res,
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
        dll.dc1394_convert_to_YUV422(inp, res,
                shape[1], shape[0], self.yuv_byte_order,
                self.color_coding, self.data_depth)
        return ndarray(shape=shape, buffer=res.data, dtype='u2')
