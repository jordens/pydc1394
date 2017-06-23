# -*- coding: utf-8 -*
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

from ctypes import byref, POINTER, c_uint32, c_int32, c_float

from .dc1394 import *
from .frame import *



__all__ = ["Context", "Feature", "Trigger", "Whitebalance", "Whiteshading",
    "Mode", "Format7", "Exif", "Camera", "DC1394Error", "DC1394Exception",
    "Frame"]


class Context(object):
    """
    The DC1394 context.

    Each application should maintain one of these, especially if it
    wants to access several cameras. But as the :class:`Camera` objects
    will create a Context themselves if not supplied with one, it is not
    strictly necessary.

    Additionally, since the context needs to stay alive for the lifespan
    of the Camera objects, their health is enforced by the
    :class:`Camera` objects.

    The available camera GUIDs can be obtained from the :attr:`cameras`
    list. To obtain a :class:`Camera` object for a certain camera,
    either the :meth:`camera` method of a :class:`Context` object 
    can be used or the context can be passed to the :class:`Camera`
    constructor.
    """
    _handle = None

    def __init__(self):
        self._handle = dll.dc1394_new()

    def __del__(self):
        self.close()

    def close(self):
        """
        Frees the library and the dc1394 context.
        
        After calling this, all cameras in this context are invalid.
        """
        if self._handle is not None:
            dll.dc1394_free(self._handle)
        self._handle = None

    @property
    def cameras(self):
        """
        The list of cameras attached to the system. Read-only.

        Each item contains the GUID of the camera and the unit number.
        Pass a (GUID, unit) tuple of the list to camera_handle() to
        obtain a handle. Since a single camera can contain several
        functional units (think stereo cameras), the GUID is not enough
        to identify an IIDC camera.
        
        If present, multiple cards will be probed.
        """
        cam_list = POINTER(camera_list_t)()
        dll.dc1394_camera_enumerate(self._handle, byref(cam_list))
        cams = [(cam.guid, cam.unit) for cam in
                cam_list.contents.ids[:cam_list.contents.num]]
        dll.dc1394_camera_free_list(cam_list)
        return cams

    def camera_handle(self, guid, unit=None):
        """
        Obtain a camera handle given the GUID and optionally the unit
        number of the camera.
        
        Pass this handle to :class:`Camera` or to :meth:`camera`.

        A :class:`DC1394Exception` will be thrown if the requested
        camera is inaccessible.
        """
        if unit is None:
            handle = dll.dc1394_camera_new(
                    self._handle, guid)
        else:
            handle = dll.dc1394_camera_new_unit(
                    self._handle, guid, unit)
        if not handle:
            raise DC1394Exception("Couldn't access camera (%s, %s)!" % (
                    guid, unit))
        return handle
 
    def camera(self, guid, unit=None, **kwargs):
        """
        Obtain a :class:`Camera` instance for a given camera GUID.
        """
        handle = self.camera_handle(guid, unit)
        return Camera(context=self, handle=handle, **kwargs)


class Feature(object):
    """
    A feature of a dc1394 camera.
    
    Features have several readable and adjustable knobs and can refer to
    different elements in the camera.  In IIDC, 'features' refer to a
    number of image parameters that can be tuned like exposure, white
    balance, etc... Features are adjusted with a number of specific
    methods. The name of each feature is relatively easy to understand.
    The only trick is that the ``exposure`` feature is actually an auto
    exposure function. Its behaviour depends on the manufacturer.

    A feature can be activated or deactived via the :attr:`active`
    attribute if they are :attr:`switchable`.

    The :attr:`mode` of operation (``"manual"``, ``"auto"``, or
    ``"one_push"``) can be one of those in :attr:`modes`.

    The value is either given via the :attr:`value` or :attr:`absolute`
    attribute depending on whether the feature is
    :attr:`absolute_capable` and whether it is in
    :attr:`absolute_control`. The non-absolute value is an integer with
    a meaning that may be internal to the camera and vendor specific.
    For the actual units of :attr:`absolute` see the IIDC standard:
    Brightness (%), Exposure (EV), WhiteBalance (K), Hue (deg), 
    Saturation (%), Shutter (s), Gain (dB), Iris (F), Focus (m),
    Trigger (times, 1), TriggerDelay (s), FrameRate (fps), Zoom (power),
    Pan (deg), Tilt (deg).

    The absolute value must be in the :attr:`absolute_range`
    and the internal value must be in :attr:`value_range`.
    """

    def __init__(self, cam, feature_id):
        self._feature_id = feature_id
        self._cam = cam

    @property
    def name(self):
        """
        The name of this feature. Read-only.

        A camera object contains this feature as a named attribute.
        """
        return feature_vals[self._feature_id]

    @property
    def present(self):
        """
        Is the feature present on this camera? Read-only.
        """
        k = bool_t()
        dll.dc1394_feature_is_present(
                self._cam, self._feature_id, byref(k))
        return k.value

    @property
    def switchable(self):
        """
        Can the feature be activated and deactivated? Read-only.

        Use :attr:`active` to enable and disable this feature.
        """
        k = bool_t()
        dll.dc1394_feature_is_switchable(
                self._cam, self._feature_id, byref(k))
        return bool(k.value)

    @property
    def active(self):
        """
        Current activation state of the feature.
        """
        k = bool_t()
        dll.dc1394_feature_get_power(
                self._cam, self._feature_id, byref(k))
        return k.value

    @active.setter
    def active(self, value):
        dll.dc1394_feature_set_power(
            self._cam, self._feature_id, bool(value))

    @property
    def modes(self):
        """
        Containes the list of allowed modes of this feature. Read-only.

        Use :attr:`mode` to get or set the current mode.
        """
        modes = feature_modes_t()
        dll.dc1394_feature_get_modes(
                self._cam, self._feature_id, byref(modes))
        return [feature_mode_vals[i]
                for i in modes.modes[:modes.num]]

    @property
    def mode(self):
        """
        The current operation mode of this feature.
        
        Feature modes are the way the feature is controlled. Three modes
        exist: ``"manual"``, ``"auto"`` and ``"one_push"``. The latter
        performs an automatic setting before self-clearing.

        Use :attr:`modes` to obtain a list of allowed values.
        """
        mode = feature_mode_t()
        dll.dc1394_feature_get_mode(
                self._cam, self._feature_id, byref(mode))
        return feature_mode_vals[mode.value]

    @mode.setter
    def mode(self, mode):
        key = feature_mode_codes[mode]
        dll.dc1394_feature_set_mode(
                self._cam, self._feature_id, key)

    @property
    def readable(self):
        """
        Can the current value of this feature be read via :attr:`value`
        or :attr:`absolute`? Read-only.
        """
        k = bool_t()
        dll.dc1394_feature_is_readable(
                self._cam, self._feature_id, byref(k))
        return k.value

    @property
    def value(self):
        """
        The current value of this feature in arbitrary integer units.
        """
        val = c_uint32()
        dll.dc1394_feature_get_value(
                self._cam, self._feature_id, byref(val))
        return val.value

    @value.setter
    def value(self, value):
        val = int(value)
        dll.dc1394_feature_set_value(
                self._cam, self._feature_id, val)

    @property
    def value_range(self):
        """
        Minimum and maximum possible values for this feature. Read-only.
        """
        min_val, max_val = c_uint32(), c_uint32()
        dll.dc1394_feature_get_boundaries(
                self._cam, self._feature_id,
                byref(min_val), byref(max_val))
        return min_val.value, max_val.value

    @property
    def absolute_capable(self):
        """
        Can this feature be controlled in absolute units? Read-only.
        """
        k = bool_t()
        dll.dc1394_feature_has_absolute_control(
                self._cam, self._feature_id, byref(k))
        return k.value

    @property
    def absolute(self):
        """
        The current value of the feature in absolute (physical) units.

        Refer to the documentation of your camera and the respective
        feature what these absolute units are.
        """
        val = c_float()
        dll.dc1394_feature_get_absolute_value(
                self._cam, self._feature_id, byref(val))
        return val.value

    @absolute.setter
    def absolute(self, value):
        val = float(value)
        dll.dc1394_feature_set_absolute_value(
                self._cam, self._feature_id, val)

    @property
    def absolute_control(self):
        """
        Is the value of the feature controlled by the :attr:`value`
        integer or by physical units via :attr:`absolute`?
        """
        k = bool_t()
        dll.dc1394_feature_get_absolute_control(
                self._cam, self._feature_id, byref(k))
        return k.value

    @absolute_control.setter
    def absolute_control(self, value):
        val = int(value)
        dll.dc1394_feature_set_absolute_control(
                self._cam, self._feature_id, val)

    @property
    def absolute_range(self):
        """
        Minumum and maximum possible value for this feature in absolute
        (physical) units. Read-only.
        """
        min_val, max_val = c_float(), c_float()
        dll.dc1394_feature_get_absolute_boundaries(
                self._cam, self._feature_id,
                byref(min_val), byref(max_val))
        return min_val.value, max_val.value

    def setup(self, value=None, active=True, mode="manual", absolute=True,
            **kwargs):
        """
        Set up several properties of this feature with one call.

        The function defaults to activating the feature, subjecting it
        to "manual" and "absolute" control. You can pass a value if
        desired. Any additional keyword arguments will be set as
        attributes.
        """
        self.active = active
        if not active:
            return
        if mode is not None:
            self.mode = mode
            if mode == ("auto", "one_push"):
                return
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)
        if absolute is not None and self.absolute_capable:
            self.absolute_control = absolute
        if value is None:
            return
        if absolute:
            self.absolute = value
        else:
            self.value = value


class Trigger(Feature):
    @property
    def active(self):
        """
        Switches between internal and external trigger.
        """
        k = bool_t()
        dll.dc1394_external_trigger_get_power(
                self._cam, byref(k))
        return bool(k.value)

    @active.setter
    def active(self, value):
        k = bool(value)
        dll.dc1394_external_trigger_set_power(
                self._cam, k)

    @property
    def modes(self):
        """
        Possible modes to trigger the camera. Read-only.

        Trigger mode mostly refer to external trigger. Each trigger has
        a meaning specified in the IIDC specifications.

        * Mode 0: Exposure starts with a falling edge and stops when the
          the exposure specified by the ``shutter`` feature is elapsed.
        * Mode 1: Exposure starts with a falling edge and stops with the
          next rising edge.
        * Mode 2: The camera starts the exposure at the first falling
          edge and stops the integration at the ``n`` th falling edge. The
          parameter ``n`` is a prameter of the trigger that can be set with
          :attr:`value`.
        * Mode 3: This is an internal trigger mode. The trigger is
          generated every ``n`` periods of the fastest framerate.
          Once again, the parameter ``n`` can be set with attr:`value`.
        * Mode 4: A multiple exposure mode. ``n`` exposures are performed
          each time a falling edge is observed on the trigger signal. Each
          exposure is as long as defined by the ``shutter`` feature.
        * Mode 5: Another multiple exposure mode. Same as Mode 4 except
          that the exposure is is defined by the length of the trigger
          pulse instead of the ``shutter`` feature.
        * Mode 14 and 15: both are vendor specified trigger modes. 
        """
        finfo = feature_info_t()
        finfo.id = self._feature_id
        dll.dc1394_feature_get(
                self._cam, byref(finfo))
        modes = finfo.trigger_modes
        return [trigger_mode_vals_short[i]
                for i in modes.modes[:modes.num]]

    @property
    def mode(self):
        """
        The currently active trigger mode.

        You camera may not support all trigger modes.
        See :attr:`modes` for a documentation of the different modes.
        """
        mode = trigger_mode_t()
        dll.dc1394_external_trigger_get_mode(
                self._cam, byref(mode))
        return trigger_mode_vals_short[mode.value]

    @mode.setter
    def mode(self, value):
        key = trigger_mode_codes_short[value]
        dll.dc1394_external_trigger_set_mode(
                self._cam, key)

    @property
    def polarity_capable(self):
        """
        Can the polarity of the trigger input be set/inverted?
        Read-only.
        """
        finfo = feature_info_t()
        finfo.id = self._feature_id
        dll.dc1394_feature_get(self._cam, byref(finfo))
        return bool(finfo.polarity_capable)
    
    @property
    def polarity(self):
        """
        The current active polarity for the trigger input.

        Either ``"ACTIVE_LOW"`` or ``"ACTIVE_HIGH"``.
        """
        pol = trigger_polarity_t()
        dll.dc1394_external_trigger_get_polarity(
                self._cam, byref(pol))
        return trigger_polarity_vals_short[pol.value]

    @polarity.setter
    def polarity(self, pol):
        key = trigger_polarity_codes_short[pol]
        dll.dc1394_external_trigger_set_polarity(
                self._cam, key)
    
    @property
    def source(self):
        """
        The physical source for the trigger signal.

        Some cameras let you select the external trigger input.
        """
        source = trigger_source_t()
        dll.dc1394_external_trigger_get_source(
                self._cam, byref(source))
        return trigger_source_vals_short[source.value]

    @source.setter
    def source(self, source):
        key = trigger_source_codes_short[source]
        dll.dc1394_external_trigger_set_source(
                self._cam, key)

    @property
    def sources(self):
        """
        Allowed sources for the trigger condition. Read-only.
        
        Use :attr:`source` to get or set the current source.
        """
        src = trigger_sources_t()
        dll.dc1394_external_trigger_get_supported_sources(
                self._cam, byref(src))
        return [trigger_source_vals_short[i]
                for i in src.sources[:src.num]]

    @property
    def software(self):
        """
        Is the software trigger condition active?
        """
        res = switch_t()
        dll.dc1394_software_trigger_get_power(
                self._cam, byref(res))
        return bool(res.value)

    @software.setter
    def software(self, value):
        k = bool(value)
        dll.dc1394_software_trigger_set_power(
                self._cam, k)


class Whitebalance(Feature):
    @property
    def value(self):
        """
        The current whitebalance is a tuple of two values:
        the blue or U (for YUV) channel and
        the red or V (for YUV) channel.
        """
        blue, red = c_uint32(), c_uint32()
        dll.dc1394_feature_whitebalance_get_value(
            self._cam, byref(blue), byref(red))
        return blue.value, red.value
            
    @value.setter
    def value(self, value):
        blue, red = value
        dll.dc1394_feature_whitebalance_set_value(
                self._cam, blue, red)


class Temperature(Feature):
    @property
    def value(self):
        """
        The current temperature is a tuple of the current target
        temperature (setpoint) and the actual temperature. Setting the
        value only requires one parameter: the new setpoint temperature
        All temperatures are given in deci-degrees kelvin.
        """
        setpoint, current = c_uint32(), c_uint32()
        dll.dc1394_feature_temperature_get_value(
            self._cam, byref(setpoint), byref(current))
        return setpoint.value, current.value
            
    @value.setter
    def value(self, value):
        setpoint = int(value)
        dll.dc1394_feature_temperature_set_value(
                self._cam, setpoint)


class Whiteshading(Feature):
    @property
    def value(self):
        """
        The current whiteshading value: a tuple of (red, green, blue)
        """
        red, green, blue = c_uint32(), c_uint32(), c_uint32()
        dll.dc1394_feature_whiteshading_get_value(
            self._cam, byref(red), byref(green), byref(blue))
        return red.value, green.value, blue.value
            
    @value.setter
    def value(self, value):
        red, green, blue = value
        dll.dc1394_feature_temperature_set_value(
                self._cam, int(red), int(green), int(blue))


_feature_map = dict((n, Feature) for n in feature_codes)
_feature_map["trigger"] = Trigger
_feature_map["white_shading"] = Whiteshading
_feature_map["white_balance"] = Whitebalance
_feature_map["temperature"] = Temperature


class Mode(object):
    """
    A video mode for a DC1394 camera.
    
    Do not instantiate this class directly. Instead use one of the modes
    in :attr:`Camera.modes` or :attr:`Camera.modes_dict` and assign it to
    :attr:`Camera.mode`.
    """

    def __init__(self, cam, mode_id):
        self._mode_id = mode_id
        self._cam = cam

    @property
    def mode_id(self):
        return self._mode_id

    @property
    def name(self):
        """
        A descriptive name for this mode. Like ``"640x480_Y8"`` or
        ``"FORMAT7_2"``. Read-only.
        """
        return video_mode_vals[self._mode_id]

    def __str__(self):
        return self.name

    @property
    def rates(self):
        """
        Allowed framerates if the camera is in this mode. Read-only.
        """
        fpss = framerates_t()
        dll.dc1394_video_get_supported_framerates(
                self._cam, self._mode_id, byref(fpss))
        return [framerate_vals[i]
                for i in fpss.framerates[:fpss.num]]

    @property
    def image_size(self):
        """
        The size in pixels of frames acquired in this mode. Read-only.
        """
        w = c_uint32()
        h = c_uint32()
        dll.dc1394_get_image_size_from_video_mode(
                self._cam, self._mode_id, byref(w), byref(h))
        return w.value, h.value

    @property
    def color_coding(self):
        """
        The type of color coding of pixels. Read-only.
        """
        cc = color_coding_t()
        dll.dc1394_get_color_coding_from_video_mode(
                self._cam, self._mode_id, byref(cc))
        return color_coding_vals[cc.value]

    @property
    def scalable(self):
        """
        Is this video mode scalable? Read-only.
        """
        return bool(dll.dc1394_is_video_mode_scalable(self._mode_id))

    @property
    def dtype(self):
        """
        A suitable numpy dtype string for the image array data.
        Read-only.
        """
        if self.color_coding.endswith("16"):
            return ">u2"
        else:
            return ">u1"

class Exif(Mode):
    pass


class Format7(Mode):
    """
    Format7 modes are flexible modes that support:
    
    * acquiring and transferring only a subsection of the frame for
      faster acquisition: regio-of-interes (ROI)
    * binning the pixels of the sensor for faster acquisition and
      reduced readout noise. The binning strategy in the different
      Format7 modes is defined by the vendor.

    Many aspects of Format7 modes can be altered while an acquisition is
    in progress. A notable exception from this is the size of the
    packet.

    Use :attr:`max_image_size`, :attr:`unit_size`, :attr:`unit_position`,
    :attr:`color_codings`, and :attr:`data_depth` to obtain information
    about the mode and then set its parameters via the attributes 
    :attr:`image_size`, :attr:`image_position`, :attr:`color_coding`, and
    :attr:`packet_size` or all of them via the :attr:`roi` attribute
    or with a call to :meth:`setup`.

    All settings are sent to the hardware right away.
    """

    @property
    def frame_interval(self):
        """
        The current frame interval in this format7 mode in seconds.
        Read-only.
        
        Use the :attr:`Camera.framerate` and :attr:`Camera.shutter`
        features (if present) to influence the framerate.
        """
        fi = c_float()
        dll.dc1394_format7_get_frame_interval(self._cam,
                    self._mode_id, byref(fi))
        return fi.value

    @property
    def max_image_size(self):
        """
        The maximum size (horizontal and vertical) of the ROI in pixels.
        Read-only.
        """
        hsize = c_uint32()
        vsize = c_uint32()
        dll.dc1394_format7_get_max_image_size(
                self._cam, self._mode_id,
                byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @property
    def image_size(self):
        """
        The current size (horizontal and vertical) of the ROI in pixels.

        The image size can only be a multiple of the :attr:`unit_size`, and
        cannot be smaller than it.
        """
        hsize = c_uint32()
        vsize = c_uint32()
        dll.dc1394_format7_get_image_size(
                self._cam, self._mode_id,
                byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @image_size.setter
    def image_size(self, value):
        width, height = value
        dll.dc1394_format7_set_image_size(
                self._cam, self._mode_id,
                width, height)

    @property
    def image_position(self):
        """
        The start position of the upper left corner of the ROI in
        pixels (horizontal and vertical).

        The image position can only be a multiple of the unit position
        (zero is acceptable).
        """
        x = c_uint32()
        y = c_uint32()
        dll.dc1394_format7_get_image_position(
                self._cam, self._mode_id,
                byref(x), byref(y))
        return x.value, y.value

    @image_position.setter
    def image_position(self, value):
        x, y = value
        dll.dc1394_format7_set_image_position(
                self._cam, self._mode_id,
                x, y)

    @property
    def color_codings(self):
        """
        Allowed color codings in this mode. Read-only.
        """
        pos_codings = color_codings_t()
        dll.dc1394_format7_get_color_codings(
                self._cam, self._mode_id,
                byref(pos_codings))
        return [color_coding_vals[i]
                for i in pos_codings.codings[:pos_codings.num]]

    @property
    def color_coding(self):
        """
        The current color coding.
        """
        cc = color_coding_t()
        dll.dc1394_format7_get_color_coding(
                self._cam, self._mode_id, byref(cc))
        return color_coding_vals[cc.value]

    @color_coding.setter
    def color_coding(self, color):
        code = color_coding_codes[color]
        dll.dc1394_format7_set_color_coding(
                self._cam, self._mode_id, code)

    @property
    def unit_position(self):
        """
        Horizontal and vertical :attr:`image_position` multiples.
        Read-only.
        """
        h_unit = c_uint32()
        v_unit = c_uint32()
        dll.dc1394_format7_get_unit_position(
                self._cam, self._mode_id,
                byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def unit_size(self):
        """
        Horizontal and vertical :attr:`image_size` multiples. Read-only.
        """
        h_unit = c_uint32()
        v_unit = c_uint32()
        dll.dc1394_format7_get_unit_size(
                self._cam, self._mode_id,
                byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def roi(self):
        """
        Get and set all Format7 parameters at once.

        The following definitions can be used to set ROI of Format7 in
        a simpler fashion:
        
        * QUERY_FROM_CAMERA (-1) will use the current value used by the
          camera,
        * USE_MAX_AVAIL will (-2) set the value to its maximum and
        * USE_RECOMMENDED (-3) can be used for the bytes-per-packet
          setting.
        """
        w, h, x, y = c_int32(), c_int32(), c_int32(), c_int32()
        cco, packet_size = color_coding_t(), c_int32()
        dll.dc1394_format7_get_roi(
            self._cam, self._mode_id, byref(cco), byref(packet_size),
            byref(x), byref(y), byref(w), byref(h))
        return ((w.value, h.value), (x.value, y.value),
            color_coding_vals[cco.value], packet_size.value)

    @roi.setter
    def roi(self, args):
        size, position, color, packet_size = args
        dll.dc1394_format7_set_roi(
            self._cam, self._mode_id, color_coding_codes[color],
            packet_size, position[0], position[1], size[0], size[1])

    @property
    def recommended_packet_size(self):
        """
        Recommended number of bytes per packet. Read-only.
        """
        packet_size = c_uint32()
        dll.dc1394_format7_get_recommended_packet_size(
            self._cam, self._mode_id, byref(packet_size))
        return packet_size.value

    @property
    def packet_parameters(self):
        """
        Maximum number and unit size of bytes per packet. Read-only.

        Get the parameters of the packet size: its maximal size and its
        unit size. The packet size is always a multiple of the unit
        bytes and cannot be zero.
        """
        packet_size_max = c_uint32()
        packet_size_unit = c_uint32()
        dll.dc1394_format7_get_packet_parameters(
            self._cam, self._mode_id, byref(packet_size_unit),
            byref(packet_size_max))
        return packet_size_unit.value, packet_size_max.value

    @property
    def packet_size(self):
        """
        Current number of bytes per packet.
        """
        packet_size = c_uint32()
        dll.dc1394_format7_get_packet_size(
            self._cam, self._mode_id, byref(packet_size))
        return packet_size.value

    @packet_size.setter
    def packet_size(self, packet_size):
        dll.dc1394_format7_set_packet_size(
            self._cam, self._mode_id, int(packet_size))

    @property
    def total_bytes(self):
        """
        Current total number of bytes per frame. Read-only.

        This includes padding (to reach an entire number of packets).
        Use :attr:`packet_size` to influence its value.
        """
        ppf = c_uint32()
        dll.dc1394_format7_get_total_bytes(
            self._cam, self._mode_id, byref(ppf))
        return ppf.value

    @property
    def data_depth(self):
        """
        The number of bits per pixel. Read-only.
        Need not be a multiple of 8.
        """
        dd = c_uint32()
        dll.dc1394_format7_get_data_depth(
            self._cam, self._mode_id, byref(dd))
        return dd.value

    @property
    def pixel_number(self):
        """
        The number of pixels per frame. Read-only.
        """
        px = c_uint32()
        dll.dc1394_format7_get_pixel_number(
            self._cam, self._mode_id, byref(px))
        return px.value

    def setup(self, image_size=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
            image_position=(QUERY_FROM_CAMERA, QUERY_FROM_CAMERA),
            color_coding=QUERY_FROM_CAMERA, packet_size=USE_RECOMMENDED):
        """
        Setup this Format7 mode.
        
        Similar to setting :attr:`roi` but size and position are made
        multiples of :attr:`unit_size` and :attr:`unit_position`. All
        arguments are optional and default to not changing the current
        value. :attr:`packet_size` is set to the recommended value.
        """
        wu, hu = self.unit_size
        xu, yu = self.unit_position
        position = xu*int(image_position[0]/xu), yu*int(image_position[1]/yu)
        size = wu*int(image_size[0]/wu), hu*int(image_size[1]/hu)
        self.roi = size, position, color_coding, packet_size
        #return size, position, color_coding, packet_size
        return self.roi


_mode_map = {
       64: Mode,
       65: Mode,
       66: Mode,
       67: Mode,
       68: Mode,
       69: Mode,
       70: Mode,
       71: Mode,
       72: Mode,
       73: Mode,
       74: Mode,
       75: Mode,
       76: Mode,
       77: Mode,
       78: Mode,
       79: Mode,
       80: Mode,
       81: Mode,
       82: Mode,
       83: Mode,
       84: Mode,
       85: Mode,
       86: Mode,
       87: Exif,
       88: Format7,
       89: Format7,
       90: Format7,
       91: Format7,
       92: Format7,
       93: Format7,
       94: Format7,
       95: Format7,
}


class Camera(object):
    """
    This class represents a DC1394 Camera on the bus.
    """

    _cam = None
    _context = None

    def __init__(self, guid=None, context=None, handle=None,
            iso_speed=None, mode=None, rate=None, **features):
        """
        Obtain a camera object either supplying:

        * nothing: the first available camera on the system will be
          chosen.
        * a GUID: the camera with this specific GUID (an integer,
          use ``int(hex_guid_as_string, 16)`` to convert)
        * a handle obtained from :meth:`Context.camera_handle`
        
        The :class:`Context` can be supplied. It will be used to obtain
        the camera handle. If no context is supplied, a new one will be
        created and maintained.

        The camera's settings and modes are left unchanged unless 
        the video :attr:`mode`, the :attr:`iso_speed`, and
        the frame :attr:`rate` are give. Additionally, arbitrary
        :attr:`features` of the camera can be set. The supplied features
        are set in undefined order.
        """
        
        if handle is None:
            if context is None:
                context = Context()
            if isinstance(guid, str):
                guid, unit = int(guid, 16), None
            elif guid is None:
                guid, unit = context.cameras[0]
            else:
                unit = None
            handle = context.camera_handle(guid, unit)
        else:
            assert context is not None
        
        # _we_ need to ensure the dc1394 context is alive
        self._context = context
        self._cam = handle

        # setup static attributes of the camera
        self._features = self._load_features()
        self._modes, self._modes_dict = self._load_modes()

        if iso_speed is not None:
            self.iso_speed = iso_speed
        if mode is not None:
            self.mode = self._modes_dict[mode]
        if rate is not None:
            self.rate = rate
        self.setup(**features)

    def __del__(self):
        self.close()

    def close(self):
        """
        Frees a camera structure.
        """
        if self._cam:
            dll.dc1394_camera_free(self._cam)
        self._cam = None
        # do not invalidate the context here as someone else could be
        # using it.
        # if self._context:
        #     self._context.close()
        # only un-reference it so it can be freed if noone needs it anymore.
        self._context = None
  
    def power(self, on=True):
        """
        Sets the camera power.
        
        This is very close to (un)plugging the camera power but note
        that there is a difference as some circuits in the camera must
        be continuously powered in order to respond to a power-up
        command. Unpowering the camera using this attribute does not
        cause a re-enumeration and does not invalidate the object or
        change settings. It can be used to enable power-saving and to
        prevent the camera from heating up to much thereby reducing the 
        dark current and read-out noise.
        """
        dll.dc1394_camera_set_power(self._cam, on)

    def reset_bus(self):
        """
        Resets the IEEE1394 bus which camera is attached to.
        
        Calling this function is "rude" to other devices because it
        causes them to re-enumerate on the bus and may cause a temporary
        disruption in their current activities.  Thus, use it sparingly.
        Its primary use is if a program shuts down uncleanly and needs
        to free leftover ISO channels or bandwidth.  A bus reset will
        free those things as a side effect.
        
        Call :meth:`close` as the camera handle is invalid afterwards.
        """
        dll.dc1394_reset_bus(self._cam)

    def reset_camera(self):
        """
        Resets the camera causing it to forget some settings and to
        re-enumerate (?).
        
        Call :meth:`close` after using this method as the camera handle
        becomes invalid.
        """
        dll.dc1394_camera_reset(self._cam)

    def memory_save(self, channel):
        """
        Saves the camera settings in the camera memory bank specified by
        the channel argument.
        
        The number of available channels is
        available from in :attr:`memory_channels`. You should wait until the
        save operation if finished before changing camera registers. You
        cannot write in channel zero as it is read-only and contains
        factory defaults.

        .. note::
           This operation can only be performed a certain number
           of times for a given camera, as it requires reprogramming of an
           EEPROM.
        """
        dll.dc1394_memory_save(self._cam, int(channel))

    def memory_load(self, channel):
        """
        Loads the settings stored in the specified channel.
        
        Channel zero is the factory defaults.
        """
        dll.dc1394_memory_load(self._cam, int(channel))

    @property
    def memory_busy(self):
        """
        Checks for pending memory operations. Read-only.

        You need to allow the camera some time to finish the saving
        operation. This function can be used in a loop to check when the
        operation finished. This could be integrated in the save
        function in the future.
        """
        v = bool_t()
        dll.dc1394_memory_busy(self._cam, byref(v))
        return bool(v.value)

    def flush(self):
        """
        Flush already acquired and transferred frames from the DMA
        buffer.
        
        These old frames would otherwise be returned by :meth:`dequeue`.
        """
        frame = POINTER(video_frame_t)()
        while True:
            dll.dc1394_capture_dequeue(self._cam,
                    CAPTURE_POLICY_POLL, byref(frame))
            if not bool(frame):
                break
            dll.dc1394_capture_enqueue(self._cam,
                    frame)

    def dequeue(self, poll=False):
        """
        Capture a frame.

        When capturing a frame you can choose to either wait for the
        frame indefinitely (``poll=False``, the default) or return
        ``None`` immediately if no frame arrived yet (``poll=True``).

        Release the returned frame as soon as possible via
        :meth:`pydc1394.frame.Frame.enqueue` to return it to the DMA buffer
        and recycle it.
        """
        frame = POINTER(video_frame_t)()
        policy = poll and CAPTURE_POLICY_POLL or CAPTURE_POLICY_WAIT
        dll.dc1394_capture_dequeue(self._cam,
                policy, byref(frame))
        if not bool(frame):
            return
        return Frame(self._cam, frame)

    def start_capture(self, bufsize=4, capture_flags="DEFAULT"):
        """
        Setup the capture session.

        ``bufsize`` is the number of images in the ring buffer. Thanks to
        some hack you can even set this parameter to 1 but the
        recommended value is between four to ten. If you request too much
        memory (above 30M) there is a chance that the function will fail.

        Use ``capture_flags`` to setup bandwidth and channel allocation
        and to enable automatic start of iso transmission.
        """
        dll.dc1394_capture_setup(
                self._cam, bufsize,
                capture_flag_codes_short[capture_flags])

    def stop_capture(self):
        """
        End the capture session.
        """
        dll.dc1394_capture_stop(self._cam)

    def start_video(self):
        """
        Instruct the camera to start capturing and transferring frames.
        """
        dll.dc1394_video_set_transmission(self._cam, 1)

    def stop_video(self):
        """
        Instruct the camera to stop capturing and transmitting frames.
        """
        dll.dc1394_video_set_transmission(self._cam, 0)

    def start_one_shot(self):
        """
        Instruct the camera to acquire and transmit exactly one frame.
        """
        dll.dc1394_video_set_one_shot(self._cam, 1)

    def stop_one_shot(self):
        """
        Stop single shot tramsission mode.
        """
        dll.dc1394_video_set_one_shot(self._cam, 0)

    def start_multi_shot(self, n):
        """
        Instruct the camera to acquire and transfer ``n`` frames.
        """
        dll.dc1394_video_set_multi_shot(self._cam, n, 1)

    def stop_multi_shot(self):
        """
        Stop multi shot acquisition.
        """
        dll.dc1394_video_set_multi_shot(self._cam, 0, 0)

    @property
    def fileno(self):
        """
        A file descriptor suitable for passing to :func:`select.select`.
        Read-only.

        The file descriptor can be used to determine whether and when 
        new frames are available for reading as part of an application's
        event loop.

        An alternative to blocking access with ``select()``
        is to use the polling mode of :meth:`dequeue`.
        """
        return dll.dc1394_capture_get_fileno(self._cam)

    def _load_features(self):
        """
        Return feature objects for all available features.
        """
        fs = featureset_t()
        dll.dc1394_feature_get_all(self._cam, byref(fs))
        features = {}
        for i in range(FEATURE_NUM):
            s = fs.feature[i]
            if s.available:
                name = feature_vals[s.id]
                feature = _feature_map[name](self._cam, s.id)
                features[name] = feature
                setattr(self, name, feature)
        return features

    @property
    def features(self):
        """
        A list of all available features of the camera. Read-only.
        """
        return self._features

    def setup(self, active=True, mode="manual", absolute=True,
            **features):
        """
        Setup several features of the camera in one call.
        
        Pass all features and the values to set as additional keyword
        arguments. By default the specified features are activated, set
        to ``manual`` and ``absolute`` mode.
        """
        for k, v in features.items():
            self.features[k].setup(v, active, mode, absolute)

    def _load_modes(self):
        """
        Obtain and return a list and a dictionary of all supported modes
        of the camera.
        """
        modes = video_modes_t()
        dll.dc1394_video_get_supported_modes(self._cam, byref(modes))
        modes = [_mode_map[i](self._cam, i)
                for i in modes.modes[:modes.num]]
        modes_dict = dict((m.name, m) for m in modes)
        return modes, modes_dict

    @property
    def modes(self):
        """
        A list of modes supported by this camera. Read-only.
        """
        return self._modes

    @property
    def modes_dict(self):
        """
        A dictionary of modes supported by this camera. Read-only.
        """
        return self._modes_dict

    def get_register(self, offset):
        """
        Returns the current value of the register at address ``offset``.
        """
        val = c_uint32()
        dll.dc1394_get_control_registers(
                self._cam, offset, byref(val), 1)
        return val.value

    def set_register(self, offset, value):
        """
        Set the register at ``offset`` to ``value``.
        """
        val = c_uint32(value)
        dll.dc1394_set_control_registers(
                self._cam, offset, byref(val), 1)

    # shortcuts for getting and setting registers.
    # these make toggling bits simpler (cam[0x100] |= 1<<6 versus
    # cam.set_register(0x100, cam.get_register(0x100) | (1<<6)))
    __getitem__ = get_register
    __setitem__ = set_register

    @property
    def broadcast(self):
        """
        This sets if the camera tries to synchronize with other cameras on
        the bus.

        If the broadcast flag is set, all devices on the bus will
        execute the command. Useful to sync ISO start commands or
        setting a bunch of cameras at the same time. Broadcast only
        works with identical devices (brand/model). If the devices are
        not identical your mileage may vary. Some cameras may not answer
        broadcast commands at all. Also, this only works with cameras on
        the SAME bus (IOW, the same port).

        .. note::
           The behaviour might be strange if one camera tries to
           broadcast and another not.

        .. note::
           This feature is currently only supported under linux
           and has not been seen working yet. So use on your own risk.
        """
        k = bool_t()
        dll.dc1394_camera_get_broadcast(self._cam, byref(k))
        return bool(k.value)

    @broadcast.setter
    def broadcast(self, value):
        dll.dc1394_camera_set_broadcast(self._cam, value)

    @property
    def model(self):
        """
        The model name of the camera. Read-only.
        """
        return self._cam.contents.model

    @property
    def guid(self):
        """
        The (integer) GUID of the camera. Read-only.
        
        Use ``hex(cam.guid)`` or ``"%x" % cam.guid`` to get a hexadecimal
        string.
        """
        return self._cam.contents.guid

    @property
    def vendor(self):
        """
        The vendor name of the camera. Read-only.
        """
        return self._cam.contents.vendor

    def __str__(self):
        return "<Camera %x (%s/%s)>" % (self.guid,
                self.vendor, self.model)

    @property
    def mode(self):
        """
        The current video mode of the camera.

        The video modes are what let you choose the image size and color
        format. Two special format classes exist: the :class:`Exif`
        mode (which is actually not supported by any known camera)
        and :class:`Format7` which is the scalable image format.
        Format7 allows you to change the image size, framerate, color
        coding and crop region.

        Important note: your camera will not support all the video modes
        but will only supports a more or less limited subset of them.

        Use :attr:`modes` to obtain a list of valid modes for this camera.
        """
        vmod = video_mode_t()
        dll.dc1394_video_get_mode(self._cam, byref(vmod))
        return self._modes_dict[video_mode_vals[vmod.value]]

    @mode.setter
    def mode(self, mode):
        dll.dc1394_video_set_mode(self._cam, mode.mode_id)

    @property
    def rate(self):
        """
        The framerate belonging to the current camera mode.

        For non-scalable video formats (not :class:`Format7`) there is a
        set of standard frame rates one can choose from. A list
        of all the framerates supported by your camera for a specific
        video mode can be obtained from :attr:`Mode.rates`.

        .. note::
           You may also be able to set the framerate with the
           :attr:`framerate` feature if present.
 
        .. note::
           Framerates are used with fixed-size image formats (Format_0
           to Format_2).  In :class:`Format7` modes the camera can tell
           an actual value, but one can not set it.  Unfortunately the
           returned framerate may have no sense at all.  If you use
           Format_7 you should set the framerate by adjusting the number
           of bytes per packet (:attr:`Format7.packet_size`) and/or the
           shutter time.
        """
        ft = framerate_t()
        dll.dc1394_video_get_framerate(self._cam, byref(ft))
        return framerate_vals[ft.value]

    @rate.setter
    def rate(self, framerate):
        wanted_frate = framerate_codes[framerate]
        dll.dc1394_video_set_framerate(self._cam, wanted_frate)
    
    @property
    def iso_speed(self):
        """
        The isochronous speed at which the transmission should occur.

        Most (if not all) cameras are compatible with 400Mbps speed.
        Only older cameras (pre-1999) may still only work at sub-400
        speeds. However, speeds lower than 400Mbps are still useful:
        they can be used for longer distances (e.g. 10m cables).  Speeds
        over 400Mbps are only available in "B" mode (see
        :attr:`operation_mode`).
        """
        sp = speed_t()
        dll.dc1394_video_get_iso_speed(self._cam, byref(sp))
        return speed_vals[sp.value]

    @iso_speed.setter
    def iso_speed(self, iso_speed):
        sp = speed_codes[iso_speed]
        self.operation_mode = 'LEGACY' if iso_speed < 800 else '1394B'
        dll.dc1394_video_set_iso_speed(self._cam, sp)

    @property
    def operation_mode(self):
        """
        IEEE1394 legacy or IEEE1394b operation mode.

        As the IEEE1394 speeds were increased with IEEE1394b
        specifications, a new type of control is necessary when the
        camera is operating in iso speeds over 800Mbps. If you wish to
        use a 1394b camera you may need to switch the operation mode to
        1394b. Legacy mode refers to speeds less than 400Mbps.
        """
        k = operation_mode_t()
        dll.dc1394_video_get_operation_mode(self._cam, byref(k))
        return operation_mode_vals_short[k.value]

    @operation_mode.setter
    def operation_mode(self, value):
        k = operation_mode_codes_short[value]
        dll.dc1394_video_set_operation_mode(self._cam, k)

    @property
    def iso_channel(self):
        """
        The current ISO channel.
        """
        channel = c_uint32()
        dll.dc1394_video_get_iso_channel(self._cam, byref(channel))
        return channel.value

    @iso_channel.setter
    def iso_channel(self, channel):
        dll.dc1394_video_set_iso_channel(self._cam, channel)

    @property
    def data_depth(self):
        """
        Gets the current data depth, in bits. Read-only.
        
        Only meaningful for 16bpp video modes (RAW16, RGB48,
        MONO16,...).
        """
        data_depth = c_uint32()
        dll.dc1394_video_get_data_depth(self._cam, byref(data_depth))
        return data_depth.value

    @property
    def bandwidth_usage(self):
        """
        Gets the bandwidth usage of a camera. Read-only.

        This function returns the bandwidth that is used by the camera
        *if* ISO was ON.  The returned value is in bandwidth units. The
        1394 bus has 4915 bandwidth units available per cycle. Each unit
        corresponds to the time it takes to send one quadlet at ISO
        speed S1600. The bandwidth usage at S400 is thus four times the
        number of quadlets per packet. Thanks to Krisitian Hogsberg for
        clarifying this.
        """
        bandwidth = c_uint32()
        dll.dc1394_video_get_bandwidth_usage(self._cam, byref(bandwidth))
        return bandwidth.value

    def get_strobe(self, offset):
        """
        The value of the strobe configuration register at ``offset``.
        """
        k = c_uint32()
        dll.dc1394_get_strobe_register(self._cam, offset, byref(k))
        return k.value

    def set_strobe(self, offset, value):
        """
        Set the strobe configuration register at ``offset`` to ``value``.
        """
        dll.dc1394_set_strobe_register(self._cam, offset, value)

    def is_same_camera(self, other):
        """
        Tells whether two camera objects refer to the same physical
        camera unit.
        """
        return bool(dll.dc1394_is_same_camera(self._cam, other._cam))

    __eq__ = is_same_camera

    @property
    def node(self):
        """
        Gets the IEEE 1394 node ID of the camera.
        """
        node, generation = c_uint32(), c_uint32()
        dll.dc1394_camera_get_node(self._cam, byref(node),
                byref(generation))
        return node.value, generation.value
