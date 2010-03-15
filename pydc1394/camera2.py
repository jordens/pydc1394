#!/usr/bin/python
# encoding: utf-8
# Copyright 2010 Robert Jordens <jordens@phys.ethz.ch>
#
# This file is part of pydc1394.
#
# pydc1394 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# pydc1394 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pydc1394.  If not, see <http://www.gnu.org/licenses/>.

from _dc1394core import *
from _dc1394core import _dll
from ctypes import c_byte, c_int, c_uint32, c_int32, c_float

from numpy import frombuffer, ndarray

__all__ = ["DC1394Library", "Camera"]


class DC1394Library(object):
    _h = None

    def __init__(self):
        self._h = _dll.dc1394_new()

    def __del__(self):
        self.close()

    def close(self):
        if self._h is not None:
            _dll.dc1394_free(self._h)
        self._h = None

    @property
    def cameras(self):
        l = POINTER(camera_list_t)()
        _dll.dc1394_camera_enumerate(self._h, byref(l))
        cams = [(id.guid, id.unit) for id in
                l.contents.ids[:l.contents.num]]
        _dll.dc1394_camera_free_list(l)
        return cams

    def camera_handle(self, guid, unit=None):
        if unit is None:
            handle = _dll.dc1394_camera_new(
                    self._h, guid)
        else:
            handle = _dll.dc1394_camera_new_unit(
                    self._h, guid, unit)
        if not handle:
            raise DC1394Exception, "Couldn't access camera (%s,%s)!" % (
                    guid, unit)
        return handle
 

class Image(ndarray):
    @classmethod
    def from_frame(cls, frame):
        dtyp = ARRAY(c_byte, frame.contents.image_bytes)
        buf = dtyp.from_address(frame.contents.image)
        pixs = frame.contents.size[0]*frame.contents.size[1]
        end = frame.contents.little_endian and "<" or ">"
        dt = "%su%i" % (end, frame.contents.image_bytes/pixs)
        img = frombuffer(buf, dtype=dt)
        img = img.reshape(frame.contents.size).copy().view(cls)
        img._id = frame.contents.id
        img._frames_behind = frame.contents.frames_behind
        img._position = frame.contents.position
        img._packet_size = frame.contents.packet_size
        img._packets_per_frame = frame.contents.packets_per_frame
        img._timestamp = frame.contents.timestamp
        img._video_mode = video_mode_vals[frame.contents.video_mode]
        img._data_depth = frame.contents.data_depth
        return img

    @property
    def position(self):
        "ROI position (offset)"
        return self._position

    @property
    def packet_size(self):
        "The size of a datapacket in bytes."
        return self._packet_size

    @property
    def packets_per_frame(self):
        "Number of packets per frame."
        return self._packets_per_frame

    @property
    def timestamp(self):
        "The IEEE Bustime when the picture was acquired (microseconds)"
        return self._timestamp

    @property
    def frames_behind(self):
        "the number of frames left in the ring buffer"
        return self._frames_behind

    @property
    def id(self):
        "the frame position in the ring buffer"
        return self._id

    @property
    def corrupt(self):
        "corrupt image marker (libdc1394)"
        return self._corrupt
    
    @property
    def data_depth(self):
        "number of data bits"
        return self._data_depth

    @property
    def video_mode(self):
        "the valid video mode"
        return self._video_mode


class Feature(object):
    def __init__(self, cam, id):
        self._id = id
        self._cam = cam

    @property
    def present(self):
        k = bool_t()
        _dll.dc1394_feature_is_present(
                self._cam, self._id, byref(k))
        return k.value

    @property
    def switchable(self):
        k = bool_t()
        _dll.dc1394_feature_is_switchable(
                self._cam, self._id, byref(k))
        return bool(k.value)

    @property
    def active(self):
        k = bool_t()
        _dll.dc1394_feature_get_power(
                self._cam, self._id, byref(k))
        return k.value

    @active.setter
    def active(self, value):
        _dll.dc1394_feature_set_power(
            self._cam, self._id, bool(value))

    @property
    def modes(self):
        modes = feature_modes_t()
        _dll.dc1394_feature_get_modes(
                self._cam, self._id, byref(modes))
        return [feature_mode_vals_short[i]
                for i in modes.modes[:modes.num]]

    @property
    def mode(self):
        mode = feature_mode_t()
        _dll.dc1394_feature_get_mode(
                self._cam, self._id, byref(mode))
        return feature_mode_vals[mode.value]

    @mode.setter
    def mode(self, mode):
        key = feature_mode_codes[mode]
        _dll.dc1394_feature_set_mode(
                self._cam, self._id, key)

    @property
    def readable(self):
        k = bool_t()
        _dll.dc1394_feature_is_readable(
                self._cam, self._id, byref(k))
        return k.value

    @property
    def value(self):
        val = c_uint32()
        _dll.dc1394_feature_get_value(
                self._cam, self._id, byref(val))
        return val.value

    @value.setter
    def value(self, value):
        val = int(value)
        _dll.dc1394_feature_set_value(
                self._cam, self._id, val)

    @property
    def value_range(self):
        min, max = c_uint32(), c_uint32()
        _dll.dc1394_feature_get_boundaries(
                self._cam, self._id, byref(min), byref(max))
        return (min.value,max.value)

    @property
    def absolute_capable(self):
        k = bool_t()
        _dll.dc1394_feature_has_absolute_control(
                self._cam, self._id, byref(k))
        return k.value

    @property
    def absolute(self):
        val = c_float()
        _dll.dc1394_feature_get_absolute_value(
                self._cam, self._id, byref(val))
        return val.value

    @absolute.setter
    def absolute(self, value):
        val = float(value)
        _dll.dc1394_feature_set_absolute_value(
                self._cam, self._id, val)

    @property
    def absolute_control(self):
        k = bool_t()
        _dll.dc1394_feature_get_absolute_control(
                self._cam, self._id, byref(k))
        return k.value

    @absolute_control.setter
    def absolute_control(self, value):
        val = int(value)
        _dll.dc1394_feature_set_absolute_control(
                self._cam, self._id, val)

    @property
    def absolute_range(self):
        min, max = c_float(), c_float()
        _dll.dc1394_feature_get_absolute_boundaries(
                self._cam, self._id, byref(min), byref(max))
        return (min.value,max.value)

    def setup(self, value, active=True, mode="manual", absolute=True,
            **kwargs):
        self.active = active
        if not active:
            return
        if mode is not None:
            self.mode = mode
            if mode is "auto":
                return
        for k,v in kwargs.items():
            setattr(self, k, v)
        if absolute is not None:
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
        k = bool_t()
        _dll.dc1394_external_trigger_get_power(
                self._cam, byref(k))
        return bool(k.value)

    @active.setter
    def active(self, value):
        k = bool(value)
        _dll.dc1394_external_trigger_set_power(
                self._cam, k)

    @property
    def modes(self):
        finfo = feature_info_t()
        finfo.id = self._id
        _dll.dc1394_feature_get(
                self._cam, byref(finfo))
        modes = finfo.trigger_modes
        return [trigger_mode_vals_short[i]
                for i in mode.modes[:modes.num]]

    @property
    def mode(self):
        mode = trigger_mode_t()
        _dll.dc1394_external_trigger_get_mode(
                self._cam, byref(mode))
        return trigger_mode_vals_short[mode.value]

    @mode.setter
    def mode(self, value):
        key = trigger_mode_codes_short[value]
        _dll.dc1394_external_trigger_set_mode(
                self._cam, key)

    @property
    def polarity_capable(self):
        finfo = feature_info_t()
        finfo.id = self._id
        _dll.dc1394_feature_get(self._cam, byref(finfo))
        return bool(finfo.polarity_capable)
    
    @property
    def polarity(self):
        pol = trigger_polarity_t()
        _dll.dc1394_external_trigger_get_polarity(
                self._cam, byref(pol))
        return trigger_polarity_vals_short[pol.value]

    @polarity.setter
    def polarity(self, pol):
        key = trigger_polarity_codes_short[pol]
        _dll.dc1394_external_trigger_set_polarity(
                self._cam, key)
    
    @property
    def source(self):
        source = trigger_source_t()
        _dll.dc1394_external_trigger_get_source(
                self._cam, byref(source))
        return trigger_source_vals_short[source.value]

    @source.setter
    def source(self, source):
        key = trigger_source_codes_short[source]
        _dll.dc1394_external_trigger_set_source(
                self._cam, key)

    @property
    def sources(self):
        src = trigger_sources_t()
        _dll.dc1394_external_trigger_get_supported_sources(
                self._cam, byref(src))
        return [trigger_source_vals_short[i]
                for i in src.sources[:src.num]]

    @property
    def software(self):
        res = switch_t()
        _dll.dc1394_software_trigger_get_power(
                self._cam, byref(res))
        return bool(res.value)

    @software.setter
    def software(self, value):
        k = bool(value)
        _dll.dc1394_software_trigger_set_power(
                self._cam, k)


class Whitebalance(Feature):
    @property
    def value(self):
        blue, red = c_uint32(), c_uint32()
        _dll.dc1394_feature_whitebalance_get_value(
            self._cam, byref(blue), byref(red))
        return (blue.value, red.value)
            
    @value.setter
    def value(self, value):
        blue, red = value
        _dll.dc1394_feature_whitebalance_set_value(
                self._cam, blue, red)


class Temperature(Feature):
    @property
    def value(self):
        setpoint, current = c_uint32(), c_uint32()
        _dll.dc1394_feature_temperature_get_value(
            self._cam, byref(setpoint), byref(current))
        return (setpoint.value, current.value)
            
    @value.setter
    def value(self, value):
        setpoint = int(value)
        _dll.dc1394_feature_temperature_set_value(
                self._cam, setpoint)


class Whiteshading(Feature):
    @property
    def value(self):
        r, g, b = c_uint32(), c_uint32(), c_uint32()
        _dll.dc1394_feature_whiteshading_get_value(
            self._cam, byref(r), byref(g), byref(b))
        return (r.value, g.value, b.value)
            
    @value.setter
    def value(self, value):
        r, g, b = map(int, value)
        _dll.dc1394_feature_temperature_set_value(
                self._cam, r, g, b)


feature_map = dict((n, Feature) for n in feature_codes)
feature_map["trigger"] = Trigger
feature_map["white_shading"] = Whiteshading
feature_map["white_balance"] = Whitebalance
feature_map["temperature"] = Temperature


class Mode(object):
    def __init__(self, cam, id):
        self._id = id
        self._cam = cam

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return video_mode_vals[self._id]

    def __str__(self):
        return self.name

    @property
    def rates(self):
        fpss = framerates_t()
        _dll.dc1394_video_get_supported_framerates(
                self._cam, self._id, byref(fpss))
        return [framerate_vals[i]
                for i in fpss.framerates[:fpss.num]]


class Exif(Mode):
    pass


class Format7(Mode):
    @property
    def frame_interval(self):
        fi = c_float()
        _dll.dc1394_format7_get_frame_interval(self._cam,
                    self._id, byref(fi))
        return fi.value

    @property
    def max_size(self):
        hsize = c_uint32()
        vsize = c_uint32()
        _dll.dc1394_format7_get_max_image_size(
                self._cam, self._id,
                byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @property
    def size(self):
        hsize = c_uint32()
        vsize = c_uint32()
        _dll.dc1394_format7_get_image_size(
                self._cam, self._id,
                byref(hsize), byref(vsize))
        return hsize.value, vsize.value

    @size.setter
    def size(self, width, height):
        _dll.dc1394_format7_set_image_size(
                self._cam, self._id,
                width, height)

    @property
    def position(self):
        x = c_uint32()
        y = c_uint32()
        _dll.dc1394_format7_get_image_position(
                self._cam, self._id,
                byref(x), byref(y))
        return x.value, y.value

    @position.setter
    def position(self, x, y):
        _dll.dc1394_format7_set_image_position(
                self._cam, self._id,
                x, y)

    @property
    def color_codings(self):
        pos_codings = color_codings_t()
        _dll.dc1394_format7_get_color_codings(
                self._cam, self._id,
                byref(pos_codings))
        return [color_coding_vals[i]
                for i in pos_codings.codings[:pos_codings.num]]

    @property
    def color_coding(self):
        cc = color_coding_t()
        _dll.dc1394_format7_get_color_coding(
                self._cam, self._id, byref(cc))
        return color_coding_vals[cc.value]

    @color_coding.setter
    def color_coding(self, color):
        code = color_coding_codes[color]
        _dll.dc1394_format7_set_color_coding(
                self._cam, self._id, code)

    @property
    def unit_position(self):
        h_unit = c_uint32()
        v_unit = c_uint32()
        _dll.dc1394_format7_get_unit_position(
                self._cam, self._id,
                byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def unit_size(self):
        h_unit = c_uint32()
        v_unit = c_uint32()
        _dll.dc1394_format7_get_unit_size(
                self._cam, self._id,
                byref(h_unit), byref(v_unit))
        return h_unit.value, v_unit.value

    @property
    def roi(self):
        w, h, x, y = c_int32(), c_int32(), c_int32(), c_int32()
        cco, bpp = color_coding_t(), c_int32()
        _dll.dc1394_format7_get_roi(
            self._cam, self._id, byref(cco), byref(bpp),
            byref(x), byref(y), byref(w), byref(h))
        return ((w.value, h.value), (x.value, y.value),
            color_coding_vals[cco.value], bpp.value)

    @roi.setter
    def roi(self, args):
        size, position, color, bpp = args
        _dll.dc1394_format7_set_roi(
            self._cam, self._id, color_coding_codes[color],
            bpp, position[0], position[1], size[0], size[1])

    def setup(self, size, offset=(0,0), color="Y8", bpp=USE_MAX_AVAIL):
        wu, hu = self.unit_size
        xu, yu = self.unit_position
        position = xu*int(offset[0]/xu), yu*int(offset[1]/yu)
        size = wu*int(size[0]/wu), hu*int(size[1]/hu)
        self.roi = size, position, color, bpp
        return self.roi


video_mode_map = {
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
    _cam = None
    _lib = None

    def __init__(self, guid=None, lib=None, handle=None,
            isospeed=None, mode=None, **features):
        # additionally supplied features are set in undefined order!
        
        if handle is None:
            if lib is None:
                lib = DC1394Library()
            if isinstance(guid, basestring):
                guid, unit = int(guid, 16), None
            elif guid is None:
                guid, unit = lib.cameras[0]
            else:
                unit = None
            handle = lib.camera_handle(guid, unit)
        else:
            assert lib is not None
        self._lib = lib # _we_ need to ensure the dc1394 context is alive
        self._cam = handle

        self._load_features()
        self._load_modes()
        if isospeed is not None:
            self.isospeed = isospeed
        if mode is not None:
            self.mode = self._modes_dict[video_mode_codes[mode]]
        self.setup(**features)

    def __del__(self):
        self.close()

    def close(self):
        if self._cam:
            _dll.dc1394_camera_free(self._cam)
        self._cam = None
        if self._lib:
            self._lib.close()
        self._lib = None
  
    def power(self, on=True):
        _dll.dc1394_camera_set_power(self._cam, on)

    def reset_bus(self):
        _dll.dc1394_reset_bus(self._cam)

    def flush(self):
        frame = POINTER(video_frame_t)()
        while True:
            _dll.dc1394_capture_dequeue(self._cam,
                    CAPTURE_POLICY_POLL, byref(frame))
            if not bool(frame):
                break
            _dll.dc1394_capture_enqueue(self._cam,
                    frame)

    def capture(self, poll=False, mark_corrupt=False):
        frame = POINTER(video_frame_t)()
        policy = poll and CAPTURE_POLICY_POLL or CAPTURE_POLICY_WAIT
        _dll.dc1394_capture_dequeue(self._cam,
                policy, byref(frame))
        if not bool(frame):
            return
        img = Image.from_frame(frame)
        if mark_corrupt:
            img._corrupt = bool(_dll.dc1394_capture_is_frame_corrupt(
                    self._cam, frame))
        _dll.dc1394_capture_enqueue(self._cam, frame)
        return img

    def start_capture(self, bufsize=4):
        _dll.dc1394_capture_setup(
                self._cam, bufsize,
                capture_flag_codes_short["DEFAULT"])

    def stop_capture(self):
        _dll.dc1394_capture_stop(self._cam)

    def start_video(self):
        _dll.dc1394_video_set_transmission(self._cam, 1)

    def stop_video(self):
        _dll.dc1394_video_set_transmission(self._cam, 0)

    def start_one_shot(self):
        _dll.dc1394_video_set_one_shot(self._cam, 1)

    def stop_one_shot(self):
        _dll.dc1394_video_set_one_shot(self._cam, 0)

    def start_multi_shot(self, n):
        _dll.dc1394_video_set_multi_shot(self._cam, n, 1)

    def stop_multi_shot(self):
        _dll.dc1394_video_set_multi_shot(self._cam, 0, 0)

    @property
    def fileno(self):
        return _dll.dc1394_capture_get_fileno(self._cam)

    def _load_features(self):
        fs = featureset_t()
        _dll.dc1394_feature_get_all(self._cam, byref(fs))
        features = {}
        for i in range(FEATURE_NUM):
            s = fs.feature[i]
            if s.available:
                name = feature_vals[s.id]
                feature = feature_map[name](self._cam, s.id)
                features[name] = feature
                setattr(self, name, feature)
        self._features = features

    @property
    def features(self):
        return self._features

    def setup(self, active=True, mode="manual", absolute=True,
            **features):
        for k, v in features.items():
            self.features[k].setup(v, active, mode, absolute)

    def _load_modes(self):
        modes = video_modes_t()
        _dll.dc1394_video_get_supported_modes(self._cam, byref(modes))
        self._modes = [video_mode_map[i](self._cam, i)
                for i in modes.modes[:modes.num]]
        self._modes_dict = dict((m.id, m) for m in self._modes)

    @property
    def modes(self):
        return self._modes

    def get_register(self, offset):
        val = c_uint32()
        _dll.dc1394_get_control_registers(
                self._cam, offset, byref(val), 1)
        return val.value

    def set_register(self, offset, value):
        val = c_uint32(value)
        _dll.dc1394_set_control_registers(
                self._cam, offset, byref(val), 1)

    __getitem__ = get_register
    __setitem__ = set_register

    @property
    def broadcast(self):
        k = bool_t()
        _dll.dc1394_camera_get_broadcast(self._cam, byref(k))
        return bool(k.value)

    @broadcast.setter
    def broadcast(self, value):
        _dll.dc1394_camera_set_broadcast(self._cam, value)

    @property
    def model(self):
        return self._cam.contents.model

    @property
    def guid(self):
        return self._cam.contents.guid

    @property
    def vendor(self):
        return self._cam.contents.vendor

    @property
    def mode(self):
        vmod = video_mode_t()
        _dll.dc1394_video_get_mode(self._cam, byref(vmod))
        return self._modes_dict[vmod.value]

    @mode.setter
    def mode(self, mode):
        _dll.dc1394_video_set_mode(self._cam, mode.id)

    @property
    def rate(self):
        ft = framerate_t()
        _dll.dc1394_video_get_framerate(self._cam, byref(ft))
        return framerate_vals[ft.value]

    @rate.setter
    def rate(self, framerate):
        wanted_frate = framerate_codes[framerate]
        _dll.dc1394_video_set_framerate(self._cam, wanted_frate)
    
    @property
    def isospeed(self):
        sp = speed_t()
        _dll.dc1394_video_get_iso_speed(self._cam, byref(sp))
        return speed_vals[sp.value]

    @isospeed.setter
    def isospeed(self, isospeed):
        sp = speed_codes[isospeed]
        self.operation_mode = 'LEGACY' if isospeed < 800 else '1394B'
        _dll.dc1394_video_set_iso_speed(self._cam, sp)

    @property
    def operation_mode(self):
        k = operation_mode_t()
        _dll.dc1394_video_get_operation_mode(self._cam, byref(k))
        return operation_mode_vals_short[k.value]

    @operation_mode.setter
    def operation_mode(self, value):
        k = operation_mode_codes_short[value]
        _dll.dc1394_video_set_operation_mode(self._cam, k)

