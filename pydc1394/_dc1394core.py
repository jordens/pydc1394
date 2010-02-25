#!/usr/bin/env python
# encoding: utf-8
#
# File: _dc1394core.py
#
# Created by Holger Rapp on 2008-07-26.
# Copyright (c) 2008 HolgerRapp@gmx.net. All rights reserved.

"""This file wraps the dc1394 library functions"""

# __all__ = [ "_dll", "dc1394camera_t", "dc1394camera_list_t", "dc1394camera_id_t" ]

from ctypes import *
import sys

if sys.platform.startswith("linux"):
    _dll = cdll.LoadLibrary("libdc1394.so.22")
else:
    def _try_loadcdll(dll, path):
        try:
            return cdll.LoadLibrary(path + "libdc1394.22.dylib")
        except OSError, e:
            return None
    for p in ("", "/usr/local/lib/", "/opt/local/lib/", "/sw/lib/"):
        _dll = _try_loadcdll("libdc1394.22.dylib", p)
        if _dll is not None:
            break
    if _dll is None:
        raise OSError("Dc1394 library was not found!")


###########################################################################
#                                  ENUMS                                  #
###########################################################################
DC1394_SUCCESS                     =  0
DC1394_FAILURE                     = -1
DC1394_NOT_A_CAMERA                = -2
DC1394_FUNCTION_NOT_SUPPORTED      = -3
DC1394_CAMERA_NOT_INITIALIZED      = -4
DC1394_MEMORY_ALLOCATION_FAILURE   = -5
DC1394_TAGGED_REGISTER_NOT_FOUND   = -6
DC1394_NO_ISO_CHANNEL              = -7
DC1394_NO_BANDWIDTH                = -8
DC1394_IOCTL_FAILURE               = -9
DC1394_CAPTURE_IS_NOT_SET          = -10
DC1394_CAPTURE_IS_RUNNING          = -11
DC1394_RAW1394_FAILURE             = -12
DC1394_FORMAT7_ERROR_FLAG_1        = -13
DC1394_FORMAT7_ERROR_FLAG_2        = -14
DC1394_INVALID_ARGUMENT_VALUE      = -15
DC1394_REQ_VALUE_OUTSIDE_RANGE     = -16
DC1394_INVALID_FEATURE             = -17
DC1394_INVALID_VIDEO_FORMAT        = -18
DC1394_INVALID_VIDEO_MODE          = -19
DC1394_INVALID_FRAMERATE           = -20
DC1394_INVALID_TRIGGER_MODE        = -21
DC1394_INVALID_TRIGGER_SOURCE      = -22
DC1394_INVALID_ISO_SPEED           = -23
DC1394_INVALID_IIDC_VERSION        = -24
DC1394_INVALID_COLOR_CODING        = -25
DC1394_INVALID_COLOR_FILTER        = -26
DC1394_INVALID_CAPTURE_POLICY      = -27
DC1394_INVALID_ERROR_CODE          = -28
DC1394_INVALID_BAYER_METHOD        = -29
DC1394_INVALID_VIDEO1394_DEVICE    = -30
DC1394_INVALID_OPERATION_MODE      = -31
DC1394_INVALID_TRIGGER_POLARITY    = -32
DC1394_INVALID_FEATURE_MODE        = -33
DC1394_INVALID_LOG_TYPE            = -34
DC1394_INVALID_BYTE_ORDER          = -35
DC1394_INVALID_STEREO_METHOD       = -36
DC1394_BASLER_NO_MORE_SFF_CHUNKS   = -37
DC1394_BASLER_CORRUPTED_SFF_CHUNK  = -38
DC1394_BASLER_UNKNOWN_SFF_CHUNK    = -39
dc1394error_t_vals = {
        0: 'DC1394_SUCCESS', 
       -1: 'DC1394_FAILURE', 
       -2: 'DC1394_NOT_A_CAMERA', 
       -3: 'DC1394_FUNCTION_NOT_SUPPORTED', 
       -4: 'DC1394_CAMERA_NOT_INITIALIZED', 
       -5: 'DC1394_MEMORY_ALLOCATION_FAILURE', 
       -6: 'DC1394_TAGGED_REGISTER_NOT_FOUND', 
       -7: 'DC1394_NO_ISO_CHANNEL', 
       -8: 'DC1394_NO_BANDWIDTH', 
       -9: 'DC1394_IOCTL_FAILURE', 
      -10: 'DC1394_CAPTURE_IS_NOT_SET', 
      -11: 'DC1394_CAPTURE_IS_RUNNING', 
      -12: 'DC1394_RAW1394_FAILURE', 
      -13: 'DC1394_FORMAT7_ERROR_FLAG_1', 
      -14: 'DC1394_FORMAT7_ERROR_FLAG_2', 
      -15: 'DC1394_INVALID_ARGUMENT_VALUE', 
      -16: 'DC1394_REQ_VALUE_OUTSIDE_RANGE', 
      -17: 'DC1394_INVALID_FEATURE', 
      -18: 'DC1394_INVALID_VIDEO_FORMAT', 
      -19: 'DC1394_INVALID_VIDEO_MODE', 
      -20: 'DC1394_INVALID_FRAMERATE', 
      -21: 'DC1394_INVALID_TRIGGER_MODE', 
      -22: 'DC1394_INVALID_TRIGGER_SOURCE', 
      -23: 'DC1394_INVALID_ISO_SPEED', 
      -24: 'DC1394_INVALID_IIDC_VERSION', 
      -25: 'DC1394_INVALID_COLOR_CODING', 
      -26: 'DC1394_INVALID_COLOR_FILTER', 
      -27: 'DC1394_INVALID_CAPTURE_POLICY', 
      -28: 'DC1394_INVALID_ERROR_CODE', 
      -29: 'DC1394_INVALID_BAYER_METHOD', 
      -30: 'DC1394_INVALID_VIDEO1394_DEVICE', 
      -31: 'DC1394_INVALID_OPERATION_MODE', 
      -32: 'DC1394_INVALID_TRIGGER_POLARITY', 
      -33: 'DC1394_INVALID_FEATURE_MODE', 
      -34: 'DC1394_INVALID_LOG_TYPE', 
      -35: 'DC1394_INVALID_BYTE_ORDER', 
      -36: 'DC1394_INVALID_STEREO_METHOD', 
      -37: 'DC1394_BASLER_NO_MORE_SFF_CHUNKS', 
      -38: 'DC1394_BASLER_CORRUPTED_SFF_CHUNK', 
      -39: 'DC1394_BASLER_UNKNOWN_SFF_CHUNK', 
}
dc1394error_t = c_int
DC1394_ERROR_MIN = DC1394_BASLER_UNKNOWN_SFF_CHUNK
DC1394_ERROR_MAX = DC1394_SUCCESS
DC1394_ERROR_NUM =(DC1394_ERROR_MAX-DC1394_ERROR_MIN+1)


DC1394_IIDC_VERSION_1_04   = 544
DC1394_IIDC_VERSION_1_20   = 545
DC1394_IIDC_VERSION_PTGREY = 546
DC1394_IIDC_VERSION_1_30   = 547
DC1394_IIDC_VERSION_1_31   = 548
DC1394_IIDC_VERSION_1_32   = 549
DC1394_IIDC_VERSION_1_33   = 550
DC1394_IIDC_VERSION_1_34   = 551
DC1394_IIDC_VERSION_1_35   = 552
DC1394_IIDC_VERSION_1_36   = 553
DC1394_IIDC_VERSION_1_37   = 554
DC1394_IIDC_VERSION_1_38   = 555
DC1394_IIDC_VERSION_1_39   = 556
dc1394iidc_version_t_vals = {
      544: 'DC1394_IIDC_VERSION_1_04',
      545: 'DC1394_IIDC_VERSION_1_20',
      546: 'DC1394_IIDC_VERSION_PTGREY',
      547: 'DC1394_IIDC_VERSION_1_30',
      548: 'DC1394_IIDC_VERSION_1_31',
      549: 'DC1394_IIDC_VERSION_1_32',
      550: 'DC1394_IIDC_VERSION_1_33',
      551: 'DC1394_IIDC_VERSION_1_34',
      552: 'DC1394_IIDC_VERSION_1_35',
      553: 'DC1394_IIDC_VERSION_1_36',
      554: 'DC1394_IIDC_VERSION_1_37',
      555: 'DC1394_IIDC_VERSION_1_38',
      556: 'DC1394_IIDC_VERSION_1_39',
}
dc1394iidc_version_t = c_int


DC1394_VIDEO_MODE_160x120_YUV444   = 64
DC1394_VIDEO_MODE_320x240_YUV422   = 65
DC1394_VIDEO_MODE_640x480_YUV411   = 66
DC1394_VIDEO_MODE_640x480_YUV422   = 67
DC1394_VIDEO_MODE_640x480_RGB8     = 68
DC1394_VIDEO_MODE_640x480_MONO8    = 69
DC1394_VIDEO_MODE_640x480_MONO16   = 70
DC1394_VIDEO_MODE_800x600_YUV422   = 71
DC1394_VIDEO_MODE_800x600_RGB8     = 72
DC1394_VIDEO_MODE_800x600_MONO8    = 73
DC1394_VIDEO_MODE_1024x768_YUV422  = 74
DC1394_VIDEO_MODE_1024x768_RGB8    = 75
DC1394_VIDEO_MODE_1024x768_MONO8   = 76
DC1394_VIDEO_MODE_800x600_MONO16   = 77
DC1394_VIDEO_MODE_1024x768_MONO16  = 78
DC1394_VIDEO_MODE_1280x960_YUV422  = 79
DC1394_VIDEO_MODE_1280x960_RGB8    = 80
DC1394_VIDEO_MODE_1280x960_MONO8   = 81
DC1394_VIDEO_MODE_1600x1200_YUV422 = 82
DC1394_VIDEO_MODE_1600x1200_RGB8   = 83
DC1394_VIDEO_MODE_1600x1200_MONO8  = 84
DC1394_VIDEO_MODE_1280x960_MONO16  = 85
DC1394_VIDEO_MODE_1600x1200_MONO16 = 86
DC1394_VIDEO_MODE_EXIF             = 87
DC1394_VIDEO_MODE_FORMAT7_0        = 88
DC1394_VIDEO_MODE_FORMAT7_1        = 89
DC1394_VIDEO_MODE_FORMAT7_2        = 90
DC1394_VIDEO_MODE_FORMAT7_3        = 91
DC1394_VIDEO_MODE_FORMAT7_4        = 92
DC1394_VIDEO_MODE_FORMAT7_5        = 93
DC1394_VIDEO_MODE_FORMAT7_6        = 94
DC1394_VIDEO_MODE_FORMAT7_7        = 95
dc1394video_mode_t_vals = {
       64: (160,120,'YUV444'),
       65: (320,240,'YUV422'),
       66: (640,480,'YUV411'),
       67: (640,480,'YUV422'),
       68: (640,480,'RGB8'),
       69: (640,480,'Y8'),
       70: (640,480,'Y16'),
       71: (800,600,'YUV422'),
       72: (800,600,'RGB8'),
       73: (800,600,'Y8'),
       74: (1024,768,'YUV422'),
       75: (1024,768,'RGB8'),
       76: (1024,768,'Y8'),
       77: (800,600,'Y16'),
       78: (1024,768,'Y16'),
       79: (1280,960,'YUV422'),
       80: (1280,960,'RGB8'),
       81: (1280,960,'Y8'),
       82: (1600,1200,'YUV422'),
       83: (1600,1200,'RGB8'),
       84: (1600,1200,'Y8'),
       85: (1280,960,'Y16'),
       86: (1600,1200,'Y16'),
       87: 'EXIF',
       88: 'FORMAT7_0',
       89: 'FORMAT7_1',
       90: 'FORMAT7_2',
       91: 'FORMAT7_3',
       92: 'FORMAT7_4',
       93: 'FORMAT7_5',
       94: 'FORMAT7_6',
       95: 'FORMAT7_7',
}
dc1394video_mode_t = c_int
DC1394_VIDEO_MODE_MIN = DC1394_VIDEO_MODE_160x120_YUV444
DC1394_VIDEO_MODE_MAX = DC1394_VIDEO_MODE_FORMAT7_7
DC1394_VIDEO_MODE_NUM = (DC1394_VIDEO_MODE_MAX - DC1394_VIDEO_MODE_MIN + 1)
DC1394_VIDEO_MODE_FORMAT7_MIN = DC1394_VIDEO_MODE_FORMAT7_0
DC1394_VIDEO_MODE_FORMAT7_MAX = DC1394_VIDEO_MODE_FORMAT7_7
DC1394_VIDEO_MODE_FORMAT7_NUM = (DC1394_VIDEO_MODE_FORMAT7_MAX - DC1394_VIDEO_MODE_FORMAT7_MIN + 1)

# Video modes
class dc1394video_modes_t(Structure):
     pass
dc1394video_modes_t._fields_ = [
    ("num", c_uint32),
    ("modes", (dc1394video_mode_t)*DC1394_VIDEO_MODE_NUM),
]

# Color codings
DC1394_COLOR_CODING_MONO8   = 352
DC1394_COLOR_CODING_YUV411  = 353
DC1394_COLOR_CODING_YUV422  = 354
DC1394_COLOR_CODING_YUV444  = 355
DC1394_COLOR_CODING_RGB8    = 356
DC1394_COLOR_CODING_MONO16  = 357
DC1394_COLOR_CODING_RGB16   = 358
DC1394_COLOR_CODING_MONO16S = 359
DC1394_COLOR_CODING_RGB16S  = 360
DC1394_COLOR_CODING_RAW8    = 361
DC1394_COLOR_CODING_RAW16   = 362
dc1394color_coding_t_vals = {
      352: 'Y8',
      353: 'YUV411',
      354: 'YUV422',
      355: 'YUV444',
      356: 'RGB8',
      357: 'Y16',
      358: 'RGB16',
      359: 'Y16S',
      360: 'RGB16S',
      361: 'RAW8',
      362: 'RAW16',
}
dc1394color_coding_t = c_int
DC1394_COLOR_CODING_MIN =  DC1394_COLOR_CODING_MONO8
DC1394_COLOR_CODING_MAX =  DC1394_COLOR_CODING_RAW16
DC1394_COLOR_CODING_NUM = (DC1394_COLOR_CODING_MAX - DC1394_COLOR_CODING_MIN + 1)


DC1394_COLOR_FILTER_RGGB = 512
DC1394_COLOR_FILTER_RGGB = 512
DC1394_COLOR_FILTER_GBRG = 513
DC1394_COLOR_FILTER_GRBG = 514
DC1394_COLOR_FILTER_BGGR = 515
dc1394color_filter_t_vals = {
      512: 'DC1394_COLOR_FILTER_RGGB',
      513: 'DC1394_COLOR_FILTER_GBRG',
      514: 'DC1394_COLOR_FILTER_GRBG',
      515: 'DC1394_COLOR_FILTER_BGGR',
}
dc1394color_filter_t = c_int
DC1394_COLOR_FILTER_MIN =  DC1394_COLOR_FILTER_RGGB
DC1394_COLOR_FILTER_MAX =  DC1394_COLOR_FILTER_BGGR
DC1394_COLOR_FILTER_NUM = (DC1394_COLOR_FILTER_MAX - DC1394_COLOR_FILTER_MIN + 1)

# Byte order
DC1394_BYTE_ORDER_UYVY = 800
DC1394_BYTE_ORDER_YUYV = 801
dc1394byte_order_t_vals = {
      800: 'DC1394_BYTE_ORDER_UYVY',
      801: 'DC1394_BYTE_ORDER_YUYV',
}
dc1394byte_order_t = c_int
DC1394_BYTE_ORDER_MIN =  DC1394_BYTE_ORDER_UYVY
DC1394_BYTE_ORDER_MAX =  DC1394_BYTE_ORDER_YUYV
DC1394_BYTE_ORDER_NUM = (DC1394_BYTE_ORDER_MAX - DC1394_BYTE_ORDER_MIN + 1)
class dc1394color_codings_t(Structure):
     pass
dc1394color_codings_t._fields_ = [
    ("num", c_uint32),
    ("codings", (dc1394color_coding_t)*DC1394_COLOR_CODING_NUM),
]
class dc1394video_mode_t(Structure):
     pass
dc1394video_mode_t._fields_ = [
    ("num", c_uint32),
    ("modes", (dc1394video_mode_t)*DC1394_VIDEO_MODE_NUM),
]
dc1394bool_t = c_int
dc1394switch_t = c_int


###########################################################################
#                               STRUCTURES                                #
###########################################################################
class dc1394camera_t(Structure):
     pass
dc1394camera_t._fields_ = [
    ("guid", c_uint64),
    ("unit", c_int),
    ("unit_spec_ID", c_uint32),
    ("unit_sw_version", c_uint32),
    ("unit_sub_sw_version", c_uint32),
    ("command_registers_base", c_uint32),
    ("unit_directory", c_uint32),
    ("unit_dependent_directory", c_uint32),
    ("advanced_features_csr", c_uint64),
    ("PIO_control_csr", c_uint64),
    ("SIO_control_csr", c_uint64),
    ("strobe_control_csr", c_uint64),
    ("format7_csr", (c_uint64)*8),
    ("iidc_version", dc1394iidc_version_t),
    ("vendor", c_char_p),
    ("model", c_char_p),
    ("vendor_id", c_uint32),
    ("model_id", c_uint32),
    ("bmode_capable", dc1394bool_t),
    ("one_shot_capable", dc1394bool_t),
    ("multi_shot_capable", dc1394bool_t),
    ("can_switch_on_off", dc1394bool_t),
    ("has_vmode_error_status", dc1394bool_t),
    ("has_feature_error_status", dc1394bool_t),
    ("max_mem_channel", c_int),
    ("flags", c_uint32),
]

class dc1394camera_id_t(Structure):
     pass
dc1394camera_id_t._fields_ = [
    ("unit", c_uint16),
    ("guid", c_uint64),
]

class dc1394camera_list_t(Structure):
     pass
dc1394camera_list_t._fields_ = [
    ("num", c_uint32),
    ("ids", POINTER(dc1394camera_id_t)),
]

DC1394_TRIGGER_MODE_0  = 384
DC1394_TRIGGER_MODE_1  = 385
DC1394_TRIGGER_MODE_2  = 386
DC1394_TRIGGER_MODE_3  = 387
DC1394_TRIGGER_MODE_4  = 388
DC1394_TRIGGER_MODE_5  = 389
DC1394_TRIGGER_MODE_14 = 390
DC1394_TRIGGER_MODE_15 = 391
dc1394trigger_mode_t_vals = {
      384: 'DC1394_TRIGGER_MODE_0',
      385: 'DC1394_TRIGGER_MODE_1',
      386: 'DC1394_TRIGGER_MODE_2',
      387: 'DC1394_TRIGGER_MODE_3',
      388: 'DC1394_TRIGGER_MODE_4',
      389: 'DC1394_TRIGGER_MODE_5',
      390: 'DC1394_TRIGGER_MODE_14',
      391: 'DC1394_TRIGGER_MODE_15',
}
dc1394trigger_mode_t = c_int
DC1394_TRIGGER_MODE_MIN = DC1394_TRIGGER_MODE_0
DC1394_TRIGGER_MODE_MAX = DC1394_TRIGGER_MODE_15
DC1394_TRIGGER_MODE_NUM =(DC1394_TRIGGER_MODE_MAX - DC1394_TRIGGER_MODE_MIN + 1)


DC1394_FRAMERATE_1_875 = 32
DC1394_FRAMERATE_3_75  = 33
DC1394_FRAMERATE_7_5   = 34
DC1394_FRAMERATE_15    = 35
DC1394_FRAMERATE_30    = 36
DC1394_FRAMERATE_60    = 37
DC1394_FRAMERATE_120   = 38
DC1394_FRAMERATE_240   = 39
dc1394framerate_t_vals = {
       32: 1.875, 
       33: 3.75, 
       34: 7.5, 
       35: 15., 
       36: 30, 
       37: 60., 
       38: 120., 
       39: 240., 
}
dc1394framerate_t = c_int
DC1394_FRAMERATE_MIN = DC1394_FRAMERATE_1_875
DC1394_FRAMERATE_MAX = DC1394_FRAMERATE_240
DC1394_FRAMERATE_NUM =(DC1394_FRAMERATE_MAX - DC1394_FRAMERATE_MIN + 1)

class dc1394framerates_t(Structure):
    pass
dc1394framerates_t._fields_ = [
    ("num", c_uint32),
    ("framerates", (dc1394framerate_t)*DC1394_FRAMERATE_NUM),
]

DC1394_ISO_SPEED_100  = 0
DC1394_ISO_SPEED_200  = 1
DC1394_ISO_SPEED_400  = 2
DC1394_ISO_SPEED_800  = 3
DC1394_ISO_SPEED_1600 = 4
DC1394_ISO_SPEED_3200 = 5
dc1394speed_t_vals = {
        0: 100, 
        1: 200, 
        2: 400, 
        3: 800, 
        4: 1600, 
        5: 3200, 
}
dc1394speed_t = c_int
DC1394_ISO_SPEED_MIN =  DC1394_ISO_SPEED_100
DC1394_ISO_SPEED_MAX =  DC1394_ISO_SPEED_3200
DC1394_ISO_SPEED_NUM = (DC1394_ISO_SPEED_MAX - DC1394_ISO_SPEED_MIN + 1)

DC1394_FEATURE_BRIGHTNESS      = 416
DC1394_FEATURE_EXPOSURE        = 417
DC1394_FEATURE_SHARPNESS       = 418
DC1394_FEATURE_WHITE_BALANCE   = 419
DC1394_FEATURE_HUE             = 420
DC1394_FEATURE_SATURATION      = 421
DC1394_FEATURE_GAMMA           = 422
DC1394_FEATURE_SHUTTER         = 423
DC1394_FEATURE_GAIN            = 424
DC1394_FEATURE_IRIS            = 425
DC1394_FEATURE_FOCUS           = 426
DC1394_FEATURE_TEMPERATURE     = 427
DC1394_FEATURE_TRIGGER         = 428
DC1394_FEATURE_TRIGGER_DELAY   = 429
DC1394_FEATURE_WHITE_SHADING   = 430
DC1394_FEATURE_FRAME_RATE      = 431
DC1394_FEATURE_ZOOM            = 432
DC1394_FEATURE_PAN             = 433
DC1394_FEATURE_TILT            = 434
DC1394_FEATURE_OPTICAL_FILTER  = 435
DC1394_FEATURE_CAPTURE_SIZE    = 436
DC1394_FEATURE_CAPTURE_QUALITY = 437
dc1394feature_t_vals = {
      416: 'brightness',
      417: 'exposure',
      418: 'sharpness',
      419: 'white_balance',
      420: 'hue',
      421: 'saturation',
      422: 'gamma',
      423: 'shutter',
      424: 'gain',
      425: 'iris',
      426: 'focus',
      427: 'temperature',
      428: 'trigger',
      429: 'trigger_delay',
      430: 'white_shading',
      431: 'framerate',
      432: 'zoom',
      433: 'pan',
      434: 'tilt',
      435: 'optical_filter',
      436: 'capture_size',
      437: 'capture_quality',
}
dc1394feature_t = c_int
DC1394_FEATURE_MIN =  DC1394_FEATURE_BRIGHTNESS
DC1394_FEATURE_MAX =  DC1394_FEATURE_CAPTURE_QUALITY
DC1394_FEATURE_NUM = (DC1394_FEATURE_MAX - DC1394_FEATURE_MIN + 1)


DC1394_TRIGGER_SOURCE_0        = 576
DC1394_TRIGGER_SOURCE_1        = 577
DC1394_TRIGGER_SOURCE_2        = 578
DC1394_TRIGGER_SOURCE_3        = 579
DC1394_TRIGGER_SOURCE_SOFTWARE = 580
dc1394trigger_source_t_0_vals = {
      576: 'DC1394_TRIGGER_SOURCE_0',
      577: 'DC1394_TRIGGER_SOURCE_1',
      578: 'DC1394_TRIGGER_SOURCE_2',
      579: 'DC1394_TRIGGER_SOURCE_3',
      580: 'DC1394_TRIGGER_SOURCE_SOFTWARE',
}
dc1394trigger_source_t = c_int
DC1394_TRIGGER_SOURCE_MIN = DC1394_TRIGGER_SOURCE_0
DC1394_TRIGGER_SOURCE_MAX = DC1394_TRIGGER_SOURCE_SOFTWARE
DC1394_TRIGGER_SOURCE_NUM =(DC1394_TRIGGER_SOURCE_MAX - DC1394_TRIGGER_SOURCE_MIN + 1)


DC1394_TRIGGER_ACTIVE_LOW  = 704
DC1394_TRIGGER_ACTIVE_HIGH = 705
dc1394trigger_polarity_t_vals = {
      704: 'DC1394_TRIGGER_ACTIVE_LOW',
      705: 'DC1394_TRIGGER_ACTIVE_HIGH',
}
dc1394trigger_polarity_t = c_int
DC1394_TRIGGER_ACTIVE_MIN = DC1394_TRIGGER_ACTIVE_LOW
DC1394_TRIGGER_ACTIVE_MAX = DC1394_TRIGGER_ACTIVE_HIGH
DC1394_TRIGGER_ACTIVE_NUM =(DC1394_TRIGGER_ACTIVE_MAX - DC1394_TRIGGER_ACTIVE_MIN + 1)


DC1394_FEATURE_MODE_MANUAL        = 736
DC1394_FEATURE_MODE_AUTO          = 737
DC1394_FEATURE_MODE_ONE_PUSH_AUTO = 738
dc1394feature_mode_t_vals = {
      736: 'manual',
      737: 'auto',
      738: 'one_push',
}
dc1394feature_mode_t = c_int
DC1394_FEATURE_MODE_MIN =  DC1394_FEATURE_MODE_MANUAL
DC1394_FEATURE_MODE_MAX =  DC1394_FEATURE_MODE_ONE_PUSH_AUTO
DC1394_FEATURE_MODE_NUM = (DC1394_FEATURE_MODE_MAX - DC1394_FEATURE_MODE_MIN + 1)

class dc1394feature_modes_t(Structure):
     pass
dc1394feature_modes_t._fields_ = [
    ("num", c_uint32),
    ("modes", (dc1394feature_mode_t)*DC1394_FEATURE_MODE_NUM),
]

class dc1394trigger_modes_t(Structure):
     pass
dc1394trigger_modes_t._fields_ = [
    ("num", c_uint32),
    ("modes", (dc1394trigger_mode_t)*DC1394_TRIGGER_MODE_NUM),
]

class dc1394trigger_sources_t(Structure):
     pass
dc1394trigger_sources_t._fields_ = [
    ("num", c_uint32),
    ("sources", (dc1394trigger_source_t)*DC1394_TRIGGER_SOURCE_NUM),
]

class dc1394feature_info_t(Structure):
     pass
dc1394feature_info_t._fields_ = [
    ("id", dc1394feature_t),
    ("available", dc1394bool_t),
    ("absolute_capable", dc1394bool_t),
    ("readout_capable", dc1394bool_t),
    ("on_off_capable", dc1394bool_t),
    ("polarity_capable", dc1394bool_t),
    ("is_on", dc1394switch_t),
    ("current_mode", dc1394feature_mode_t),
    ("modes", dc1394feature_modes_t),
    ("trigger_modes", dc1394trigger_modes_t),
    ("trigger_mode", dc1394trigger_mode_t),
    ("trigger_polarity", dc1394trigger_polarity_t),
    ("trigger_sources", dc1394trigger_sources_t),
    ("trigger_source", dc1394trigger_source_t),
    ("min", c_uint32),
    ("max", c_uint32),
    ("value", c_uint32),
    ("BU_value", c_uint32),
    ("RV_value", c_uint32),
    ("B_value", c_uint32),
    ("R_value", c_uint32),
    ("G_value", c_uint32),
    ("target_value", c_uint32),
    ("abs_control", dc1394switch_t),
    ("abs_value", c_float),
    ("abs_max", c_float),
    ("abs_min", c_float),
]

class dc1394featureset_t(Structure):
     pass
dc1394featureset_t._fields_ = [
    ("feature", (dc1394feature_info_t)*DC1394_FEATURE_NUM),
]

DC1394_CAPTURE_FLAGS_CHANNEL_ALLOC   = 0x00000001
DC1394_CAPTURE_FLAGS_BANDWIDTH_ALLOC = 0x00000002
DC1394_CAPTURE_FLAGS_DEFAULT         = 0x00000004
DC1394_CAPTURE_FLAGS_AUTO_ISO        = 0x00000008 


class dc1394video_frame_t(Structure):
     pass
dc1394video_frame_t._fields_ = [
    ("image", c_void_p), 
    ("size", (c_uint32)*2),
    ("position", (c_uint32)*2),
    ("color_coding", dc1394color_coding_t),
    ("color_filter", dc1394color_filter_t),
    ("yuv_byte_order", c_uint32),
    ("data_depth", c_uint32),
    ("stride", c_uint32),
    ("video_mode", dc1394video_mode_t),
    ("total_bytes", c_uint64),
    ("image_bytes", c_uint32),
    ("padding_bytes", c_uint32),
    ("packet_size", c_uint32),
    ("packets_per_frame", c_uint32),
    ("timestamp", c_uint64),
    ("frames_behind", c_uint32),
    ("camera", POINTER(dc1394camera_t)),
    ("id", c_uint32),
    ("allocated_image_bytes", c_uint64),
    ("little_endian", dc1394bool_t),
    ("data_in_padding", dc1394bool_t),
]
DC1394_CAPTURE_POLICY_WAIT=672
DC1394_CAPTURE_POLICY_POLL=673


DC1394_QUERY_FROM_CAMERA= -1
DC1394_USE_MAX_AVAIL    = -2
DC1394_USE_RECOMMENDED  = -3

###########################################################################
#                            PYTHON FUNCTIONS                             #
###########################################################################
# Global Error checking functions
def _errcheck( rtype, func, arg ):
    """This function checks for a non zero return type. This
    is in most cases an error"""
    if rtype != 0:
        raise RuntimeError, "Error in dc1394 function call: %s" % dc1394error_t_vals[rtype]
    return rtype


###########################################################################
#                                FUNCTIONS                                #
###########################################################################

##################
# INITIALIZATION #
##################
_dll.dc1394_new.restype = c_void_p
_dll.dc1394_free.restype = None
_dll.dc1394_free.argtypes = [ c_void_p ]
_dll.dc1394_reset_bus.errcheck = _errcheck
_dll.dc1394_reset_bus.argtypes = [ POINTER(dc1394camera_t) ]

###################
# OPENING/CLOSING #
###################
_dll.dc1394_camera_enumerate.argtypes = [ c_void_p, POINTER(POINTER(dc1394camera_list_t)) ]

_dll.dc1394_camera_free_list.restype = None

_dll.dc1394_camera_new.argtypes = [ c_void_p, c_uint64 ]
_dll.dc1394_camera_new.restype = POINTER(dc1394camera_t)
_dll.dc1394_camera_free.argtypes = [ POINTER(dc1394camera_t) ]
_dll.dc1394_camera_free.restype = None

###################
# FEATURE CONTROL #
###################
_dll.dc1394_feature_get_all.argtypes = [ c_void_p, POINTER(dc1394featureset_t) ]
_dll.dc1394_camera_get_broadcast.argtypes = [ c_void_p, POINTER(dc1394bool_t) ]
_dll.dc1394_camera_get_broadcast.errcheck = _errcheck
_dll.dc1394_camera_set_broadcast.argtypes = [ c_void_p, dc1394bool_t ]
_dll.dc1394_camera_set_broadcast.errcheck = _errcheck
_dll.dc1394_feature_whitebalance_get_value.argtypes = [ c_void_p, POINTER(c_uint32), POINTER(c_uint32) ]
_dll.dc1394_feature_whitebalance_get_value.errcheck = _errcheck
_dll.dc1394_feature_whitebalance_set_value.argtypes = [ c_void_p, c_uint32, c_uint32 ]
_dll.dc1394_feature_whitebalance_set_value.errcheck = _errcheck
_dll.dc1394_feature_temperature_get_value.argtypes = [ c_void_p, POINTER(c_uint32), POINTER(c_uint32) ]
_dll.dc1394_feature_temperature_get_value.errcheck = _errcheck
_dll.dc1394_feature_temperature_set_value.argtypes = [ c_void_p, c_uint32, c_uint32 ]
_dll.dc1394_feature_temperature_set_value.errcheck = _errcheck
_dll.dc1394_feature_get_value.argtypes = [ c_void_p, c_int, POINTER(c_uint32) ]
_dll.dc1394_feature_get_value.errcheck = _errcheck
_dll.dc1394_feature_set_value.argtypes = [ c_void_p, c_int, c_uint32 ]
_dll.dc1394_feature_set_value.errcheck = _errcheck
_dll.dc1394_feature_get_absolute_value.argtypes = [ c_void_p, c_int, POINTER(c_float) ]
_dll.dc1394_feature_get_absolute_value.errcheck = _errcheck
_dll.dc1394_feature_set_absolute_value.argtypes = [ c_void_p, c_int, c_float ]
_dll.dc1394_feature_set_absolute_value.errcheck = _errcheck
_dll.dc1394_feature_set_absolute_control.errcheck = _errcheck
_dll.dc1394_feature_get_boundaries.errcheck = _errcheck
_dll.dc1394_feature_get_absolute_boundaries.errcheck = _errcheck
_dll.dc1394_feature_is_switchable.errcheck = _errcheck
_dll.dc1394_feature_get_power.errcheck = _errcheck
_dll.dc1394_feature_set_power.errcheck = _errcheck
_dll.dc1394_feature_get_modes.errcheck = _errcheck
_dll.dc1394_feature_get_mode.errcheck = _errcheck
_dll.dc1394_feature_set_mode.errcheck = _errcheck

###################
# VIDEO FUNCTIONS #
###################
_dll.dc1394_video_get_supported_modes.argtypes = [ c_void_p, POINTER(dc1394video_modes_t) ]
_dll.dc1394_video_set_transmission.argtypes = [ c_void_p, c_int ]
_dll.dc1394_video_set_transmission.errcheck = _errcheck
_dll.dc1394_capture_stop.argtypes = [ c_void_p ]
_dll.dc1394_capture_stop.errcheck = _errcheck
_dll.dc1394_capture_dequeue.argtypes = [ c_void_p, c_int, POINTER(POINTER(dc1394video_frame_t)) ]
_dll.dc1394_capture_enqueue.argtypes = [ c_void_p, POINTER(dc1394video_frame_t) ]
_dll.dc1394_video_get_operation_mode.argtypes = [ c_void_p, POINTER(c_int) ]
_dll.dc1394_video_get_operation_mode.errcheck = _errcheck
_dll.dc1394_video_set_operation_mode.argtypes = [ c_void_p, c_int]
_dll.dc1394_video_set_operation_mode.errcheck = _errcheck
_dll.dc1394_video_set_iso_speed.errcheck = _errcheck
_dll.dc1394_video_set_mode.errcheck = _errcheck
_dll.dc1394_video_set_framerate.errcheck = _errcheck
_dll.dc1394_video_get_supported_framerates.errcheck = _errcheck

######################
# REGISTER FUNCTIONS #
######################
_dll.dc1394_get_control_registers.errcheck = _errcheck
_dll.dc1394_get_control_registers.argtypes = [ c_void_p, c_uint64, POINTER(c_uint32), c_uint32 ]
_dll.dc1394_set_control_registers.errcheck = _errcheck
_dll.dc1394_set_control_registers.argtypes = [ c_void_p, c_uint64, POINTER(c_uint32),c_uint32 ]

#####################
# CAPTURE FUNCTIONS #
#####################
_dll.dc1394_capture_setup.errcheck = _errcheck

######################
# FORMAT 7 FUNCTIONS #
######################
_dll.dc1394_format7_set_roi.errcheck = _errcheck

