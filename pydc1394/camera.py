#!/usr/bin/env python
# encoding: utf-8
#
# File: camera.py
#
# Created by Holger Rapp on 2008-07-26.
# Copyright (c) 2008 HolgerRapp@gmx.net. All rights reserved.

from _dc1394core import *
from _dc1394core import _dll
from ctypes import *


from numpy import fromstring, ndarray
from threading import *

from Queue import Queue, Full

__all__ = [ "DC1394Library", "Camera", "SynchronizedCams" ]

class DC1394Library(object):
    """
    This wraps the dc1394 library object which is a nuisance to have
    around. This is bad design on behalve of DC1394. Oh well...  This
    object must stay valid untill all cameras are closed
    But then use it well: it not only opens the library, collects
    a reference to the library and the camera list.

    h:				a handler of the library, many functions require it

    cameralist:		a list of dicts. Each camera has a dict, containing:
            "unit":		the unit ID of the camera
            "guid":		GUID -> requested for the camera allocation
            "vendor":	vendor name
            "model" :	model name
                        Using this list the user may identify the camera and
                        pass cameralist[i]['guid'] to the Camera class below.

        close():	frees up the library
        enumerate_cameras():	reinitializes the camera list
    """
    def __init__( self ):
        # we cache the dll, so it gets not deleted before we cleanup
        self._dll = _dll
        self._h = _dll.dc1394_new()
        self.cameralist = self.enumerate_cameras()

    def __del__(self):
        self.close()

    @property
    def h(self):
        "The handle to the library context."
        return self._h

    ###################################################################
    # Functions:
    ###################################################################

    def close( self ):
        if self._h is not None:
            self._dll.dc1394_free( self._h )
        self._h = None
        self.cameralist = []

    def enumerate_cameras( self ):
        """
        Enumerate the cameras currently attached to the bus.

        returns a list of {'guid','vendor','model'}
        """
        l = POINTER(camera_list_t)()

        _dll.dc1394_camera_enumerate(self.h, byref(l))

        #cams -> clist renamed for using cam as a camera
        clist = []
        for i in xrange(l.contents.num):
            ids = l.contents.ids[i]
            #we can be nice to the users, providing some more
            #than mere GUID and unitIDs
            #also, if this fails, we have a problem:
            cam = self._dll.dc1394_camera_new(self.h, ids.guid)

            #it seems not all cameras have these fields:
            vendor = cam.contents.vendor if cam.contents.vendor else "unknown"
            model = cam.contents.model if cam.contents.model else "unknown"

            clist.append(
                { "unit": ids.unit,
                    # For what do we need this L anyway?!?!
                  #"guid": hex(ids.guid)[2:].strip("L"),
                  "guid": ids.guid,
                  "vendor":	vendor,
                  "model":	model,
                }
            )

            self._dll.dc1394_camera_free(cam)
        _dll.dc1394_camera_free_list(l)

        return clist


class Image(ndarray):
    """
    This class is a image returned by the camera. It is basically a
    numpy array with some additional informations (like timestamps).
    It is not based on the vide_frame structure of the dc1394, but rather
    augments the information from numpy through information of the acquisition
    of this image.
    """

    def __new__(cls, camera=None, frame=None): 
        """ Create an Image instance from a grabbed frame.
            This is a constructor needed to be a subclass of ndarray. Here
            we collect parameters only, no defaults.

            Parameters:
            cls:    subclass
            camera: camera instance
            frame:  image frame collected by the camera
        """
        fr = frame.contents
        #get the buffer from the frame (this is part of the ring buffer):
        Dtype = c_char*fr.image_bytes
        buf = Dtype.from_address(fr.image)

        size = fr.size

        if fr.data_depth == 8:
            dtype = 'u1'
        elif fr.data_depth == 16:
            if fr.little_endian:
                dtype = '<u2'
            else:
                dtype = '>u2'
            #end if endianness
        #end if depth

        #convert the data:
        img0 = fromstring(buf, dtype=dtype)
        img = ndarray.__new__(cls,shape=size, dtype=dtype,buffer=img0)

        #inherit the attributes:
        img.position = fr.position
        img.color_coding = fr.color_coding

        img.color_filter = fr.color_filter
        img.yuv_byte_order = fr.yuv_byte_order
        img.stride = fr.stride
        #img.video_mode dropped
        img.packet_size = fr.packet_size
        img.packets_per_frame = fr.packets_per_frame
        img.timestamp = fr.timestamp
        img.frames_behind = fr.frames_behind
        img.id = fr.id
        img.data_depth = fr.data_depth

        #the camera object used to capture this image:
        img._cam = camera
        return img
    #end __new__

    def __array_finalize__(self, img):
        """ Finalize the new Image class array:
            if called with an image object at the end, inherit 
            the properties of that image.
        """
        if img != None:
            self.position = getattr(img, 'position', -1)
            self.color_coding = getattr(img, 'color_coding', -1)

            self.color_filter = getattr( img, 'color_filter', -1)
            self.yuv_byte_order = getattr( img, 'yuv_byte_order', -1)
            self.stride = getattr( img, 'stride', -1)
            #self.video_mode dropped
            self.packet_size = getattr(img, 'packet_size', -1)
            self.packets_per_frame =  getattr( img, 'packets_per_frame', -1)
            self.timestamp = getattr( img, 'timestamp', -1)
            self.frames_behind = getattr(img, 'frames_behind', -1)
            self.id = getattr(img, 'id', -1)
            self.data_depth = getattr(img, 'data_depth', 8)
            self._cam = getattr(img, '_cam', None)
        #end if
    #end of __array_finalize_

    def __array_wrap__(self, res, context=None):
        """ to inherit the extra properties after an ufunc call """
        # ndarray.__array_wrap__ will call the __array_finalize__ to get
        #the attributes right
        return ndarray.__array_wrap__(self, res, context)
    #end of __array_wrap__

    def as_RGB(self):
        """ Convert the image to an RGB image. Returns an RGB image """
        if self._cam == None:
            print "Can not convert without the camera isntance"
            return None
        #end if

        res = ndarray( 3*self.size, dtype='u1')
        shape = self.shape
        inp = ndarray( shape=len(self.data), buffer=self.data, dtype='u1')

        self._cam._dll.dc1394_convert_to_RGB8( inp, res, 
                shape[1], shape[0], self.yuv_byte_order,
                self.color_coding, self.data_depth)
        
        res.shape = (shape[0],shape[1],3)

        return res
    #end of as_RGB
    
    def as_MONO8(self):
        """ Convert he image to gray scale 8 bit. Returns a new image """
        if self._cam == None:
            print "Conversion needs an opened camera instance"
            return None
        #end if

        res = ndarray( self.size, dtype='u1')
        shape = self.shape
        inp = ndarray( shape = len(self.data), buffer=self.data, dtype='u1')

        self._cam._dll.dc1394_convert_to_MONO8( inp, res,
                shape[1], shape[0], self.color_coding, self.yuv_byte_order,
                self.color_coding, self.data_depth)
        res.shape = shape

        return res
    #end of as_MONO8

    def as_YUV422(self):
        """ Converts an image to YUV422 format if someone needs that """
        if self._cam == None:
            print "Conversion needs a camera instance "
            return None

        res = ndarray( self.size, dtype='u1')
        shape = self.shape
        inp = ndarray( shape = len(self.data), buffer=self.data, dtype='u1')
        
        self._cam._dll.dc1394_convert_to_YUV422( inp, res,
                shape[1], shape[0], self.color_coding, self.yuv_byte_order,
                self.color_coding, self.data_depth)

        #I am not sure abuot this, it has to be tested:
        return ndarray(shape=shape, buffer=res.data, dtype='u2')

#end of Image class

class _CamAcquisitonThread(Thread):
    def __init__(self,cam, condition ):
        """
        This class is created and launched whenever a camera is start()ed.
        It continously acquires the pictures from the camera and sets a
        condition to inform other threads of the arrival of a new picture
        """
        Thread.__init__(self)

        self._cam = cam
        self._should_abort = False

        self._last_frame = None
        self._condition = condition

        #create a lock object from the threading module; see LockType.__doc__
        self._abortLock = Lock()

        self.start()

    def abort(self):
        self._abortLock.acquire()
        self._should_abort = True
        self._abortLock.release()

    def run(self):
        """
        Core function which contains the acquisition loop
        """

        while 1:
            self._abortLock.acquire()
            sa = self._should_abort
            self._abortLock.release()

            if sa:
                break

            if self._last_frame:
                self._cam._dll.dc1394_capture_enqueue(
                    self._cam._cam, self._last_frame
                )

            #the actual image acquisition:
            frame = POINTER(video_frame_t)()
            self._cam._dll.dc1394_capture_dequeue(self._cam._cam,
                CAPTURE_POLICY_WAIT, byref(frame)
            );

            #store the last frame:
            self._last_frame = frame

            self._condition.acquire()
            #generate an Image class from the frame
            #all copy/inheritance is handled there:
            img = Image(self._cam, frame)
            #one way of setting this:
            img.shape = self._cam._shape

            #img = fromstring(buf, dtype=self._cam._dtype).reshape(
            #    self._cam._shape
            #).view(Image)

            #img._position, img._packet_size, img._packets_per_frame, \
            #    img._timestamp, img._frames_behind, img._id = \
            #    frame.contents.position, frame.contents.packet_size, \
            #    frame.contents.packets_per_frame, frame.contents.timestamp, \
            #    frame.contents.frames_behind,frame.contents.id
            self._cam._current_img = img

            #is the camera streaming to a queue?
            if self._cam._queue:
                # Will throw an exception if you're to slow while processing
                self._cam._queue.put_nowait(img)

            self._condition.notifyAll()
            self._condition.release()

        # Return the last frame
        if self._last_frame:
            self._cam._dll.dc1394_capture_enqueue(
                self._cam._cam, self._last_frame
        )
        self._last_frame = None


##################################
# Camera Property implementation #
##################################
class CameraProperty(object):
    def __init__( self, cam, name, id, absolute_capable ):
        """This class implements a simple Property of the camera"""
        self._id = id
        self._name = name
        self._absolute_capable = absolute_capable
        self._dll = cam._dll
        self._cam = cam

    def val():
        doc = "The current value of this property"
        def fget(self):
            if self._name == "white_balance":
                #white has its own call since it returns 2 values
                blue = c_uint32()
                red = c_uint32()
                self._dll.dc1394_feature_whitebalance_get_value(
                    self._cam._cam, byref(blue), byref(red)
                )
                return (blue.value, red.value)

            if self._absolute_capable:
                val = c_float()
                self._dll.dc1394_feature_get_absolute_value(
                    self._cam._cam, self._id, byref(val)
                )

                if self._name == "shutter" :
                    # We want shutter in ms -> if it is absolute capable.
                    val.value *= 1000.

            else:
                val = c_uint32()
                self._dll.dc1394_feature_get_value(
                    self._cam._cam, self._id, byref(val)
                )
            return val.value

        def fset(self, value):
            if self._name == "white_balance":
                #white has its own call since it returns 2 values
                blue, red = value
                self._dll.dc1394_feature_whitebalance_set_value(
                    self._cam._cam, blue, red
                )
            else:
                if self._absolute_capable:
                    val = float(value)
                    # We want shutter in ms
                    if self._name == "shutter":
                        value /= 1000.
                    self._dll.dc1394_feature_set_absolute_value(
                        self._cam._cam, self._id, val
                    )
                else:
                    val = int(value)
                    self._dll.dc1394_feature_set_value(
                        self._cam._cam, self._id, val
                    )
        return locals()
    val = property(**val())

    @property
    def range(self):
        "The RO foo property."
        if self._absolute_capable:
            min, max = c_float(), c_float()
            self._dll.dc1394_feature_get_absolute_boundaries( self._cam._cam,\
                                                self._id, byref(min),byref(max))
            # We want shutter in ms
            if self._name == "shutter":
                min.value *=1000
                max.value *=1000
        else:
            min, max = c_uint32(), c_uint32()
            self._dll.dc1394_feature_get_boundaries(
                self._cam._cam, self._id, byref(min),byref(max)
            )
        return (min.value,max.value)

    @property
    def can_be_disabled(self):
        "Can this property be disabled"
        k = bool_t()
        self._dll.dc1394_feature_is_switchable(self._cam._cam,
                self._id, byref(k)
        )
        return bool(k.value)

    def on():
        doc = """\
        Toggle this feature on and off;
        For the trigger this means the external trigger ON/OFF"""

        def fget(self):
            k = bool_t()
            if self._name.lower() == "trigger":
                self._dll.dc1394_external_trigger_get_power(
                    self._cam._cam, byref(k)
                )
            else:
                self._dll.dc1394_feature_get_power(self._cam._cam, self._id,
                   byref(k)
                )
            return bool(k.value)

        def fset(self, value):
            k = bool(value)
            if self._name.lower() == "trigger":
                self._dll.dc1394_external_trigger_set_power(self._cam._cam, k)
            else :
                self._dll.dc1394_feature_set_power( self._cam._cam, self._id, k)
        return locals()
    on = property(**on())

    @property
    def pos_modes(self):
        "The possible control modes for this feature (auto,manual,...)"
        if self._name.lower() == "trigger":
            #we need a trick:
            finfo = feature_info_t()
            finfo.id = self._id
            _dll.dc1394_feature_get( self._cam._cam,  byref(finfo) )
            modes = finfo.trigger_modes

            return [ trigger_mode_vals[ modes.modes[i]] for
                    i in xrange(modes.num)
            ]
        modes = feature_modes_t()
        _dll.dc1394_feature_get_modes(self._cam._cam, self._id, byref(modes))
        return [ feature_mode_vals[modes.modes[i]] for i in xrange(modes.num) ]

    def mode():
        doc = """The current control mode this feature is running in.

                For the trigger it shows the trigger modes (from the dc1394
                website):
                mode 0:     Exposure starts with a falling edge and stops when
                            the the exposure specified by the SHUTTER feature
                            is elapsed.
                mode 1:     Exposure starts with a falling edge and stops with
                            the next rising edge.
                mode 2:     The camera starts the exposure at the first falling
                            edge and stops the integration at the nth falling
                            edge. The parameter n is a prameter of the trigger
                            that can be set with camera.trigger.val parameter.
                mode 3:     This is an internal trigger mode. The trigger is
                            generated every n*(period of fastest framerate).
                            Once again, the parameter n can be set with
                            camera.trigger.val.
                mode 4:     A multiple exposure mode. N exposures are performed
                            each time a falling edge is observed on the trigger
                            signal. Each exposure is as long as defined by the
                            SHUTTER (camera.shutter) feature.
                mode 5:     Another multiple exposure mode. Same as Mode 4
                            except that the exposure is is defined by the
                            length of the trigger pulse instead of the SHUTTER
                            feature.

                mode 14 and 15: vendor specified trigger mode.
            """
        def fget(self):
            if self._name.lower() == "trigger":
                mode = trigger_mode_t()
                _dll.dc1394_external_trigger_get_mode(
                    self._cam._cam, byref(mode)
                )
                return trigger_mode_vals[ mode.value ]

            mode = feature_mode_t()
            _dll.dc1394_feature_get_mode(self._cam._cam, self._id, byref(mode))
            return feature_mode_vals[mode.value]

        def fset(self, value):
            if value in self.pos_modes:
                if self._name.lower() == "trigger":
                    key = trigger_mode_codes[ value ]
                    _dll.dc1394_external_trigger_set_mode(self._cam._cam, key)
                else:
                    key = feature_mode_codes[ value ]
                    _dll.dc1394_feature_set_mode(self._cam._cam, self._id,key )
            else:
                    print "Invalid %s mode: %s" %(self._name, value)
        return locals()
    mode = property(**mode())

    def polarity_capable(self):
        """
        Is this feature polarity capable?  This is valid for the trigger
        only.
        """
        finfo = feature_info_t()
        finfo.id = self._id
        _dll.dc1394_feature_get( self._cam._cam,  byref(finfo) )
        #polarity_capable is an bool_t = int field:
        return bool( finfo.polarity_capable )

    def polarity():
        doc = """The polarity of the external trigger. If the trigger
                has polarity (camera.trigger.polarity_capable == True),
                then it has two possible values. These are returned by:
                camera.trigger.pos_polarities.
                """
        def fget(self):
            pol = trigger_polarity_t()
            _dll.dc1394_external_trigger_get_polarity(self._cam._cam,
                byref(pol)
            )
            if trigger_polarity_vals.has_key( pol.value ):
                return trigger_polarity_vals[ pol.value ]
            else :
                return pol.value

        def fset(self, pol):
            if self.polarity_capable:
                if trigger_polarity_codes.has_key( pol ):
                    key = trigger_polarity_codes[ pol ]
                    _dll.dc1394_external_trigger_set_polarity(
                        self._cam._cam, key
                    )
                else:
                    print "Invalid external trigger polarity: %s" %pol
        return locals()
    polarity = property( **polarity())

    def pos_polarities(self):
        return trigger_polarity_codes.keys()

    def source():
        doc = "Actual source of the external trigger"
        def fget(self):
            source = trigger_source_t()
            _dll.dc1394_external_trigger_get_source(self._cam._cam,
                byref(source)
            )
            return trigger_source_vals[ source.value ]

        def fset(self, source):
            if trigger_source_codes.has_key( source ):
                key = trigger_source_codes[ source ]
                _dll.dc1394_external_trigger_set_source(self._cam._cam, key)
            else:
                print "Invalid external trigger source: %s" %source
        return locals()
    source = property( **source())

    def pos_sources(self):
        """ List the possible external trigger sources of the camera"""
        src = trigger_sources_t()
        _dll.dc1394_external_trigger_get_supported_sources(self._cam._cam,
            byref(src)
        )
        return [ trigger_source_vals[src.sources[i]] for i in xrange(src.num) ]

    def software_trigger():
        doc = "Set and get the software trigger (active or not)."
        def fget(self):
            res = switch_t()
            _dll.dc1394_software_trigger_get_power(self._cam._cam, byref(res))
            return bool( res.value )
        def fset(self, value):
            k = bool(value)
            _dll.dc1394_software_trigger_set_power(self._cam._cam, k)
        return locals()
    software_trigger = property( **software_trigger() )



class Camera(object):
    def __init__( self, lib, guid, mode = None,
                 framerate = None, isospeed = 400, **feat):
        """
        This class represents a IEEE1394 Camera on the BUS. It currently
        supports all features of the cameras except white balancing.

        You can pass all features the camera supports as additional arguments
        to this classes constructor.  For example: shutter = 7.4, gain = 8

        The cameras pictures can be accessed in two ways. Either way, use
        start() to beginn the capture.  If you are always interested in the
        latest picture use the new_image Condition, wait for it, then use
        cam.current_image for your processing. This mode is called interactive
        because it is used in live displays.  An alternative way is to use
        shot() which gurantees to deliver all pictures the camera acquires
        in the correct order. Note though that you have to process these
        pictures with a certain speed, otherwise the caching queue will
        overrun. This mode is called serial. Note that you can theoretically
        also use the first acquisition mode here, but this seldom makes
        sense since you need a processing of the pictures anyway.

        lib       - the DC1394Library object is needed to open a camera
        guid      - GUID of this camera, can be a hexstring or the integer
                    value
        mode      - acquisition mode, e.g. (640, 480, "Y8"). If you pass None,
                    the current mode is kept. One can also use a string, such
                    as 'FORMAT7_0'

        framerate - wanted framerate, if you pass None, the current camera
                    setting is kept

        isospeed  - wanted isospeed, you might want to use 800 if your bus
                    supports it
        """
        self._lib = lib
        if isinstance(guid,basestring):
            guid = int(guid,16)

        self._guid = guid
        self._cam = None
        #it is the same as _dll anyway...
        self._dll = lib._dll

        self._running = False
        self._running_lock = Lock()

        # For image acquisition
        self._new_image = Condition()
        self._current_img = None
        self._worker = None
        #a default numeric image type
        self._dtype = '<u1'

        self.open()

        # Gather some informations about this camera
        # Will also set the properties accordingly
        self._all_features = self.__get_all_features()
        self._all_modes = self.__get_supported_modes()

        # Set all features to manual (if possible)
        for f in self._all_features:
            if 'manual' in self.__getattribute__(f).pos_modes:
                self.__getattribute__(f).mode = 'manual'

        # Set acquisition mode and framerate
        #self.mode = mode if mode is not None else self._all_modes[0]
        #do not change mode if not needed, but set the self._wanted_mode
        #variable to the code of the current mode.
        self.mode = tuple(mode) if mode is not None else self.mode

        #set the framerate:
        #does this work if mode is set to a FORMAT7 mode?
        #now it does, it does nothing:
        if 'FORMAT7' not in self.mode[-1].upper():
            #it sets the framerate if it is available, or nothing
            self.fps = framerate = framerate or \
                    self.get_framerates_for_mode(self.mode)[0]

        # If we are not using a FORMAT_7 format, set the framerate feature to
        # auto again. This control is not available on all cameras, if it is
        # missing, the framerate is only controllable by the current mode

        try:
            self.framerate.mode = "auto"
        except AttributeError:
            pass # Feature not around, so what?

        # Set isospeed
        if isospeed :
            # If the speed is >= 800, set other operation mode
            #this is done automatically by the isospeed setting
            # self._operation_mode = "legacy" if isospeed < 800 else "1394b"
            self.isospeed = isospeed

        # Set other parameters
        for n,v in feat.items():
            if v is None:
                continue
            self.__getattribute__(n).val = v

    def __del__(self):
        self.close()

    def start( self, bufsize = 4, interactive = False ):
        """
        Start the camera in free running acquisition

        bufsize     - how many DMA buffers should be used? If this value is
                      high, the lag between your currently processed picture
                      and reality might be higher but your risk to miss a frame
                      is also much lower.
        interactive - If this is true, shot() is not supported and no queue
                      overrun can occure
        """
        if self.running:
            return

        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        # Set video mode and everything: done by the actual fsets in
        #self.mode, self.isospeed and self.fps
        #self._dll.dc1394_video_set_iso_speed( self._cam, self._wanted_speed )
        #self._dll.dc1394_video_set_mode( self._cam, self._wanted_mode )
        #self._dll.dc1394_video_set_framerate( self._cam, self._wanted_frate )
        self._dll.dc1394_capture_setup( self._cam, bufsize, \
                    capture_flag_codes["CAPTURE_FLAGS_DEFAULT"])

        # Start the acquisition
        self._dll.dc1394_video_set_transmission( self._cam, 1 )

        self._queue = None if interactive else Queue(1000)

        # Now, start the Worker thread
        self._worker = _CamAcquisitonThread( self, self.new_image )

        self._running_lock.acquire()
        self._running = True
        self._running_lock.release()

    def stop( self ):
        """Stop the camera and return all frames to the driver"""

        if not self.running:
            return

        assert(self._cam) # Otherwise it couldn't be running
        self._worker.abort()
        self._worker.join()

        self._queue = None

        #stop the camera:
        self._dll.dc1394_capture_stop( self._cam )
        self._dll.dc1394_video_set_transmission( self._cam, 0 )

        #stop the thread:
        self._running_lock.acquire()
        self._running = False
        self._running_lock.release()

    def reset_bus( self ):
        """
        This function resets the bus the camera is attached to. Note that
        this means that all cameras have to reenumerate and will drop frames.
        So only use this if you know what you are doing.
        """
        if self.running:
            self.stop()

        self._dll.dc1394_reset_bus( self._cam )

    def shot( self ):
        """
        If the camera is running, this will acquire one frame from it and
        return it as a Image (numpy array + some informations).The memory
        is not copied, therefore you should not write on the array.

        Note that acquisition is always running in the background. This
        function alone is guaranteed to return all frames in running order.
        Use this function for your image processing, use cam.current_image
        for visualisation.
        """
        if not self.running:
            raise RuntimeError, "Camera is not running!"
        if not self._queue:
            raise RuntimeError, "Camera is running in interactive mode!"

        return self._queue.get()

    def open( self ):
        """Open the camera"""
        self._cam = _dll.dc1394_camera_new( self._lib.h, self._guid )
        if not self._cam:
            raise RuntimeError, "Couldn't access camera!"

    def close(self):
        """Close the camera. Stops it, if it was running"""
        if self.running:
            self.stop()

        if self._cam:
            _dll.dc1394_camera_free( self._cam )
            self._cam = None

    def program_format7_mode(self, slot, offset = (0,0), mode = (640,480,"Y8")):
        """
        Program a given Format 7 slot (0 = FORMAT7_0) with the given
        parameters.  The package size is always the maximum available. The
        framerate can then only be controlled through the framerate property
        (if available).  This also implicitly sets the correct mode for
        format 7.

        If you change the mode to a normal one and want your format 7 mode
        again, you have to recall this function. The behaviour otherwise
        is undefined.

        slot   - 0,1,2 FORMAT7 Slot to program
        offset - picture offset (for ROI)
        mode   - Resolution and data depth is extracted. A valid mode would
                 be (121,99,"RGB")
        """
        if self.running:
            raise RuntimeError(
                "Can't set Format7 mode while camera is running!")

        newmode = "FORMAT7_%i" % slot
        if newmode in video_mode_codes:
            uslot = video_mode_codes[ newmode ]

        else :
            print "Available modes:"
            print filter( lambda x: "FORMAT7" in x, video_mode_codes.keys())

            return

        #color coding:
        cco = color_coding_t()

        #What is set for the camera?
        self._dll.dc1394_format7_get_color_coding( self._cam,\
                uslot, byref(cco) )
        #if this is different, try to set it first:
        if color_coding_vals[cco.value] != mode[-1] :
            #what is available in this mode?
            pos_codings = color_codings_t()
            self._dll.dc1394_format7_get_color_codings( self._cam,\
                    uslot, byref(pos_codings))
            #convert it to a list:
            codelist = []
            for i in xrange(pos_codings.num):
                codelist.append( color_coding_vals[ pos_codings.codings[i] ])
            if mode[-1] in codelist:
                cco2 = color_coding_codes[ mode[-1] ]
                self._dll.dc1394_format7_set_color_coding( self._cam,\
                    uslot, cco2)
            else :
                print "available modes:"
                print codelist
                return

        #this invokes the standard mode setting part, which will
        #set FORMAT7 modes. The python variable shall show a 0,0 resolution:
        #that one we set by hand:
        self.mode = newmode

        #the format7 position and image size can only be set in units.
        #1. check for these units:
        h_unit = c_uint32(); v_unit = c_uint32()

        self._dll.dc1394_format7_get_unit_position(self._cam,\
                self._wanted_mode,\
                byref(h_unit), byref(v_unit))

        #Tupple does not support elementwise value changing, so do it dirty:
        offset = list(offset)
        offset[0] = h_unit.value*int(offset[0]/h_unit.value) if \
                h_unit.value > 1 else offset[0]
        offset[1] = v_unit.value*int(offset[1]/v_unit.value) if \
                v_unit.value > 1 else offset[1]

        self._dll.dc1394_format7_get_unit_size( self._cam,\
                self._wanted_mode,\
                byref(h_unit), byref(v_unit))

        mode = list(mode)
        mode[0] = h_unit.value *int(mode[0]/h_unit.value) if \
                h_unit.value > 1 else mode[0]
        mode[1] = v_unit.value*int(mode[1]/v_unit.value) if \
                v_unit.value > 1 else mode[1]

        #now the ROI should set without a problem (except out of size range):
        self._dll.dc1394_format7_set_roi(
            self._cam, uslot,
            cco, USE_MAX_AVAIL,
            offset[0], offset[1],
            mode[0],mode[1])

        # But directly overwrite the shape (the last number is 3 for RGB modes)
        self._shape[0:2] = [ mode[1], mode[0] ]

    ###########################################################################
    #                     INFORMATION GATERHING FUNCTIONS                     #
    ###########################################################################
    def __get_supported_modes( self ):
        """
        Get all the supported video modes of the camera.  This calls the
        builting dc1394 function and converts the returned codes to a
        readable list. Any element of this list can be used to set a video
        mode of the camera.

        Parameters: None
        Returns:    list of available video modes
        """
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        modes = video_modes_t()
        supmodes = []

        _dll.dc1394_video_get_supported_modes( self._cam, byref(modes))

        for i in range(modes.num):
            supmodes.append( video_mode_vals[modes.modes[i]] )

        return supmodes

    def __get_all_features( self ):
        """
        Use a built in dc1394 function to read out all available features
        of the given camera.

        All features, which are capable of absolute values, are set to
        absolute value mode.

        Parameters: None Return value: fills up and returns the self._features
        list
        """

        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        fs = featureset_t()
        _dll.dc1394_feature_get_all( self._cam, byref(fs) )


        self._features = []

        # We set all features that are capabale of it to absolute values
        for i in range(FEATURE_NUM):
            s = fs.feature[i]
            if s.available:
                if s.absolute_capable:
                    _dll.dc1394_feature_set_absolute_control(self._cam, s.id, 1)
                name = feature_vals[s.id]
                self._features.append( name )
                self.__dict__[name] = CameraProperty(self, name, s.id,
                    s.absolute_capable
                )

        return self._features

    def get_register( self, offset ):
        """Get the control register value of the camera a the given offset"""
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        val = c_uint32()
        _dll.dc1394_get_control_registers( self._cam, offset, byref(val), 1)
        return val.value

    def set_register( self, offset, value ):
        """Set the control register value of the camera at the given offset to
        the given value"""
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        val = c_uint32(value)
        _dll.dc1394_set_control_registers( self._cam, offset, byref(val), 1)

    ###########################################################################
    #                               PROPERTIES                                #
    ###########################################################################
    def broadcast():
        doc = \
        """
        This sets if the camera tries to synchronize with other cameras on
        the bus.

        Note:   that behaviour might be strange if one camera tries to
                broadcast and another not.

        Note 2: that this feature is currently only supported under linux
                and I have not seen it working yet though I tried it with
                cameras that should support it. So use on your own risk!
        """
        def fget(self):
            if not self._cam:
                raise RuntimeError, "The camera is not opened!"

            k = bool_t()
            self._dll.dc1394_camera_get_broadcast( self._cam, byref(k))
            if k.value == 1:
                return True
            else:
                return False

        def fset(self, value):
            if not self._cam:
                raise RuntimeError, "The camera is not opened!"

            use =  1 if value else 0
            self._dll.dc1394_camera_set_broadcast( self._cam, use )

        return locals()
    broadcast = property(**broadcast())

    @property
    def current_image(self):
        "Threadsafe access to the current image of the camera"
        # We do proper locking
        self._new_image.acquire()
        self._new_image.wait()
        i = self._current_img
        self._new_image.release()
        return i

    @property
    def new_image(self):
        "The Condition to wait for when you want a new Image"
        return self._new_image

    @property
    def numpy_shape(self):
        """
        This returns the shape the camera currently delivers. It is useful
        if you want to create a matching numpy array
        """
        return self._shape

    @property
    def numpy_dtype(self):
        """
        This returns the datatype the camera currently delivers. It is useful
        if you want to create a matching numpy array
        """
        return self._dtype

    @property
    def running(self):
        """
        This is a thread safe propertie which can check
        if the camera is (still) running
        """
        self._running_lock.acquire()
        rv = self._running
        self._running_lock.release()
        return rv

    @property
    def model(self):
        "The name of this camera (string)"
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        return self._cam.contents.model

    @property
    def guid(self):
        "The Guid of this camera as string"
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        return hex(self._cam.contents.guid)[2:-1]

    @property
    def vendor(self):
        "The vendor of this camera (string)"
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        return self._cam.contents.vendor

    def mode():
        doc = \
            """ The actual video mode of the camera.
                (one can query as a variable or set it.
                One can set a tupple (hsize, vsize, 'color_coding') or a
                simple text '640x480_RGB8'. Or simply one of the modes,
                such as: modes[0].

                If FORMAT7 mode is selected, it sets to the camera to maximal
                resolution. This deletes the previous ROI.

                When queried, it returns the actual resolution in a tupple:
                (hsize, vsize, "color_mode")
            """
        def fget(self):
            vmod = video_mode_t()
            self._dll.dc1394_video_get_mode( self._cam, byref(vmod))
            #set self._wanted_mode to store the code of the mode (internal use):
            self._wanted_mode = vmod.value
            #the video_mode_details have all supported keys:
            mode = video_mode_details[ vmod.value ]

            #format7 modes: the resolution depends on the camera:
            if "FORMAT7" in mode[-1]:
                hsize = c_uint32(); vsize= c_uint32()

                self._dll.dc1394_format7_get_image_size(self._cam, \
                        self._wanted_mode, \
                        byref(hsize), byref(vsize))

                return (int(hsize.value), int(vsize.value),\
                            video_mode_details[self._wanted_mode][-1])
            else:
                return video_mode_details[self._wanted_mode]

        def fset(self, mode):
            if self.running:
                raise RuntimeError, "Can't change mode while camera is running!"

            if not self._cam:
                raise RuntimeError, "The camera is not opened!"

            #one can pass here a string like camera.modes[0]
            #or a full tuple (640,480,'Y8')...
            #convert it to a valid text key:
            if mode.__class__.__name__ != 'str' :
                if "FORMAT7" not in mode[-1].upper():
                    mode = "%dx%d_%s" %(mode[0],mode[1],mode[2])
                else:
                    #a format7 mode is only the mode name:
                    mode = mode[-1]
            #...so we can search in the video_mode_codes

            if mode not in self.modes:
                raise ValueError, "This mode is not supported by this camera!"

            #if it is a supported mode, then the video_mode_codes has this key:
            #keep self._wanted_mode for historical reasons (and comfort)
            self._wanted_mode = video_mode_codes[ mode ]
            #now set the mode in the camera:
            self._dll.dc1394_video_set_mode( self._cam, self._wanted_mode)

            #now we have the code, we can set the mode back to provide
            #a return value:
            mode = video_mode_details[ self._wanted_mode ]
            #from now mode is a tupple...
            if "FORMAT7" in mode[-1].upper():
                hsize = c_uint32()
                vsize = c_uint32()
                self._dll.dc1394_format7_get_max_image_size(self._cam, \
                        self._wanted_mode, \
                        byref(hsize), byref(vsize))

                cc = color_coding_t()
                self._dll.dc1394_format7_get_color_coding( self._cam, \
                        self._wanted_mode,\
                        byref(cc))

                #first set the image position to (0,0):
                self._dll.dc1394_format7_set_image_position( self._cam,\
                        self._wanted_mode,\
                        c_uint32(0), c_uint32(0))

                #now set image size to the maximum
                self._dll.dc1394_format7_set_image_size( self._cam,\
                        self._wanted_mode,\
                        hsize, vsize)

                #delegate this to the shape of the image:
                #an array is indexed vertically then horizontally
                self._shape = [ int(vsize.value) , int(hsize.value) ]

                #a descriptive text of color coding:
                ccval = color_coding_vals[ cc.value ]

            else:
                #just to have one consistent variable for
                #identifying the color coding and dtype:
                ccval = mode[-1]
                self._shape = [ mode[1], mode[0] ]

            if '8' in ccval:
                self._dtype = '>u1'
            elif '16' in ccval:
                self._dtype = '>u2'
            elif 'YUV' in ccval:
                print "Warning: YUV image format!"
                #the data depth is 8 bit in the buffer,
                #but 12 or 16 bit in a color pixel.
                self._dtype = ">u1"
            else:
                print "Nonstandard image format: %s" %mode[-1]
                self._dtype = ">u1"

            #RGB can be both 8 and 16 bit. So let us check that
            #in a nice way:
            if "RGB" in ccval:
                self._shape.append(3)

        return locals()
    mode = property(**mode())

    def fps():
        doc = \
            """ The framerate belonging to the current camera mode.
                For not scalable video formats there is a set of
                standard frame rates one can choose from.

                In FORMAT7 format the camera can tell an actual
                value, but one can not set it.
                Unfortunately the returned framerate may have no sense at all.

                If a value is set, it is tested against the supported
                values and then set.
            """
        def fget(self):
            if 'FORMAT7' not in self.mode[-1].upper():
                ft = framerate_t()
                self._dll.dc1394_video_get_framerate( self._cam, byref(ft))
                return framerate_vals[ ft.value ]

            else:
                fi = c_float()
                self._dll.dc1394_format7_get_frame_interval(self._cam,\
                        self._wanted_mode,
                        byref(fi))
                #this should be corrected:
                return 1.0/fi.value

        def fset(self, framerate):
            if not self._cam:
                raise RuntimeError("The camera is not opened!")

            if self.running:
                raise RuntimeError(
                    "Can't change framerate while camera is running!"
                )

            if "FORMAT7" in self.mode[-1].upper():
                print "Framerate can not be set"
            else:
                #is it supported in this mode?
                if framerate in self.get_framerates_for_mode():
                    wanted_frate = framerate_codes[ framerate ]
                    #then set it:
                    self._dll.dc1394_video_set_framerate(
                        self._cam, wanted_frate
                    )
        return locals()
    fps = property(**fps())

    def isospeed():
        doc = \
            """ The isospeed of the camera.

                If queried, returns the actual isospeed value.
                If set, it tries setting the speed.

                One can get the actual set value of the camera or set from:
                100, 200, 400, 800, 1600, 3200 if the camera supports them.

                Above 400 the the 1394b, high speed mode has to be available
                (the funtion tries to set it).
            """
        def fget(self):
            sp = speed_t()
            self._dll.dc1394_video_get_iso_speed( self._cam, byref(sp) )
            return speed_vals[ sp.value ]

        def fset(self, isospeed):
            if speed_codes.has_key( isospeed ):
                try:
                    self._operation_mode = 'legacy' if isospeed < 800 \
                        else '1394b'
                except RuntimeError:
                    print "high speed ISO mode is not supported"
                else:
                    sp = speed_codes[ isospeed ]
                    self._dll.dc1394_video_set_iso_speed(self._cam, sp )
        return locals()
    isospeed = property( **isospeed() )

    def _operation_mode():
        doc = \
        """
        This can toggle the camera mode into B mode (high speeds). This is
        a private property because you definitively do not want to change
        this as a user.
        """
        def fget(self):
            if not self._cam:
                raise RuntimeError, "The camera is not opened!"

            k = c_int()
            self._dll.dc1394_video_get_operation_mode( self._cam, byref(k))
            if k.value == 480:
                return "legacy"
            else:
                return "1394b"
        def fset(self, value):
            if not self._cam:
                raise RuntimeError, "The camera is not opened!"

            use =  480 if value == "legacy" else 481
            self._dll.dc1394_video_set_operation_mode( self._cam, use )
        return locals()
    _operation_mode = property(**_operation_mode())

    @property
    def modes(self):
        "Return all supported modes for this camera"
        return self._all_modes

    def get_framerates_for_mode(self, mode = None):
        """ Returns all framerates supported for the given mode; if mode is
            None the current mode will be used.
        """
        if not self._cam:
            raise RuntimeError, "The camera is not opened!"

        use_modes = mode or self._wanted_mode

        if 'FORMAT7' in self.mode[-1].upper():
            print "Framerates are not valid for scalable mode"
            return [ self.fps ]

        fpss = framerates_t()
        _dll.dc1394_video_get_supported_framerates(self._cam, self._wanted_mode,
                        byref(fpss))
        return tuple( framerate_vals[fpss.framerates[i]]
             for i in range( fpss.num ) )


    @property
    def features(self):
        """Return all features of this camera. You can use __getattr__ to
        directly access them then."""
        return self._all_features


class SynchronizedCams(object):
    def __init__(self, cam0,cam1):
        """This class synchronizes two (not more!) cameras
        by dropping frames from one until the timestamps
        of the acquired pictures are in sync. Make sure that the
        cameras are in the same mode (framerate, shutter)

        This function assumes point gray cameras which can do autosync
        """
        self._cam0 = cam0
        self._cam1 = cam1

    def close( self ):
        "Convenient function which closes both cams"
        self._cam0.close()
        self._cam1.close()

    @property
    def cam0(self):
        return self._cam0
    @property
    def cam1(self):
        return self._cam1

    def start(self, buffers = 4):
        self._cam0.start(buffers); self._cam1.start(buffers)

        self.sync()

    def stop(self):
        self._cam0.stop(); self._cam1.stop()

    def shot(self):
        """
        This function acquires two synchronized pictures from
        the cameras. Use this if you need pictures which were
        acquired around the same time. Do not use the cams individual shot
        functions. If you need a current image you can use cam.current_image
        at all times. You can also wait for the Condition cam.new_image
        and then use cam.current_image.

        note that the user has to check for themselves if the cameras
        are out of sync and must make sure they get back in sync.
        """
        i1 = self._cam0.shot()
        i2 = self._cam1.shot()

        return i1,i2


    def sync(self):
        """
        Try to sync the two cameras to each other. This will
        only work if both cameras synchronize on the bus time
        """
        ldiff = 100000000
        while 1:
            t1 = self._cam0.shot().timestamp
            t2 = self._cam1.shot().timestamp

            diff = t1-t2

            if abs(diff) < 500:
                break
            if diff < 0:
                self._cam0.shot()
            else:
                self._cam1.shot()
            ldiff = diff

