#!/usr/bin/python

from pydc1394.camera2 import Camera
import time

def setup(cam, mode, gain, bright, shutter):
    cam.mode = cam.modes[mode]
    #cam.mode = cam.modes[-1]
    #cam.mode.setup((100,100), (200,200), "Y16")
    cam.trigger.active = True
    cam.trigger.mode = "14"
    cam.trigger.source = "0"
    cam.rate = max(cam.mode.rates)
    cam.set_register(0x1028, 2<<16) # extended shutter
    cam.framerate.mode = "manual"
    cam.framerate.absolute = max(cam.framerate.absolute_range)
    cam.framerate.active = False
    cam.shutter.active = True
    cam.shutter.mode = "manual"
    sr = cam.shutter.absolute_range
    cam.shutter.absolute = min(max(sr[0], shutter), sr[1])
    cam.exposure.active = False
    cam.gamma.active = False
    cam.gain.active = True
    cam.gain.mode = "manual"
    cam.gain.absolute = gain
    cam.brightness.active = True
    cam.brightness.mode = "manual"
    cam.brightness.absolute = bright
    
    cam[0x11f8] |= 0x7<<28 # gpio1-3 as out
    cam.set_register(0x1104, 0xc0000000) # strobe at integrate and queue
    cam.set_strobe(0x204, 0x83000000) # gpio1 strobe 0+integrate
    cam.set_strobe(0x208, 0x83200400) # gpio2 strobe 500u+1ms
    cam.set_strobe(0x20c, 0x83300600)


def capture(cam, n):
    cam.start_capture()
    cam.flush()
    cam.start_multi_shot(n)
    t = time.time()
    for i in range(n):
        im = cam.capture(mark_corrupt=True)
        print i, (time.time()-t)/(i+1), im.frames_behind, im.id
    cam.stop_multi_shot()
    cam.stop_capture()

def main():
    from optparse import OptionParser
    p = OptionParser()
    # p.add_option()
    p.set_defaults(guid=None)
    o,a = p.parse_args()
    cam = Camera(guid=o.guid)
    cam.power(True)
    setup(cam, mode=3, gain=0, bright=0, shutter=0.12)
    capture(cam, 20)
    cam.power(False)

if __name__ == "__main__":
    main()
