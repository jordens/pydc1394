#!/usr/bin/python

from __future__ import (print_function, unicode_literals, division,
        absolute_import)


from pydc1394 import Camera
import time


def setup(cam, mode, gain, bright, shutter):
    cam.mode = cam.modes[mode]
    #cam.mode = cam.modes[-1]
    #cam.mode.setup((100,100), (200,200), "Y16")
    cam.rate = max(cam.mode.rates)
    cam.trigger.active = False
    #cam.trigger.mode = "14"
    #cam.trigger.source = "0"
    cam.set_register(0x1028, 2<<16) # extended shutter
    sr = cam.shutter.absolute_range
    cam.shutter.setup(min(max(sr[0], shutter), sr[1]))
    cam.framerate.setup(max(cam.framerate.absolute_range))
    cam.exposure.active = False
    cam.gamma.active = False
    cam.gain.setup(gain)
    cam.brightness.setup(bright)
    
    cam[0x11f8] |= 0x7<<28 # gpio1-3 as out
    cam[0x1104] = 0xc0000000 # strobe at integrate and queue
    cam.set_strobe(0x204, 0x83000000) # gpio1 strobe 0+integrate
    cam.set_strobe(0x208, 0x83200400) # gpio2 strobe 500u+1ms
    cam.set_strobe(0x20c, 0x83300600)


def capture(cam, n):
    cam.start_capture()
    cam.flush()
    cam.start_multi_shot(n+1)
    t = time.time()
    for i in range(n):
        im = cam.dequeue()
        print(i, (time.time()-t)/(i+1), im.frames_behind, im.frame_id)
        im.enqueue()
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
    setup(cam, mode=2, gain=0, bright=0, shutter=0.01)
    capture(cam, 10)
    cam.power(False)

if __name__ == "__main__":
    main()
