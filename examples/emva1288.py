#!/usr/bin/python

from pydc1394.camera2 import Camera, DC1394Error
from struct import pack, unpack, calcsize
import time, select
import numpy as np
import pylab as pl
from qo.uncertain import Uncertain

def info(cam):
    print "vendor: %s" % cam.vendor
    print "model: %s" % cam.model
    print "guid: %x" % cam.guid

    for f in cam.features:
        feat = getattr(cam, f)
        if feat.absolute_capable:
            v = feat.absolute
        else:
            v = feat.value
        print "feature %s: %s" % (f, v)

    m0 = cam.mode
    for m in cam.modes:
        cam.mode = m
        try:
            r = cam.mode.rates
            cam.rate = max(r)
        except DC1394Error:
            r = None
        print "mode %s: %s (%g, %g)" % (m, r,
                cam.framerate.absolute_range[0],
                cam.framerate.absolute_range[1])
    cam.mode = m0

    temp = cam.get_register(0x82c)
    print "temp: avail %s, val %sC" % (temp>>31,
            (temp&0xfff)*.1-273.2)
    
    print "xmit fail: %s" % cam.get_register(0x12fc)

    gpio = cam.get_register(0x1100)
    print "gpio: avail %s, state %s" % ((gpio>>16)&0xf, bin(gpio&0xf))

    for i,r in enumerate(range(0x1110, 0x114b, 0x10)):
        ctrl = cam.get_register(r)
        print "gpio%i: av %s, mode %s, data %s," % (i,
                ctrl>>31, ctrl>>16&0xf, bin(ctrl&0xff)),
        xtra = cam.get_register(r+0x4)
        print "xtra1: %s, xtra2: %s," % (bin(xtra>>16&0xff),
                bin(xtra&0xff)),
        mask = cam.get_register(r+0x8)
        print "mask av: %s, en: %s" % ((mask>>31)&0x1,
                bin((mask>16)&0xff))

    gpio_x = cam.get_register(0x1104)
    print "gpio_xtra: strobe at integration: %s, queue trigger: %s" % (
            bool(gpio_x>>31), bool(gpio_x>>30&1))

    pioo = cam.get_register(0x11f0)
    pioi = cam.get_register(0x11f4)
    piod = cam.get_register(0x11f8)
    print "pio: out %s, in %s, dir %s" % (bin((pioo>>28)&0xf),
                bin((pioi>>28)&0xf), bin((piod>>28)&0xf))

    s = cam.get_strobe(0x0)
    print "strobe: presence %s" % bin(s>>28)
    for i, r in enumerate(range(0, 0x10, 0x4)):
        if s & (1<<(31-i)):
            st = cam.get_strobe(0x100+r)
            print "strobe%s av: %s, read: %s, switch: %s, pol: %s, "\
                "min: %s, max: %s" % (i, bool(st>>31), bool(st>>27&1), 
                        bool(st>>26&1), bool(st>>25&1),
                        st>>12&0xfff, st&0xfff)
            mod = cam.get_strobe(0x200+r)
            print "strobe%s av: %s, on: %s, pol: %s, dly: %s, dur: %s" % (
                    i, bool(mod>>31), bool(mod>>25&1), bool(mod>>24&1),
                    mod>>12&0xfff, mod&0xfff)

    up = cam.get_register(0x12e0)
    reset = cam.get_register(0x12e4)
    print "up %ss, reset %ss" % (up, reset)

    finfo_fields = ["timestamp", "gain", "shutter", "brightness",
        "exposure", "whitebalance", "framecount", "strobecount", "gpio",
        "roi"]
    finfo = cam.get_register(0x12f8)
    print "frameinfo: avail %s, %s (%s)" % (finfo>>31,
            bin(finfo&0x3ff), ", ".join(
        v for i,v in enumerate(finfo_fields) if finfo & (1<<i)))

    led = cam.get_register(0x1a14)
    print "led: avail %s, state %s" % (led>>31, bool(led&0xff))

    voltlo = cam.get_register(0x1a50)
    volthi = cam.get_register(0x1a54)
    print "volt: avail %s, number %s" % (voltlo>>31,
            bin((voltlo>>12)&0xfff))

    defect = cam.get_register(0x1a60)
    print "pixel defects: avail %s, on %s, max %s, cur %s" % (
            defect>>31, (defect>>25)&1, (defect>>12)&0xfff,
            (defect&0xfff))

    #cam[0x1098] |= (1<<25)
    auto_shutter_range = cam.get_register(0x1098)
    print "min dark noise: avail %s, on %s, min %s, max %s" % (
            auto_shutter_range>>31, (auto_shutter_range>>25)&1,
            (auto_shutter_range>>12)&0xfff, (auto_shutter_range&0xfff))

    f7 = cam.get_register(0x1ac8)
    print "format7: avail %s, bin %s by %s" % (f7>>31,
            ((f7>>20)&0xf)+1, ((f7>>16)&0xf)+1)

    pxclk = cam.get_register(0x1af0)
    print "pxlclock: %s MHz" % (unpack("f", pack("I", pxclk))[0]/1e6)
    hclk = cam.get_register(0x1af4)
    print "hclock: %s kHz" % (unpack("f", pack("I", hclk))[0]/1e3)

    print "msglog: %s" % `"".join([
            unpack("4s", pack("!I", cam.get_register(r)))[0]
            for r in range(0x1d00, 0x1e00, 0x4)][::-1])`

    print "serial: %s" % cam.get_register(0x1f20)

    board = cam.get_register(0x1f24)
    print "board: %s, rev %s" % (hex((board>>20)&0xfff), (board>>16)&0xf)

    sensor = cam.get_register(0x1f28)
    print "sensor: typ1 %s, rev %s, typ2 %s" % (hex(sensor>>20),
            (sensor>>16)&0xf, hex(sensor&0xf))

    built = cam.get_register(0x1f40)
    print "built %s" % time.ctime(built)

    fw = cam.get_register(0x1f60)
    print "firmware %s.%s (typ %s) rev %s" % (fw>>24, (fw>>16)&0xff,
            (fw>>20)&0xf, fw&0xfff)

    built = cam.get_register(0x1f64)
    print "firmware built %s" % time.ctime(built)

    print "firmware desc: %s" % `"".join(
            unpack("4s", pack("!I", cam.get_register(r)))[0]
            for r in range(0x1f68, 0x1f80, 0x4))`

def capture(cam, n, crop):
    cam.start_capture()
    cam.flush()
    cam.start_multi_shot(n+2) # one for corruption safety, one for dump
    cam.dequeue().enqueue() # dump this
    ims = []
    while len(ims) < n:
        im = cam.dequeue()
        if not im.corrupt:
            ims.append(im[crop:-crop, crop:-crop].copy())
        im.enqueue()
    cam.stop_multi_shot()
    cam.stop_capture()
    return np.array(ims).astype("double")

def noise_mean(ims):
    n = ims.shape[0]
    mean = ims.mean(axis=0)
    noise = .5*((ims[1:]-ims[:-1])**2).mean(axis=0)
    return mean.ravel(), noise.ravel()

def linear(x, y):
    slope = (x*y).sum()/(x**2).sum()
    slope_var = ((y-slope*x)**2).sum()/(len(x)-1)/(x**2).sum()
    return Uncertain(slope, slope_var**.5)

def affine(x, y):
    n, xm, ym = len(x), x.mean(), y.mean()
    sxx, syy = ((x-xm)**2).sum(), ((y-ym)**2).sum()
    sxy = ((x-xm)*(y-ym)).sum()
    slope = sxy/sxx
    off = ym-slope*xm
    e = ((off+slope*x-y)**2).sum()
    c2 = (e/(n-2))**.5
    slope_err = c2/sxx**.5
    off_err = c2*(1./n+xm**2/sxx)**.5
    return Uncertain(off, off_err), Uncertain(slope, slope_err)

def emva1288(cam, r=1e9):
    cam.mode = cam.modes_dict["1280x960_Y16"]
    cam.setup(active=False, trigger=None, exposure=None, gamma=None,
            framerate=None)
    cam.setup(gain=0., brightness=1., shutter=.13) # to fix shutter
    cam.set_register(0x1028, 2<<16) # extended shutter
    cam.rate = max(cam.mode.rates)
    dark, bright = [], []
    ts = np.r_[np.linspace(cam.shutter.absolute_range[0], 130e-3, 50)
            ]#np.logspace(np.log10(cam.shutter.range[0]),
            #    np.log10(min(2, cam.shutter.range[1])), 10)]
    ts.sort()
    for res, typ in zip((bright, dark), ("bright", "dark")):
        raw_input("prepare '%s', then press enter" % typ)
        for t in ts:
            print "t=%g" % t
            cam.shutter.absolute = t
            ims = capture(cam, n=2, crop=400)
            y0, sigma0 = noise_mean(ims)
            for d in (y0, sigma0**.5):
                print " mean=%g std=%g med=%g 1%%=%g 99%%=%g" % (
                        d.mean(), d.std(), np.median(d),
                        np.percentile(d, 1), np.percentile(d, 99))
            res.append((y0.mean(), sigma0.mean()))

    dark, bright = np.array(dark).T, np.array(bright).T
    f = pl.figure(figsize=(15, 10))
    for i,(x,y,yl,xl) in enumerate((
            (r*ts, bright[0], "mu_y", "mu_p"),
            (r*ts, bright[1], "sig_y_t", "mu_p"),
            (ts, dark[0], "mu_y_d", "t"),
            (ts, dark[1], "sig_y_d", "t"),
            (bright[0]-dark[0], bright[1]-dark[1],
                "sig_y_t-sig_y_d", "mu_y-mu_y_t_d"),
            (r*ts, bright[0]-dark[0], "mu_y-mu_y_d", "mu_p")),
            ):
        p = f.add_subplot(2, 3, i+1)
        p.plot(x, y, "kx", label=t)
        p.set_xlabel(xl)
        p.set_ylabel(yl)
        #p.legend()
    f.savefig("emva_%x.pdf" % cam.guid)

    ymax = float(1<<16)
    tmin = min(ts[bright[0]/bright[1]**.5>1])
    tmax = ymax/linear(
            ts[np.logical_and(ts>=2*tmin, bright[0]<ymax*.7)],
            bright[0][np.logical_and(ts>=2*tmin, bright[0]<ymax*.7)],
            ).value
    tmin, tmax = tmin+.1*(tmax-tmin), tmax-.1*(tmax-tmin)
    print "limiting exposures to %g<t<%g" % (tmin, tmax)
    dark = dark[:, np.logical_and(ts>tmin, ts<tmax)]
    bright = bright[:, np.logical_and(ts>tmin, ts<tmax)]
    ts = ts[np.logical_and(ts>tmin, ts<tmax)]

    k = linear(bright[0]-dark[0], bright[1]-dark[1])
    print "overal gain K (dn/e): %s" % k
    e = linear(k.value*r*ts, bright[0]-dark[0])
    print "total qe (e/p): %s" % e
    nd0, nd = affine(k.value*ts, dark[0])
    print "dark current (e/s): %s" % nd
    print "dark offset (e): %s" % (nd0/k)
    nd0c, ndc = affine(k.value*ts, dark[1]/k.value)
    print "dark current (compensated) (e/s): %s" % ndc
    print "dark offset (compensated) (e): %s" % (nd0c/k)
    #pl.show()



def main():
    from optparse import OptionParser
    p = OptionParser()
    # p.add_option()
    p.set_defaults(uid=None)
    o,a = p.parse_args()
    cam = Camera(o.uid)
    cam.power(True)
    info(cam)
    print
    emva1288(cam, r=1000e3/20e-3)
    cam.power(False)

if __name__ == "__main__":
    main()
