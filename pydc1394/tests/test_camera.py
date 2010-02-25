#!/usr/bin/env python -tt
# encoding: utf-8

import nose
from nose.tools import *

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from camera import DC1394Library, Camera

# Is a camera connected?
def _has_cam_attached():
    l = DC1394Library()
    if len(l.enumerate_cameras()):
        return True
    return False
_has_camera = _has_cam_attached()

def need_cam(f):
    def skipper(*args, **kwargs):
        if not _has_camera:
            raise nose.SkipTest, "This test needs a camera attached!"
        else:
            return f(*args, **kwargs)
    return nose.tools.make_decorator(f)(skipper)

class LibBase(object):
    def setUp(self):
        self.l = DC1394Library()
    def tearDown(self):
        self.l.close()
class CamBase(LibBase):
    def setUp(self):
        LibBase.setUp(self)
        cams = self.l.enumerate_cameras()
        self.c = Camera(self.l, cams[0]['guid'])
    def tearDown(self):
        self.c.stop()
        self.c.close()
        LibBase.tearDown(self)

# Camera opening and closing
class TestInstantiation(LibBase):
    @need_cam
    def test(self):
        cams = self.l.enumerate_cameras()
        c = Camera(self.l, cams[0]['guid'])
        c.close()

# From now on, we assume that we can get a camera instance
class TestCamera(CamBase):
    @need_cam
    def test_acquisition(self):
        self.c.start()
        i = self.c.shot()
        eq_(i.shape[0], self.c.mode[1])
        eq_(i.shape[1], self.c.mode[0])
        self.c.stop()

    @need_cam
    def test_reset(self):
        self.c.start()
        self.c.reset_bus()

        eq_(self.c.running, False)

        self.c.start()
        self.c.shot()
        self.c.stop()




