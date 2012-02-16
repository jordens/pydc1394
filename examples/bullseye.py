#!/usr/bin/python
# (c) Robert Jordens <jordens@debian.org>
# GPL-2

from traits.api import (HasTraits, Range, Float, Enum,
        on_trait_change, TraitError,
        Property, Instance, Button, Bool, Int, Button)
from traitsui.api import (View, Item, HGroup, VGroup,
        DefaultOverride)

# fix window color on unity
from traitsui.wx import constants
import wx
constants.WindowColor = wx.NullColor

from chaco.api import (Plot, ArrayPlotData, color_map_name_dict,
        GridPlotContainer, VPlotContainer)
from chaco.tools.api import (PanTool, ZoomTool, 
        SaveTool)
from chaco.tools.image_inspector_tool import (
        ImageInspectorTool, ImageInspectorOverlay)

from enthought.enable.component_editor import ComponentEditor

from enthought.pyface.timer.api import Timer

from pydc1394.camera2 import Camera as DC1394Camera

import urlparse, logging
import numpy as np
from scipy import stats, optimize
from threading import Thread


class Camera(HasTraits):
    cam = Instance(DC1394Camera)

    shutter = Range(1e-5, 100e-3, 1e-3)
    gain = Range(0., 24., 0.)
    average = Range(1, 10, 1)
    framerate = Range(1, 10, 2)

    auto_shutter = Button("Auto")

    pixelsize = Float(3.75)

    thread = Instance(Thread)
    active = Bool(False)

    width = Range(8, 1280, 640)
    height = Range(2, 960, 480)
    left = Range(0, 1280-8, 200)
    bottom = Range(0, 960-2, 200)

    def __init__(self, uri, **k):
        super(Camera, self).__init__(**k)
        scheme, loc, path, query, frag = urlparse.urlsplit(uri)
        if scheme == "guid":
            self.cam = DC1394Camera(path)
        elif scheme == "first":
            self.cam = DC1394Camera()
        elif scheme == "none":
            self.cam = None
        self.im = 0
        if self.cam:
            self.setup()

    def setup(self):
        self.mode = self.cam.modes_dict["FORMAT7_0"]
        self._do_mode_setup()
        self.cam.mode = self.mode
        self.cam.setup(gamma=.5, framerate=self.framerate,
                gain=self.gain, shutter=self.shutter)
        self.cam.setup(active=False, exposure=None, brightness=None)

    def start(self):
        if not self.cam:
            return
        self.cam.start_capture()
        self.cam.start_video()

    def stop(self):
        if not self.cam:
            return
        self.cam.stop_video()
        self.cam.stop_capture()

    @on_trait_change("framerate")
    def _do_framerate(self, val):
        self.cam.framerate.absolute = val

    @on_trait_change("shutter")
    def _do_shutter(self, val):
        self.cam.shutter.absolute = val

    @on_trait_change("gain")
    def _do_gain(self, val):
        self.cam.gain.absolute = val

    @on_trait_change("gain")
    def _do_gain(self, val):
        self.cam.gain.absolute = val

    @on_trait_change("width, height, left, bottom")
    def _do_mode_setup(self):
        self.mode.setup(
                image_size=(self.width, self.height),
                image_position=(self.left, 960-self.height-self.bottom),
                #image_position=(self.left, self.bottom),
                color_coding="Y8")

    @on_trait_change("auto_shutter")
    def _do_auto_shutter(self):
        ac = self.active
        fr = self.cam.framerate.absolute
        self.active = False
        self.cam.framerate.absolute = max(
                self.cam.framerate.absolute_range)
        self.start()
        for i in range(20):
            im_ = self.cam.dequeue()
            im = np.array(im_).copy()
            im_.enqueue()
            im = im.astype("float32")**2/256 # undo gamma
            p = np.percentile(im, 99)
            print "1%%>%g:" % p,
            try:
                if p > .75*256:
                    self.shutter *= .8
                    print "t-%g" % self.shutter
                elif p < .25*256:
                    self.shutter /= .8
                    print "t+%g" % self.shutter
                else:
                    print "t=%g" % self.shutter
                    break
                # ensure all frames with old settings are gone
                self.cam.flush()
                self.cam.dequeue().enqueue()
            except TraitError:
                break
        self.stop()
        # revert framerate and active state
        self.cam.framerate.absolute = fr
        self.active = ac

    def update(self):
        if self.cam:
            im_ = self.cam.dequeue()
            im = np.array(im_).copy()
            im_.enqueue()
            im = im.astype("float32")**2/256 # undo gamma
        else:
            px = self.pixelsize
            w, h = self.width, self.height
            l, b = self.left, self.bottom
            y, x = np.mgrid[b:b+h, l:l+w]
            x *= px
            y *= px
            x -= 1.1e3
            y -= 1.2e3
            t = 13./180.*np.pi
            a = 140/4
            b = 150/4
            h = 200
            x, y = np.cos(t)*x+np.sin(t)*y, -np.sin(t)*x+np.cos(t)*y
            im = h*np.exp(-x**2/a**2/2.-y**2/b**2/2.)
            im *= 1+np.random.randn(*im.shape)*.2
            #im += np.random.randn(im.shape)*30
        if self.average > 1 and self.im.shape == im.shape:
            self.im = self.im*(1-1./self.average) + im/self.average
        else:
            self.im = im
        self.data.set_data("img", self.im)

    @on_trait_change("active")
    def _start_me(self, value):
        if value:
            print "starting capture"
            if self.thread is not None:
                print "already have a thread, try again"
                return
            else:
                self.thread = Thread(target=self.run)
                self.thread.start()
        else:
            if self.thread is not None:
                self.thread.join()
                self.thread = None

    def run(self):
        print "start capture"
        self.start()
        while self.active:
            print "capture"
            self.update()
        print "stop capture"
        self.stop()


class Analysis(HasTraits):
    background = Bool(False)

    x = Float
    y = Float
    t = Float
    e = Float
    a = Float
    b = Float

    thread = Instance(Thread)
    active = Bool(False)

    @on_trait_change("active")
    def _start_me(self, value):
        if value:
            print "starting analysis"
            if self.thread:
                print "already have a thread, try again"
                return
            else:
                self.thread = Thread(target=self.run)
                self.thread.start()

    def update(self):
        im = self.data.get_data("img").copy()
        px = self.camera.pixelsize
        w, h = self.camera.width, self.camera.height
        l, b = self.camera.left, self.camera.bottom

        if self.background:
            #imr = im.ravel()
            #low = stats.scoreatpercentile(imr, 20)
            #bg = imr[np.where(imr <= low)]
            #bg_mean = bg.mean()
            #im -= bg_mean
            im -= np.percentile(im, 10)

        y, x = np.ogrid[b:b+h, l:l+w]
        x *= px
        y *= px

        self.data.set_data("x", x[0, :].copy())
        self.data.set_data("y", y[:, 0].copy())
        self.data.set_data("xbounds", np.r_[x[0, :], (l+w)*px])
        self.data.set_data("ybounds", np.r_[y[:, 0], (b+h)*px])
        self.data.set_data("imx", im.sum(axis=0))
        self.data.set_data("imy", im.sum(axis=1))

        m00 = im.sum()
        im /= m00
        m10, m01 = (im*x).sum(), (im*y).sum()
        x -= m10
        y -= m01
        m20, m02 = (im*x**2).sum(), (im*y**2).sum()
        m11 = (im*x*y).sum()

        g = np.sign(m20-m02)
        if g == 0:
            a = 2*2**.5*(m20+m02+2*np.abs(m11))**.5
            b = 2*2**.5*(m20+m02-2*np.abs(m11))**.5
            t = np.pi/4*np.sign(m11)
        else:
            q = g*((m20-m02)**2+4*m11**2)**.5
            a = 2*2**.5*((m20+m02)+q)**.5
            b = 2*2**.5*((m20+m02)-q)**.5
            t = .5*np.arctan2(2*m11, m20-m02)
        e = a/b
        ab = 2*2**.5*(m20+m02)**.5

        self.x = m10
        self.y = m01
        self.t = t/np.pi*180
        self.a = a
        self.b = b
        self.e = e

        
        gx = m00*self.camera.pixelsize/(2*np.pi*m20)**.5*np.exp(-x[:, 0]**2/m20/2)
        gy = m00*self.camera.pixelsize/(2*np.pi*m02)**.5*np.exp(-y[0, :]**2/m02/2)

        self.data.set_data("gauss_x", gx)
        self.data.set_data("gauss_y", gy)

        return

        ts = np.linspace(0, 2*np.pi, 50)
        t = self.t/180*np.pi
        ell2_x = self.x+.5*(
                self.a*np.cos(ts)*np.cos(t)-self.b*np.sin(ts)*np.sin(t))
        ell2_y = self.y+.5*(
                self.b*np.sin(ts)*np.cos(t)+self.a*np.cos(ts)*np.sin(t))
        self.data.set_data("ell2_x", ell2_x)
        self.data.set_data("ell2_y", ell2_y)
        ai = np.linspace(-self.a, self.a, 2)
        bi = np.linspace(-self.b, self.b, 2)
        self.data.set_data("a_x", ai*np.cos(t)+self.x)
        self.data.set_data("a_y", ai*np.sin(t)+self.y)
        self.data.set_data("b_x", -bi*np.sin(t)+self.x)
        self.data.set_data("b_y", bi*np.cos(t)+self.y)
        self.data.set_data("x0_mark", 2*[self.x])
        self.data.set_data("xp_mark", 2*[max(ell2_x)])
        self.data.set_data("xm_mark", 2*[min(ell2_x)])
        self.data.set_data("x_bar", [0, max(self.data.get_data("imx"))])
        self.data.set_data("y0_mark", 2*[self.y])
        self.data.set_data("yp_mark", 2*[max(ell2_y)])
        self.data.set_data("ym_mark", 2*[min(ell2_y)])
        self.data.set_data("y_bar", [0, max(self.data.get_data("imy"))])

    def run(self):
        print "start analysis"
        while self.active:
            print "analysis"
            self.update()
        print "stop analysis"
        self.thread = None


slider_editor=DefaultOverride(mode="slider")


class Bullseye(HasTraits):
    plots = Instance(GridPlotContainer)
    screen = Instance(Plot)
    horiz = Instance(Plot)
    vert = Instance(Plot)
    camera = Instance(Camera)
    analysis = Instance(Analysis)

    palette = Enum("gray", "jet", "cool", "hot", "prism", "hsv")

    traits_view = View(HGroup(
        VGroup(
            VGroup(
                Item("object.analysis.x", label="Centroid X", format_str="%g"),
                Item("object.analysis.y", label="Centroid Y", format_str="%g"),
                Item("object.analysis.t", label="Angle", format_str="%g"),
                Item("object.analysis.a", label="Major axis (4w)", format_str="%g"),
                Item("object.analysis.b", label="Minor axis (4w)", format_str="%g"),
                Item("object.analysis.e", label="Ellipticity", format_str="%g"),
                style="readonly"),
            VGroup(HGroup(
                "object.camera.shutter",
                Item("object.camera.auto_shutter", show_label=False)),
                "object.camera.gain",
                "object.camera.framerate",
                "object.camera.average"),
               HGroup(
                      "object.camera.width",
                      "object.camera.height",
                      "object.camera.left",
                      "object.camera.bottom", style="readonly"),
            VGroup(HGroup(
                Item("object.camera.active", label="Capture"),
                Item("object.analysis.active", label="Process")),
                   HGroup(
                "object.analysis.background",
                "palette"),),
            ),
        Item("plots", editor=ComponentEditor(),
            show_label=False),
        ), resizable=True, title='Bullseye')

    def __init__(self, uri="first:", **k):
        super(Bullseye, self).__init__(**k)
        self.data = ArrayPlotData()

        self.camera = Camera(uri)
        self.camera.data = self.data
        self.camera.start()
        self.camera.update()
        self.camera.stop()

        self.analysis = Analysis()
        self.analysis.data = self.data
        self.analysis.camera = self.camera
        self.analysis.update()

        self.plots = GridPlotContainer(shape=(2,2),
                padding_left=0, padding_bottom=0,
                padding_top=0, padding_right=0,
                use_backbuffer=True, fill_padding=True,
                spacing=(0,0), halign="left", valign="bottom",
                bgcolor="black",
                )
        self.screen = Plot(self.data, bgcolor="black",
                border_visible=True, border_color="white",
                padding=0, resizable="hv",
                )
        self.screen.index_axis.tick_color = "white"
        self.screen.value_axis.tick_color = "white"
        self.screen.index_axis.tick_label_color = "white"
        self.screen.value_axis.tick_label_color = "white"
        self.screen.index_axis.axis_line_color = "white"
        self.screen.value_axis.axis_line_color = "white"

        self.horiz = Plot(self.data,
                orientation="h",
                resizable="h", padding=0, height=100,
                bgcolor="black",
                border_visible=False, border_color="white")
        self.horiz.index_axis.visible = False
        self.horiz.value_axis.visible = False
        self.horiz.index_grid.visible = True
        self.horiz.value_grid.visible = True
        self.vert = Plot(self.data,
                orientation="v",
                resizable="v", padding=0, width=100,
                bgcolor="black",
                border_visible=False, border_color="white")
        self.vert.index_axis.visible = False
        self.vert.value_axis.visible = False
        self.vert.index_grid.visible = True
        self.vert.value_grid.visible = True

        self.mini = VPlotContainer(
                width=100, height=100, resizable="",
                padding=0, fill_padding=False, bgcolor="black")

        self.plots.component_grid = [
                [self.vert, self.screen],
                [self.mini, self.horiz]]
        self.horiz.index_range = self.screen.index_range
        self.vert.index_range = self.screen.value_range

        self.screen.overlays.append(ZoomTool(self.screen,
            tool_mode="box", alpha=.3,
            always_on_modifier="shift",
            always_on=False,
            x_max_zoom_factor=1e2,
            y_max_zoom_factor=1e2,
            x_min_zoom_factor=1,
            y_min_zoom_factor=1,
            zoom_factor=1.2))
        self.screen.tools.append(PanTool(self.screen))
        self.plots.tools.append(SaveTool(self.plots,
            filename="bullseye.png"))

        i = self.screen.img_plot("img", name="img",
                xbounds="xbounds", ybounds="ybounds",
                interpolation="nearest",
                colormap=color_map_name_dict["gray"],
                )[0]
        i.color_mapper.range.low_setting = 0
        i.color_mapper.range.high_setting = 256
        t = ImageInspectorTool(i)
        i.tools.append(t)
        o = ImageInspectorOverlay(component=i, image_inspector=t,
            bgcolor="darkgray", border_visible=False, tooltip_mode=True,
            font="modern 10")
        i.overlays.append(o)

        self.horiz.plot(("x", "imx"), type="line", color="red")
        self.horiz.plot(("x", "gauss_x"), type="line", color="blue")
        self.vert.plot(("y", "imy"), type="line", color="red")
        self.vert.plot(("y", "gauss_y"), type="line", color="blue")

        return

        self.screen.plot(("ell2_x", "ell2_y"), type="line",
                color="white")
        self.screen.plot(("a_x", "a_y"), type="line",
                color="white")
        self.screen.plot(("b_x", "b_y"), type="line",
                color="white")

        self.horiz.plot(("x0_mark", "x_bar"), type="line",
                color="white")
        self.horiz.plot(("xp_mark", "x_bar"), type="line",
                color="white")
        self.horiz.plot(("xm_mark", "x_bar"), type="line",
                color="white")
        self.vert.plot(("y0_mark", "y_bar"), type="line",
                color="white")
        self.vert.plot(("yp_mark", "y_bar"), type="line",
                color="white")
        self.vert.plot(("ym_mark", "y_bar"), type="line",
                color="white")

    def __del__(self):
        self.close()

    def close(self):
        self.camera.active = False
        self.analysis.active = False

    @on_trait_change("palette")
    def set_colormap(self):
        p = self.screen.plots["img"][0]
        m = color_map_name_dict[self.palette]
        p.color_mapper = m(p.value_range)


def main():
    logging.basicConfig(level=logging.WARNING)
    #b = Bullseye("first:")
    b = Bullseye("none:")
    #b = Bullseye("guid:b09d01009981f9")
    b.configure_traits()
    b.close()

if __name__ == '__main__':
    main()

