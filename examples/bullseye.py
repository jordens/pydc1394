#!/usr/bin/python
# (c) Robert Jordens <jordens@debian.org>
# GPL-2

from traits.api import (HasTraits, Range, Float, Enum,
        on_trait_change, TraitError,
        Property, Instance, Button, Bool, Int, Button)
from traitsui.api import (View, Item, HGroup, VGroup,
        DefaultOverride)

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

    auto_shutter = Button("Auto")

    pixelsize = Float(3.75)

    thread = Instance(Thread)
    active = Bool(False)

    width = Range(8, 1280, 640)
    height = Range(2, 960, 480)
    left = Range(0, 1280-8, 200)
    top = Range(0, 960-2, 200)

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
        self.cam.setup(gamma=.5, framerate=10.,
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

    @on_trait_change("shutter")
    def _do_shutter(self, val):
        self.cam.shutter.absolute = val

    @on_trait_change("gain")
    def _do_gain(self, val):
        self.cam.gain.absolute = val

    @on_trait_change("gain")
    def _do_gain(self, val):
        self.cam.gain.absolute = val

    @on_trait_change("width, height, left, top")
    def _do_mode_setup(self):
        self.mode.setup(image_size=(self.width, self.height),
                image_position=(self.left, self.top), color_coding="Y8")

    @on_trait_change("auto_shutter")
    def _do_auto_shutter(self):
        ac = self.active
        self.active = False
        fr = self.cam.framerate.absolute
        self.cam.framerate.absolute = max(self.cam.framerate.absolute_range)
        self.start()
        while True:
            self.update()
            p = np.percentile(self.im, 99)
            print "99 percentile is %g," % p,
            try:
                if p > .75*256:
                    self.shutter *= .8
                    print "decreasing shutter to %g" % self.shutter
                elif p < .5*256:
                    self.shutter /= .8
                    print "increasing shutter to %g" % self.shutter
                else:
                    print "leaving shutter at %g" % self.shutter
                    break
            except TraitError:
                break
        self.stop()
        self.cam.framerate.absolute = fr
        self.active = ac

    def update(self):
        if self.cam:
            im_ = self.cam.dequeue()
            im = im_.astype("float")**2/256 # undo gamma
            im_.enqueue()
        else:
            y, x = np.mgrid[:640,:480]
            x -= 300.
            y -= 200.
            a = -10./180.*np.pi
            x, y = np.cos(a)*x-np.sin(a)*y, np.sin(a)*x+np.cos(a)*y
            im = 200*np.exp(-x**2/30.**2/2.-y**2/20.**2/2.)
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
    leastsq = Bool(False)

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
        im = self.data.get_data("img").astype("float32")
        px = self.camera.pixelsize

        if self.background:
            #imr = im.ravel()
            #low = stats.scoreatpercentile(imr, 20)
            #bg = imr[np.where(imr <= low)]
            #bg_mean = bg.mean()
            #im -= bg_mean
            im -= np.percentile(im, 10)

        y, x = np.ogrid[:im.shape[0], :im.shape[1]]

        self.data.set_data("x", x[0, :]*px)
        self.data.set_data("y", y[:, 0]*px)
        self.data.set_data("im_x", im.sum(axis=0))
        self.data.set_data("im_y", im.sum(axis=1))

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
        self.a = a*px
        self.b = b*px
        self.e = e

        gx = 0*m00*self.camera.pixelsize/(2*np.pi*m20)**.5*np.exp(-x[:, 0]**2/m20/2)
        gy = 0*m00*self.camera.pixelsize/(2*np.pi*m02)**.5*np.exp(-y[0, :]**2/m02/2)

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
        self.data.set_data("x_bar", [0, max(self.data.get_data("im_x"))])
        self.data.set_data("y0_mark", 2*[self.y])
        self.data.set_data("yp_mark", 2*[max(ell2_y)])
        self.data.set_data("ym_mark", 2*[min(ell2_y)])
        self.data.set_data("y_bar", [0, max(self.data.get_data("im_y"))])

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
        VGroup(Item("plots", editor=ComponentEditor(),
            show_label=False),
            HGroup(
                Item("object.analysis.x", label="x0", format_str="%g", width=50),
                Item("object.analysis.y", label="y0", format_str="%g", width=50),
                Item("object.analysis.t", label="ang", format_str="%g", width=50),
                Item("object.analysis.a", label="maj", format_str="%g", width=50),
                Item("object.analysis.b", label="min", format_str="%g", width=50),
                Item("object.analysis.e", label="ell", format_str="%g", width=50),
                springy=True, padding=0, style="readonly"),
            VGroup(HGroup("object.camera.shutter",
                        Item("object.camera.auto_shutter", show_label=False),
                        "object.camera.gain",
                        "object.camera.average"),
                  HGroup(
                      Item("object.camera.width", editor=slider_editor),
                      Item("object.camera.height", editor=slider_editor),
                      Item("object.camera.left", editor=slider_editor),
                      Item("object.camera.top", editor=slider_editor)),
                   HGroup(Item("object.camera.active", label="capture"),
                       Item("object.analysis.active", label="process"),
                       "object.analysis.background",
                       "object.analysis.leastsq",
                       "palette"),),
        ),
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

        self.plots = GridPlotContainer(shape=(2,2),
                padding_left=40, padding_bottom=20,
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
            tool_mode="box"))
        self.screen.tools.append(PanTool(self.screen))
        self.plots.tools.append(SaveTool(self.plots,
            filename="bullseye.png"))
        i = self.screen.img_plot("img", name="img",
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

        self.analysis.update()

        self.horiz.plot(("x", "im_x"), type="line", color="red")
        self.horiz.plot(("x", "gauss_x"), type="line", color="blue")
        self.vert.plot(("y", "im_y"), type="line", color="red")
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

    @on_trait_change("screen.index_mapper.range.low")
    def _up_low(self, val):
        self.analysis.left = val

    @on_trait_change("screen.index_mapper.range.high")
    def _up_low(self, val):
        self.analysis.right = val

    @on_trait_change("screen.value_mapper.range.low")
    def _up_low(self, val):
        self.analysis.bottom = val

    @on_trait_change("screen.value_mapper.range.high")
    def _up_low(self, val):
        self.analysis.top = val

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

