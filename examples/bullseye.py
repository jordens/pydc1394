#!/usr/bin/python
# -*- coding: utf8 -*-
# (c) Robert Jordens <jordens@debian.org>
#
# GPL-2

from traits.trait_base import ETSConfig
ETSConfig.toolkit = "wx"
from traitsui.api import toolkit
# fix window color on unity
if ETSConfig.toolkit == "wx":
    from traitsui.wx import constants
    import wx
    constants.WindowColor = wx.NullColor

from traits.api import (HasTraits, Range, Float, Enum,
        on_trait_change, TraitError, Event, ListFloat,
        Instance, Bool, Int, Button)

from traitsui.api import (View, Item, UItem,
        HGroup, VGroup,
        DefaultOverride)

from chaco.api import (Plot, ArrayPlotData, color_map_name_dict,
        GridPlotContainer, VPlotContainer, HPlotContainer)
from chaco.tools.pan_tool2 import PanTool
from chaco.tools.api import (ZoomTool, SaveTool, ImageInspectorTool,
        ImageInspectorOverlay)

from enthought.enable.component_editor import ComponentEditor

from enthought.pyface.timer.api import Timer

from pydc1394.camera2 import Camera as DC1394Camera

from angle_sum import angle_sum

import urlparse, logging
import numpy as np
from scipy import stats, optimize
from threading import Thread



class Camera(HasTraits):
    cam = Instance(DC1394Camera)

    shutter = Range(5e-6, 100e-3, 1e-3)
    gain = Range(0., 24., 0.)
    framerate = Range(1, 10, 2)
    average = Range(1, 10, 1)

    auto_shutter = Button("Auto")
    auto_shutter_requested = Bool(False)

    pixelsize = Float(3.75)
    height = Int(960)
    width = Int(1280)

    thread = Instance(Thread)
    active = Bool(False)

    roi = ListFloat([-1280/2, -960/2, 1280, 960], minlen=4, maxlen=4)

    background = Range(0, 50, 5)

    update_image = Bool(True)

    x = Float
    y = Float
    t = Float
    e = Float
    a = Float
    b = Float
    d = Float
    black = Float
    peak = Float

    def __init__(self, uri, **k):
        super(Camera, self).__init__(**k)
        scheme, loc, path, query, frag = urlparse.urlsplit(uri)
        if scheme == "guid":
            self.cam = DC1394Camera(path)
        elif scheme == "first":
            self.cam = DC1394Camera()
        elif scheme == "none":
            self.cam = None
        self.im = None
        self.grid = None
        if self.cam:
            self.setup()

    def initialize(self):
        self.update_roi()
        self.start()
        self.capture()
        self.process()
        self.stop()

    def setup(self):
        self.mode = self.cam.modes_dict["FORMAT7_0"]
        self.cam.mode = self.mode
        self.cam.setup(framerate=self.framerate,
                gain=self.gain, shutter=self.shutter)
        self.cam.setup(active=False,
                exposure=None, brightness=None, gamma=None)

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

    @on_trait_change("auto_shutter")
    def _do_auto_shutter(self):
        self.auto_shutter_requested = True

    def auto(self):
        fr = self.cam.framerate.absolute
        self.cam.framerate.absolute = max(
                self.cam.framerate.absolute_range)
        for i in range(20):
            im_ = self.cam.dequeue()
            im = np.array(im_).astype("float")/(1<<16)
            im_.enqueue()
            # undo gamma
            p = np.percentile(im, 99)
            
            try:
                if p > .75:
                    self.shutter *= .6
                    s = "-"
                elif p < .25:
                    self.shutter /= .6
                    s = "+"
                else:
                    s = "="
            except TraitError:
                break
            # ensure all frames with old settings are gone
            self.cam.flush()
            self.cam.dequeue().enqueue()
            logging.debug("1%%>%g, t%s: %g" % (p, s, self.shutter))
            if s == "=":
                break
        # revert framerate and active state
        self.cam.framerate.absolute = fr

    def update_roi(self):
        l, b, w, h = self.roi
        l = int(min(self.width, max(0, l+self.width/2)))
        b = int(min(self.height, max(0, b+self.height/2)))
        w = int(min(self.width-l, max(128, w)))
        h = int(min(self.height-b, max(128, h)))
        t = self.height-h-b
        if self.cam is not None:
            (w, h), (l, t), _, _ = self.mode.setup(
                    (w, h), (l, t), "Y16")
            logging.debug("new roi %s" % (self.mode.roi,))
        b = self.height-h-t
        self.bounds = l, b, w, h
        logging.debug("new bounds %s" % (self.bounds,))

        px = self.pixelsize
        x = np.arange(l-self.width/2, l+w-self.width/2)*px
        y = np.arange(b-self.height/2, b+h-self.height/2)*px
        xbounds = (np.r_[x, x[-1]+px]-.5*px)
        ybounds = (np.r_[y, y[-1]+px]-.5*px)
        upd = dict((("x", x), ("y", y),
            ("xbounds", xbounds), ("ybounds", ybounds)))
        self.data.arrays.update(upd)
        self.data.data_changed = {"changed": upd.keys()}
        if self.grid is not None:
            self.grid.set_data(xbounds, ybounds)
            #enforce data/screen aspect ratio 1
            sl, sr, sb, st = self.gridm.screen_bounds
            dl, db = self.gridm.range.low
            dr, dt = self.gridm.range.high
            dsdx = float(sr-sl)/(dr-dl)
            dt_new = db+(st-sb)/dsdx
            #dsdy = float(st-sb)/(dt-db)
            #print dsdx, dsdy, dt, dt_new
            self.gridm.range.y_range.high_setting = dt_new

    def get_dummy(self):
        px = self.pixelsize
        l, b, w, h = self.bounds
        y, x = np.mgrid[b:b+h, l:l+w]
        x -= self.width/2
        y -= self.height/2
        x *= px
        y *= px
        x -= 600
        y -= 700
        t = np.deg2rad(15)
        b = 30/4.
        a = 250/4.
        h = .8
        x, y = np.cos(t)*x+np.sin(t)*y, -np.sin(t)*x+np.cos(t)*y
        im = h*np.exp(((x/a)**2+(y/b)**2)/-2.)
        im *= 1+np.random.randn(*im.shape)*.2
        #im += np.random.randn(im.shape)*30
        #logging.debug("im shape %s" % (im.shape,))
        return im

    def capture(self):
        if self.cam:
            im_ = self.cam.dequeue()
            im = np.array(im_).astype("float")/(1<<16)
            im_.enqueue()
            # undo gamma
            logging.debug("im shape, ptp %s %s" % (im.shape, im.ptp()))
        else:
            im = self.get_dummy()
        if self.average > 1 and self.im.shape == im.shape:
            self.im = self.im*(1-1./self.average) + im/self.average
        else:
            self.im = im

    def gauss_process(self, im, background=0):
        if background > 0:
            #imr = im.ravel()
            #low = stats.scoreatpercentile(imr, background)
            #bg = imr[np.where(imr <= low)]
            #bg_mean = bg.mean()
            #im -= bg_mean
            black = np.percentile(im, background)
            im -= black
        y, x = np.ogrid[:im.shape[0], :im.shape[1]]
        m00 = im.sum() or 1.
        m10, m01 = (im*x).sum()/m00, (im*y).sum()/m00
        x -= m10
        y -= m01
        m20, m02 = (im*x**2).sum()/m00, (im*y**2).sum()/m00
        m11 = (im*x*y).sum()/m00
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
        e = b/a
        ab = 2*2**.5*(m20+m02)**.5
        return black, m00, m10, m01, m20, m02, m11, a, b, t, e, ab

    def process(self):
        im = self.im
        px = self.pixelsize
        l, b, w, h = self.bounds

        black, m00, m10, m01, m20, m02, m11, wa, wb, wt, we, wab = \
                self.gauss_process(im, background=self.background)

        self.m00 = m00
        self.m20 = m20
        self.m02 = m02
        self.black = black
        self.peak = m00/(2*np.pi*(m02*m20-m11**2)**.5)
        self.x = (m10+l-self.width/2)*px
        self.y = (m01+b-self.height/2)*px
        self.t = np.rad2deg(wt)
        self.a = wa*px
        self.b = wb*px
        self.d = wab*px
        self.e = we

        ima = angle_sum(im, wt, binsize=1)
        imb = angle_sum(im, wt+np.pi/2, binsize=1)
        xc, yc = m10-im.shape[1]/2., m01-im.shape[0]/2.
        xcr = np.cos(wt)*xc+np.sin(wt)*yc+ima.shape[0]/2.
        ycr = -np.sin(wt)*xc+np.cos(wt)*yc+imb.shape[0]/2.
        rad = 3
        ima = ima[int(max(0, xcr-rad*wa)):
                  int(min(ima.shape[0], xcr+rad*wa))]
        imb = imb[int(max(0, ycr-rad*wb)):
                  int(min(imb.shape[0], ycr+rad*wb))]
        xa = np.arange(ima.shape[0]) - (min(0, rad*wa-xcr)+xcr)
        yb = np.arange(imb.shape[0]) - (min(0, rad*wb-ycr)+ycr)

        upd = dict((
            ("imx", im.sum(axis=0)),
            ("imy", im.sum(axis=1)),
            ("ima", ima),
            ("imb", imb),
            ("a", xa*px),
            ("b", yb*px),
            ))
        if self.update_image:
            upd["img"] = im
        self.data.arrays.update(upd)
        self.data.data_changed = {"changed": upd.keys()}

        x = np.arange(l, l+w)-self.width/2
        y = np.arange(b, b+h)-self.height/2

        grx = m00/(np.pi**.5*wa/2/2**.5)*np.exp(-(2**.5*2*xa/wa)**2)
        gry = m00/(np.pi**.5*wb/2/2**.5)*np.exp(-(2**.5*2*yb/wb)**2)
        gx = m00/(2*np.pi*m20)**.5*np.exp(-(x-self.x/px)**2/m20/2)
        gy = m00/(2*np.pi*m02)**.5*np.exp(-(y-self.y/px)**2/m02/2)
        self.data.set_data("gx", gx)
        self.data.set_data("gy", gy)
        self.data.set_data("grx", grx)
        self.data.set_data("gry", gry)

        self.update_markers()

    def update_markers(self):
        px = self.pixelsize
        ts = np.linspace(0, 2*np.pi, 50)
        ex, ey = self.a*np.cos(ts), self.b*np.sin(ts)
        t = np.deg2rad(self.t)
        ex = ex*np.cos(t)-ey*np.sin(t)
        ey = ex*np.sin(t)+ey*np.cos(t)
        self.data.set_data("ell1_x", self.x+.5*ex)
        self.data.set_data("ell1_y", self.y+.5*ey)
        self.data.set_data("ell3_x", self.x+3/2.*ex)
        self.data.set_data("ell3_y", self.y+3/2.*ey)
        k = np.array([-3/2., 3/2.])
        self.data.set_data("a_x", self.a*k*np.cos(t)+self.x)
        self.data.set_data("a_y", self.a*k*np.sin(t)+self.y)
        self.data.set_data("b_x", -self.b*k*np.sin(t)+self.x)
        self.data.set_data("b_y", self.b*k*np.cos(t)+self.y)

        self.data.set_data("x0_mark", 2*[self.x])
        self.data.set_data("xp_mark", 2*[self.x+2*px*self.m20**.5])
        self.data.set_data("xm_mark", 2*[self.x-2*px*self.m20**.5])
        self.data.set_data("x_bar", [0, self.m00/(2*np.pi*self.m20)**.5])
        self.data.set_data("y0_mark", 2*[self.y])
        self.data.set_data("yp_mark", 2*[self.y+2*px*self.m02**.5])
        self.data.set_data("ym_mark", 2*[self.y-2*px*self.m02**.5])
        self.data.set_data("y_bar", [0, self.m00/(2*np.pi*self.m02)**.5])

        self.data.set_data("a0_mark", 2*[0])
        self.data.set_data("ap_mark", 2*[self.a/2])
        self.data.set_data("am_mark", 2*[-self.a/2])
        self.data.set_data("a_bar", [0, self.m00/(np.pi**.5*self.a/px/2/2**.5)])
        self.data.set_data("b0_mark", 2*[0])
        self.data.set_data("bp_mark", 2*[self.b/2])
        self.data.set_data("bm_mark", 2*[-self.b/2])
        self.data.set_data("b_bar", [0, self.m00/(np.pi**.5*self.b/px/2/2**.5)])


    @on_trait_change("active")
    def _start_me(self, value):
        if value:
            if self.thread is not None:
                if not self.thread.is_alive():
                    self.thread.join()
                    self.thread = None
                else:
                    logging.warning("already have a capture thread, try again")
                return
            else:
                self.thread = Thread(target=self.run)
                self.thread.start()
        else:
            if self.thread is not None:
                self.thread.join()
                assert self.thread is None

    def run(self):
        self.update_roi()
        logging.debug("start")
        self.start()
        while self.active:
            if self.auto_shutter_requested:
                self.auto()
                self.auto_shutter_requested = False
            self.capture()
            logging.debug("captured")
            self.process()
            logging.debug("processed")
        logging.debug("stop")
        self.stop()
        self.thread = None

slider_editor=DefaultOverride(mode="slider")


class Bullseye(HasTraits):
    plots = Instance(GridPlotContainer)
    abplots = Instance(HPlotContainer)
    all_plots = Instance(VPlotContainer)
    screen = Instance(Plot)
    horiz = Instance(Plot)
    vert = Instance(Plot)
    asum = Instance(Plot)
    bsum = Instance(Plot)
    camera = Instance(Camera)

    palette = Enum("gray", "jet", "cool", "hot", "prism", "hsv")

    traits_view = View(HGroup(
        VGroup(
            VGroup(
                Item("object.camera.x", label="Centroid X",
                    format_str=u"%.4g µm"),
                Item("object.camera.y", label="Centroid Y",
                    format_str=u"%.4g µm"),
                Item("object.camera.t", label="Rotation",
                    format_str=u"%.4g°"),
                # widths are full width at 1/e^2 intensity
                Item("object.camera.a", label="Major width",
                    format_str=u"%.4g µm"),
                Item("object.camera.b", label="Minor width",
                    format_str=u"%.4g µm"),
                Item("object.camera.d", label="Mean width",
                    format_str=u"%.4g µm"),
                # minor/major
                Item("object.camera.e", label="Ellipticity",
                    format_str=u"%.4g"),
                Item("object.camera.black", label="Black",
                    format_str=u"%.4g"),
                Item("object.camera.peak", label="Peak",
                    format_str=u"%.4g"),
                style="readonly"),
            VGroup(
                "object.camera.shutter",
                "object.camera.gain",
                "object.camera.framerate",
                "object.camera.average",
                "object.camera.background",
                ),
            HGroup(
                "object.camera.active",
                "object.camera.update_image",
                UItem("object.camera.auto_shutter"),
                "palette"),
            ),
        UItem("all_plots", editor=ComponentEditor(),
            width=600, height=700),
        #VGroup(
        #    UItem("plots", editor=ComponentEditor(),
        #        width=(1280+105)/2, height=(960+105)/2),
        #    UItem("abplots", editor=ComponentEditor(),
        #        height=-200, resizable=False),
        #    ),
        ), resizable=True, title="Bullseye")

    def __init__(self, uri="first:", **k):
        super(Bullseye, self).__init__(**k)
        self.data = ArrayPlotData()

        self.camera = Camera(uri)
        self.camera.data = self.data
        self.camera.initialize()

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
        self.screen.index_grid.visible = False
        self.screen.value_grid.visible = False

        self.horiz = Plot(self.data,
                orientation="h",
                resizable="h", padding=0, height=100,
                bgcolor="black", title="vertical sum",
                border_visible=False, border_color="white")
        self.horiz.title_color = "white"
        self.horiz.title_font = "modern 10"
        self.horiz.title_position = "left"
        self.horiz.title_angle = 90
        self.horiz.index_axis.visible = False
        self.horiz.value_axis.visible = False
        self.horiz.index_grid.visible = True
        self.horiz.value_grid.visible = False
        self.horiz.value_mapper.range.low_setting = 0
        self.horiz.index_range = self.screen.index_range
        self.vert = Plot(self.data,
                orientation="v",
                resizable="v", padding=0, width=100,
                bgcolor="black", title="horizontal sum",
                border_visible=False, border_color="white")
        self.vert.index_axis.visible = False
        self.vert.value_axis.visible = False
        self.vert.index_grid.visible = True
        self.vert.value_grid.visible = False
        self.vert.title_color = "white"
        self.vert.title_font = "modern 10"
        self.vert.title_position = "bottom"

        #self.vert.value_range = self.horiz.value_range
        self.vert.index_range = self.screen.value_range

        self.mini = VPlotContainer(
                width=100, height=100, resizable="",
                padding=0, fill_padding=False, bgcolor="black")

        self.plots = GridPlotContainer(shape=(2,2), padding=0,
                use_backbuffer=True, fill_padding=True,
                spacing=(5,5), halign="left", valign="bottom",
                bgcolor="black",
                component_grid = [
                    [self.vert, self.screen],
                    [self.mini, self.horiz]])

        self.screen.overlays.append(ZoomTool(self.screen,
            tool_mode="box", alpha=.3,
            always_on_modifier="shift",
            always_on=False,
            x_max_zoom_factor=1e2,
            y_max_zoom_factor=1e2,
            x_min_zoom_factor=0.5,
            y_min_zoom_factor=0.5,
            zoom_factor=1.2))
        self.screen.tools.append(PanTool(self.screen))

        self.screenplot = self.screen.img_plot("img", name="img",
                xbounds="xbounds", ybounds="ybounds",
                interpolation="nearest",
                colormap=color_map_name_dict[self.palette],
                )[0]
        self.screenplot.color_mapper.range.low_setting = 0
        self.screenplot.color_mapper.range.high_setting = 1
        self.camera.grid = self.screenplot.index
        self.camera.gridm = self.screenplot.index_mapper
        t = ImageInspectorTool(self.screenplot)
        self.screen.tools.append(t)
        self.screenplot.overlays.append(ImageInspectorOverlay(
            component=self.screenplot, image_inspector=t,
            border_size=0, bgcolor="darkgray", align="ur",
            tooltip_mode=False, font="modern 10"))

        self.asum = Plot(self.data,
                padding=0,
                bgcolor="black", title="major axis sum",
                border_visible=False, border_color="white")
        self.asum.index_axis.tick_color = "white"
        self.asum.value_axis.visible = False
        self.asum.index_axis.tick_label_color = "white"
        self.asum.index_axis.axis_line_color = "white"
        self.asum.value_grid.visible = False
        self.asum.title_color = "white"
        self.asum.title_font = "modern 10"
        self.asum.title_position = "left"
        self.asum.title_angle = 90

        self.bsum = Plot(self.data,
                padding=0,
                bgcolor="black",
                title="minor axis sum",
                border_visible=False, border_color="white")
        self.bsum.index_axis.tick_color = "white"
        self.bsum.value_axis.visible = False
        self.bsum.index_axis.tick_label_color = "white"
        self.bsum.index_axis.axis_line_color = "white"
        self.bsum.value_grid.visible = False
        self.bsum.title_color = "white"
        self.bsum.title_font = "modern 10"
        self.bsum.title_position = "left"
        self.bsum.title_angle = 90
        # lock scales
        #self.bsum.value_range = self.asum.value_range
        #self.bsum.index_range = self.asum.index_range

        self.abplots = HPlotContainer(padding=20,
                use_backbuffer=True, fill_padding=True,
                spacing=10, bgcolor="black")
        self.abplots.add(self.asum)
        self.abplots.add(self.bsum)

        self.all_plots = VPlotContainer(padding=0,
                use_backbuffer=True, fill_padding=True,
                spacing=0, bgcolor="black")
        self.all_plots.add(self.abplots)
        self.all_plots.add(self.plots)
        self.all_plots.tools.append(SaveTool(self.all_plots,
            filename="bullseye.pdf"))

        self.horiz.plot(("x", "imx"), type="line", color="red")
        self.vert.plot(("y", "imy"), type="line", color="red")
        self.horiz.plot(("x", "gx"), type="line", color="blue")
        self.vert.plot(("y", "gy"), type="line", color="blue")
        self.asum.plot(("a", "ima"), type="line", color="red")
        self.bsum.plot(("b", "imb"), type="line", color="red")
        self.asum.plot(("a", "grx"), type="line", color="blue")
        self.bsum.plot(("b", "gry"), type="line", color="blue")

        for p in [("ell1_x", "ell1_y"), ("ell3_x", "ell3_y"),
                ("a_x", "a_y"), ("b_x", "b_y")]:
            self.screen.plot(p, type="line", color="green", alpha=.5)

        for r, s in [("x", self.horiz), ("y", self.vert),
                ("a", self.asum), ("b", self.bsum)]:
            for p in "0 p m".split():
                q = ("%s%s_mark" % (r, p), "%s_bar" % r)
                logging.debug(q)
                s.plot(q, type="line", color="green")

 
    def __del__(self):
        self.close()

    def close(self):
        self.camera.active = False

    @on_trait_change("palette")
    def set_colormap(self):
        p = self.screen.plots["img"][0]
        m = color_map_name_dict[self.palette]
        p.color_mapper = m(p.value_range)

    #@on_trait_change("screen.index_range.updated")
    @on_trait_change("screen.value_range.updated")
    def set_range(self):
        #l, b = self.screenplot.range.low
        #r, t = self.screenplot.range.high
        l, r = self.screen.index_range.low, self.screen.index_range.high
        b, t = self.screen.value_range.low, self.screen.value_range.high
        px = self.camera.pixelsize
        self.camera.roi = [l/px, b/px, (r-l)/px, (t-b)/px]

def main():
    logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s')
    b = Bullseye("first:")
    #b = Bullseye("none:")
    #b = Bullseye("guid:b09d01009981f9")
    b.configure_traits()
    b.close()

if __name__ == '__main__':
    main()

