#!/usr/bin/python
# -*- coding: utf8 -*-
#
#   bullseye - ccd laser beam profilers (pydc1394 + chaco)
#   Copyright (C) 2012 Robert Jordens <jordens@phys.ethz.ch>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from traits.trait_base import ETSConfig
ETSConfig.toolkit = "qt4"
# fix window color on unity
if ETSConfig.toolkit == "wx":
    from traitsui.wx import constants
    constants.WindowColor = constants.wx.NullColor

from traits.api import (HasTraits, Range, Int, Float, Enum, Bool,
        Unicode, Str, ListFloat, Instance, Delegate,
        on_trait_change, TraitError)

from traitsui.api import (View, Item, UItem,
        HGroup, VGroup, DefaultOverride)

from chaco.api import (Plot, ArrayPlotData, color_map_name_dict,
        GridPlotContainer, VPlotContainer, PlotLabel)
from chaco.tools.api import (ZoomTool, SaveTool, ImageInspectorTool,
        ImageInspectorOverlay, PanTool)

from enthought.enable.component_editor import ComponentEditor

from pydc1394.camera2 import Camera as DC1394Camera

from angle_sum import angle_sum

import urlparse, logging, time
from contextlib import closing
import numpy as np
from threading import Thread


class Camera(HasTraits):
    cam = Instance(DC1394Camera)

    pixelsize = Float(3.75)
    height = Int(960)
    width = Int(1280)
    maxval = Int((1<<8)-1)
    min_shutter = 5e-6
    max_shutter = 100e-3

    shutter = Range(min_shutter, max_shutter, 1e-3)
    auto_shutter = Bool(False)
    gain = Range(0., 24., 0.)
    framerate = Range(1, 10, 2)
    average = Range(1, 20, 1)

    roi = ListFloat([-1280/2, -960/2, 1280, 960], minlen=4, maxlen=4)

    thread = Instance(Thread)
    active = Bool(False)

    crops = Int(2) # crop iterations
    rad = Float(3/2.) # crop radius

    background = Range(0, 50, 5)

    x = Float
    y = Float
    t = Float
    e = Float
    a = Float
    b = Float
    d = Float
    black = Float
    peak = Float

    text = Unicode
    save_format = Str

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
        self.start()
        self.capture()
        self.process()
        self.stop()

    def setup(self):
        self.mode = self.cam.modes_dict["1280x960_Y8"]
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

    def auto(self, im, percentile=99, maxiter=10,
            minval=.25, maxval=.75, adjustment_factor=.6):
        p = np.percentile(im, 99)/float(self.maxval)
        if not ((p < .25 and self.shutter < self.max_shutter) or
                (p > .75 and self.shutter > self.min_shutter)):
            return im
        fr = self.cam.framerate.absolute
        self.cam.framerate.absolute = max(
                self.cam.framerate.absolute_range)
        while True:
            with closing(self.cam.dequeue()) as im_:
                p = np.percentile(im_, percentile)/float(self.maxval)
                s = "="
                if p > maxval:
                    self.shutter = max(self.min_shutter,
                            self.shutter*adjustment_factor)
                    s = "-"
                elif p < minval:
                    self.shutter = min(self.max_shutter,
                            self.shutter/adjustment_factor)
                    s = "+"
                logging.debug("1%%>%g, t%s: %g" % (p, s, self.shutter))
                if s == "=" or self.shutter in (self.min_shutter,
                        self.max_shutter) or maxiter == 0:
                    im = np.array(im_).copy()
                    break
            # ensure all frames with old settings are gone
            self.cam.flush()
            self.cam.dequeue().enqueue()
            maxiter -= 1
        # revert framerate
        self.cam.framerate.absolute = fr
        return im

    def bounds(self):
        l, b, w, h = self.roi
        l = int(min(self.width, max(0, l+self.width/2)))
        b = int(min(self.height, max(0, b+self.height/2)))
        w = int(min(self.width-l, max(8, w)))
        h = int(min(self.height-b, max(8, h)))
        return l, b, w, h

    def get_dummy(self):
        px = self.pixelsize
        l, b, w, h = self.bounds()
        x, y = 600., 700.
        a, c = 250/4., 150/4.
        m = .8
        t = np.deg2rad(15.)
        j, i = np.ogrid[b:b+h, l:l+w]
        i, j = (i-self.width/2)*px-x, (j-self.height/2)*px-y
        i, j = np.cos(t)*i+np.sin(t)*j, -np.sin(t)*i+np.cos(t)*j
        im = self.maxval*m*np.exp(-((i/a)**2+(j/c)**2)/2)
        im *= 1+np.random.randn(*im.shape)*.2
        #im += np.random.randn(im.shape)*30
        return (im+.5).astype(np.int)

    def capture(self):
        if self.cam:
            with closing(self.cam.dequeue()) as im_:
                im = np.array(im_).copy()
            if self.auto_shutter:
                im = self.auto(im)
            if self.save_format:
                name = time.strftime(self.save_format)
                np.savez_compressed(name, im)
                logging.debug("saved as %s" % name)
            l, b, w, h = self.bounds()
            im = im[b:b+h, l:l+w].astype(np.int)
        else:
            im = self.get_dummy()
        if self.average > 1 and self.im.shape == im.shape:
            # TODO: rounding errors since int
            self.im = (self.im*self.average + im)/(self.average + 1)
        else:
            self.im = im
  
    def moments(self, im):
        y, x = np.ogrid[:im.shape[0], :im.shape[1]]
        imx, imy = im.sum(axis=0)[None, :], im.sum(axis=1)[:, None]
        m00 = float(imx.sum()) or 1.
        m10, m01 = (imx*x).sum()/m00, (imy*y).sum()/m00
        x, y = x-m10, y-m01
        m20, m02 = (imx*x**2).sum()/m00, (imy*y**2).sum()/m00
        m11 = (im*x*y).sum()/m00
        return m00, m10, m01, m20, m02, m11

    def gauss_axes(self, m00, m10, m01, m20, m02, m11):
        q = ((m20-m02)**2+4*m11**2)**.5
        a = 2*2**.5*((m20+m02)+q)**.5
        b = 2*2**.5*((m20+m02)-q)**.5
        t = .5*np.arctan2(2*m11, m20-m02)
        return a, b, t

    def process(self):
        im = self.im

        if self.background > 0:
            black = np.percentile(im, self.background)
            im -= black
        else:
            black = 0

        imc = im
        lc, bc = 0, 0
        for i in range(self.crops):
            m00, m10, m01, m20, m02, m11 = self.moments(imc)
            w20 = self.rad*4*m20**.5
            w02 = self.rad*4*m02**.5
            if i < self.crops-1:
                lc += int(max(0, m10-w20))
                bc += int(max(0, m01-w02))
                imc = imc[
                      int(max(0, m01-w02)):
                      int(min(imc.shape[0], m01+w02)),
                      int(max(0, m10-w20)):
                      int(min(imc.shape[1], m10+w20))]
        wa, wb, wt = self.gauss_axes(m00, m10, m01, m20, m02, m11)

        m10 += lc
        m01 += bc
        px = self.pixelsize
        l, b, w, h = self.bounds()

        self.m00 = m00
        self.m20 = m20
        self.m02 = m02
        self.black = black/self.maxval
        peak = m00/(2*np.pi*(m02*m20-m11**2)**.5)
        self.peak = (peak+black)/self.maxval
        self.x = (m10+l-self.width/2)*px
        self.y = (m01+b-self.height/2)*px
        self.t = np.rad2deg(wt)
        self.a = wa*px
        self.b = wb*px
        self.d = ((self.a**2+self.b**2)/2)**.5
        self.e = wb/wa

        self.update_text()

        x = np.arange(l, l+w)-self.width/2
        y = np.arange(b, b+h)-self.height/2
        xbounds = (np.r_[x, x[-1]+1]-.5)*px
        ybounds = (np.r_[y, y[-1]+1]-.5)*px
        imx = im.sum(axis=0)
        imy = im.sum(axis=1)
        gx = (m00/(2*np.pi*m20)**.5)*np.exp(-(x-self.x/px)**2/(m20*2))
        gy = (m00/(2*np.pi*m02)**.5)*np.exp(-(y-self.y/px)**2/(m02*2))

        #TODO: fix half pixel offset
        xc, yc = m10-im.shape[1]/2., m01-im.shape[0]/2.
        ima = angle_sum(im, wt, binsize=1)
        imb = angle_sum(im, wt+np.pi/2, binsize=1)
        xcr = np.cos(wt)*xc+np.sin(wt)*yc+ima.shape[0]/2.
        ycr = -np.sin(wt)*xc+np.cos(wt)*yc+imb.shape[0]/2.
        ima = ima[int(max(0, xcr-self.rad*wa)):
                  int(min(ima.shape[0], xcr+self.rad*wa))]
        imb = imb[int(max(0, ycr-self.rad*wb)):
                  int(min(imb.shape[0], ycr+self.rad*wb))]
        a = np.arange(ima.shape[0]) - min(xcr, self.rad*wa)
        b = np.arange(imb.shape[0]) - min(ycr, self.rad*wb)
        ga = (m00/(np.pi**.5*wa/2/2**.5))*np.exp(-a**2*(2**.5*2/wa)**2)
        gb = (m00/(np.pi**.5*wb/2/2**.5))*np.exp(-b**2*(2**.5*2/wb)**2)

        upd = dict((
            ("img", im),
            ("xbounds", xbounds), ("ybounds", ybounds),
            ("x", x*px), ("y", y*px),
            ("imx", imx), ("imy", imy),
            ("gx", gx), ("gy", gy),
            ("a", a*px), ("b", b*px),
            ("ima", ima), ("imb", imb),
            ("ga", ga), ("gb", gb),
            ))
        upd.update(self.update_markers())
        self.data.arrays.update(upd)
        self.data.data_changed = {"changed": upd.keys()}
        if self.grid is not None:
            self.grid.set_data(xbounds, ybounds)


    def update_markers(self):
        px = self.pixelsize
        ts = np.linspace(0, 2*np.pi, 40)
        ex, ey = self.a/2*np.cos(ts), self.b/2*np.sin(ts)
        t = np.deg2rad(self.t)
        ex, ey = ex*np.cos(t)-ey*np.sin(t), ex*np.sin(t)+ey*np.cos(t)
        k = np.array([-3/2., 3/2.])

        upd = dict((
            ("ell1_x", self.x+ex),
            ("ell1_y", self.y+ey),
            ("ell3_x", self.x+3*ex),
            ("ell3_y", self.y+3*ey),
            ("a_x", self.a*k*np.cos(t)+self.x),
            ("a_y", self.a*k*np.sin(t)+self.y),
            ("b_x", -self.b*k*np.sin(t)+self.x),
            ("b_y", self.b*k*np.cos(t)+self.y),
            ("x0_mark", 2*[self.x]),
            ("xp_mark", 2*[self.x+2*px*self.m20**.5]),
            ("xm_mark", 2*[self.x-2*px*self.m20**.5]),
            ("x_bar", [0, self.m00/(2*np.pi*self.m20)**.5]),
            ("y0_mark", 2*[self.y]),
            ("yp_mark", 2*[self.y+2*px*self.m02**.5]),
            ("ym_mark", 2*[self.y-2*px*self.m02**.5]),
            ("y_bar", [0, self.m00/(2*np.pi*self.m02)**.5]),
            ("a0_mark", 2*[0]),
            ("ap_mark", 2*[self.a/2]),
            ("am_mark", 2*[-self.a/2]),
            ("a_bar", [0, self.m00/(np.pi**.5*self.a/px/2/2**.5)]),
            ("b0_mark", 2*[0]),
            ("bp_mark", 2*[self.b/2]),
            ("bm_mark", 2*[-self.b/2]),
            ("b_bar", [0, self.m00/(np.pi**.5*self.b/px/2/2**.5)]),
        ))
        return upd

    def update_text(self):
        fields = (self.x, self.y,
                self.a, self.b,
                self.t, self.e,
                self.black, self.peak)

        logging.info("beam: "+(("% 5.4g,"*len(fields)) % fields))

        self.text = (
            u"centroid x: %.4g µm\n"
            u"centroid y: %.4g µm\n"
            u"major: %.4g µm\n"
            u"minor: %.4g µm\n"
            u"angle: %.4g°\n"
            u"ellipticity: %.4g\n"
            u"black: %.4g\n"
            u"peak: %.4g\n"
            ) % fields

    @on_trait_change("active")
    def _start_me(self, value):
        if value:
            if self.thread is not None:
                if self.thread.is_alive():
                    logging.warning(
                            "already have a capture thread running")
                    return
                else:
                    self.thread.join()
            self.thread = Thread(target=self.run)
            self.thread.start()
        else:
            if self.thread is not None:
                self.thread.join(timeout=5)
                if self.thread is not None:
                    if self.thread.is_alive():
                        logging.warning(
                                "capture thread did not terminate")
                        return
                    else:
                        logging.warning(
                                "capture thread crashed")
                        self.thread = None
            else:
                logging.warning(
                    "capture thread terminated")

    def run(self):
        logging.debug("start")
        self.start()
        while self.active:
            self.capture()
            self.process()
            #logging.debug("image processed")
        logging.debug("stop")
        self.stop()
        self.thread = None

slider_editor=DefaultOverride(mode="slider")


class Bullseye(HasTraits):
    plots = Instance(GridPlotContainer)
    abplots = Instance(VPlotContainer)
    screen = Instance(Plot)
    horiz = Instance(Plot)
    vert = Instance(Plot)
    asum = Instance(Plot)
    bsum = Instance(Plot)
    camera = Instance(Camera)

    palette = Enum("gray", "jet", "cool", "hot", "prism", "hsv")
    invert = Bool(True)

    save_format = Delegate("camera", prefix="save_format", modify=True)

    traits_view = View(HGroup(VGroup(
        HGroup(
            VGroup(
                Item("object.camera.x", label="Centroid X",
                    format_str=u"%.4g µm"),
                # widths are full width at 1/e^2 intensity
                Item("object.camera.a", label="Major width",
                    format_str=u"%.4g µm"),
                Item("object.camera.t", label="Rotation",
                    format_str=u"%.4g°"),
                Item("object.camera.black", label="Black",
                    format_str=u"%.4g"),
            ), VGroup(
                Item("object.camera.y", label="Centroid Y",
                    format_str=u"%.4g µm"),
                Item("object.camera.b", label="Minor width",
                    format_str=u"%.4g µm"),
                #Item("object.camera.d", label="Mean width",
                #    format_str=u"%.4g µm"),
                # minor/major
                Item("object.camera.e", label="Ellipticity",
                    format_str=u"%.4g"),
                Item("object.camera.peak", label="Peak",
                    format_str=u"%.4g")),
            style="readonly",
        ), VGroup(
            "object.camera.shutter",
            "object.camera.gain",
            "object.camera.framerate",
            "object.camera.average",
            "object.camera.background",
        ), HGroup(
            "object.camera.active",
            "object.camera.auto_shutter",
            UItem("palette"),
            "invert"
        ), UItem("abplots", editor=ComponentEditor(),
                width=-200, height=-300, resizable=False,
        ),
    ), UItem("plots", editor=ComponentEditor(),
            width=600),
    layout="split",
    ), resizable=True, title=u"Bullseye ― Beam Profiler", width=1000)

    def __init__(self, uri="first:", **k):
        super(Bullseye, self).__init__(**k)
        self.label = None
        self.gridm = None

        self.data = ArrayPlotData()
        self.camera = Camera(uri)
        self.camera.data = self.data
        self.camera.initialize()

        self.setup_plots()
        self.populate_plots()

    def setup_plots(self):
        self.screen = Plot(self.data,
                resizable="hv", padding=0, bgcolor="lightgray",
                border_visible=False)
        self.screen.index_grid.visible = False
        self.screen.value_grid.visible = False
        px = self.camera.pixelsize
        w, h = self.camera.width, self.camera.height
        self.screen.index_range.low_setting = -w/2*px
        # self.screen.index_range.high_setting
        self.screen.value_range.low_setting = -h/2*px
        self.screen.value_range.high_setting = h/2*px

        self.horiz = Plot(self.data,
                resizable="h", padding=0, height=100,
                bgcolor="lightgray", border_visible=False)
        self.horiz.value_mapper.range.low_setting = \
                -.1*self.camera.maxval
        self.horiz.index_range = self.screen.index_range
        self.vert = Plot(self.data, orientation="v",
                resizable="v", padding=0, width=100,
                bgcolor="lightgray", border_visible=False)
        for p in self.horiz, self.vert:
            p.index_axis.visible = False
            p.value_axis.visible = False
            p.index_grid.visible = True
            p.value_grid.visible = False
        self.vert.value_mapper.range.low_setting = \
                -.1*self.camera.maxval
        self.vert.index_range = self.screen.value_range

        #self.vert.value_range = self.horiz.value_range

        self.mini = Plot(self.data,
                width=100, height=100, resizable="", padding=0,
                bgcolor="lightgray", border_visible=False)
        self.mini.index_axis.visible = False
        self.mini.value_axis.visible = False
        self.label = PlotLabel(component=self.mini,
                overlay_position="inside left", font="modern 10",
                text=self.camera.text)
        self.mini.overlays.append(self.label)

        self.plots = GridPlotContainer(shape=(2,2), padding=0,
                spacing=(5,5), use_backbuffer=True,
                bgcolor="lightgray")
        self.plots.component_grid = [[self.vert, self.screen],
                                     [self.mini, self.horiz ]]

        self.screen.overlays.append(ZoomTool(self.screen,
            x_max_zoom_factor=1e2, y_max_zoom_factor=1e2,
            x_min_zoom_factor=0.5, y_min_zoom_factor=0.5,
            zoom_factor=1.2))
        self.screen.tools.append(PanTool(self.screen))
        self.plots.tools.append(SaveTool(self.plots,
            filename="bullseye.pdf"))

        self.asum = Plot(self.data,
                padding=0, height=100, bgcolor="lightgray",
                title="major axis", border_visible=False)
        self.bsum = Plot(self.data,
                padding=0, height=100, bgcolor="lightgray",
                title="minor axis", border_visible=False)
        for p in self.asum, self.bsum:
            p.value_axis.visible = False
            p.value_grid.visible = False
            p.title_font = "modern 10"
            p.title_position = "left"
            p.title_angle = 90
        # lock scales
        #self.bsum.value_range = self.asum.value_range
        #self.bsum.index_range = self.asum.index_range

        self.abplots = VPlotContainer(padding=20, spacing=20,
                use_backbuffer=True,bgcolor="lightgray",
                fill_padding=True)
        self.abplots.add(self.bsum, self.asum)

    def populate_plots(self):
        self.screenplot = self.screen.img_plot("img",
                xbounds="xbounds", ybounds="ybounds",
                interpolation="nearest",
                colormap=color_map_name_dict[self.palette],
                )[0]
        self.set_invert()
        self.camera.grid = self.screenplot.index
        self.gridm = self.screenplot.index_mapper
        t = ImageInspectorTool(self.screenplot)
        self.screen.tools.append(t)
        self.screenplot.overlays.append(ImageInspectorOverlay(
            component=self.screenplot, image_inspector=t,
            border_size=0, bgcolor="transparent", align="ur",
            tooltip_mode=False, font="modern 10"))

        self.horiz.plot(("x", "imx"), type="line", color="red")
        self.vert.plot(("y", "imy"), type="line", color="red")
        self.horiz.plot(("x", "gx"), type="line", color="blue")
        self.vert.plot(("y", "gy"), type="line", color="blue")
        self.asum.plot(("a", "ima"), type="line", color="red")
        self.bsum.plot(("b", "imb"), type="line", color="red")
        self.asum.plot(("a", "ga"), type="line", color="blue")
        self.bsum.plot(("b", "gb"), type="line", color="blue")

        for p in [("ell1_x", "ell1_y"), ("ell3_x", "ell3_y"),
                ("a_x", "a_y"), ("b_x", "b_y")]:
            self.screen.plot(p, type="line", color="green", alpha=.5)

        for r, s in [("x", self.horiz), ("y", self.vert),
                ("a", self.asum), ("b", self.bsum)]:
            for p in "0 p m".split():
                q = ("%s%s_mark" % (r, p), "%s_bar" % r)
                s.plot(q, type="line", color="green")

    def __del__(self):
        self.close()

    def close(self):
        self.camera.active = False

    @on_trait_change("palette")
    def set_colormap(self):
        p = self.screenplot
        m = color_map_name_dict[self.palette]
        p.color_mapper = m(p.value_range)
        self.set_invert()
        p.request_redraw()

    @on_trait_change("invert")
    def set_invert(self):
        p = self.screenplot
        if self.invert:
            a, b = self.camera.maxval, 0
        else:
            a, b = 0, self.camera.maxval
        p.color_mapper.range.low_setting = a
        p.color_mapper.range.high_setting = b

    # value_range seems to be updated after index_range, take this
    @on_trait_change("screen.value_range.updated")
    def set_range(self):
        if self.gridm is not None:
            #enforce data/screen aspect ratio 1
            sl, sr, sb, st = self.gridm.screen_bounds
            dl, db = self.gridm.range.low
            dr, dt = self.gridm.range.high
            #dsdx = float(sr-sl)/(dr-dl)
            dsdy = float(st-sb)/(dt-db)
            #dt_new = db+(st-sb)/dsdx
            if dsdy:
                dr_new = dl+(sr-sl)/dsdy
                self.gridm.range.x_range.high_setting = dr_new
        l, r = self.screen.index_range.low, self.screen.index_range.high
        b, t = self.screen.value_range.low, self.screen.value_range.high
        px = self.camera.pixelsize
        self.camera.roi = [l/px, b/px, (r-l)/px, (t-b)/px]

    @on_trait_change("camera.text")
    def set_text(self):
        if self.label is not None:
            self.label.text = self.camera.text


def main():
    import optparse
    p = optparse.OptionParser(usage="%prog [options]")
    p.add_option("-c", "--camera", default="first:",
            help="camera uri (none:, first:, guid:b09d01009981f9) "
                 "[%default]")
    p.add_option("-s", "--save", default="",
            help="save images accordint to strftime() "
                "format string, compressed npz format [%default]")
    p.add_option("-l", "--log",
            help="log output file [stderr]")
    p.add_option("-d", "--debug", default="info",
            help="log level (debug, info, warn, error, "
                "critical, fatal) [%default]")
    o, a = p.parse_args()
    logging.basicConfig(filename=o.log,
            level=getattr(logging, o.debug.upper()),
            format='%(asctime)s %(levelname)s %(message)s')
    b = Bullseye(o.camera)
    b.save_format = o.save
    b.configure_traits()
    b.close()

if __name__ == '__main__':
    main()
