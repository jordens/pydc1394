# -*- coding: utf-8 -*-
# Copyright 2010 Robert Jordens <robert@joerdens.org>
#
# This file is part of pydc1394.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
# MA  02110-1301  USA

from __future__ import (print_function, unicode_literals, division,
        absolute_import)


from threading import Thread, Condition, Event
try:
    from queue import Queue, Full
except ImportError:
    from Queue import Queue, Full

from . import Camera


__all__ = ["ThreadedCamera"]


class ThreadedCamera(Camera):
    def start(self, queue=0, mark_corrupt=True):
        """
        Start the handling of acquired frames.

        Use :meth:`start_capture` and :meth:`start_video` before calling
        this method to trigger the transmission. 
        
        If ``queue=0`` (the default) then no queue will be used and only
        the most recent image is available as :meth:`current_image`.
        Otherwise the frames are also stored in a queue of size
        ``queue`` (``queue=0`` corresponds to an infinitely large
        queue) and can be obtained sequentially using
        :meth:`next_image`.

        If ``mark_corrupt=True``, the frames returned have a corruption
        marker attached.

        End the acquisition by calling :meth:`stop`.
        """
        self.mark_corrupt = mark_corrupt
        self.abort_thread = Event()
        self.new_image = Condition()
        if queue == 1:
            self.queue = None
        else:
            self.queue = Queue(queue)
        self.acquisition = Thread(target=self.run)
        self.acquisition.start()

    def run(self):
        """
        Called in the acquisition thread.

        Acquires images, copies them, adds them to :attr:`queue` and saves
        the most recent as :attr:`current`.
        """
        while not self.abort_thread.is_set():
            img = self.dequeue(poll=False)
            if img is None:
                continue
            img_copy = img.copy()
            if self.mark_corrupt:
                img_copy.corruption_marker = img.corrupt
            img.enqueue() # need to enqueue in the same thread
            img = img_copy
            with self.new_image:
                self.current = img
                if self.queue:
                    try:
                        self.queue.put_nowait(self.current)
                    except Full:
                        pass # drop the frame
                self.new_image.notify_all()

    def next_image(self):
        """
        The next image from the queue of acquired images.

        Blocks if there is none.
        """
        return self.queue.get()

    def current_image(self, new=False):
        """
        The most recently acquired image.
        
        If ``new=True``, wait for a new one to come in and return that one.
        """
        with self.new_image:
            if new:
                self.new_image.wait()
            return self.current

    def stop(self):
        """
        Stop the handling of acuired frames.

        Use :meth:`stop_video` and :meth:`stop_capture` to halt the
        camera.
        """
        self.abort_thread.set()
        self.acquisition.join()
