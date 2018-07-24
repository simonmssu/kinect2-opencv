#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#


from lib.gui import *
from lib.cv.util import *


import sys
import cv2 as cv
import numpy as np

import collections

import time
import os

import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw

try:
    import matplotlib as mpl
    import matplotlib.pyplot as mplplt
    import matplotlib.figure as mplfig
    import matplotlib.backends.tkagg as mpltkagg
    import matplotlib.backends.backend_tkagg as mplbetkagg
    import mpl_toolkits.mplot3d as mpl3d
    import mpl_toolkits.mplot3d.art3d as mpl3dart
except ImportError:
    pass

try:
    import moderngl as mgl
    import flask
except ImportError:
    pass

try:
    from pylibfreenect2 import Freenect2, SyncMultiFrameListener
    from pylibfreenect2 import FrameType, Registration, Frame, FrameMap
    from pylibfreenect2 import createConsoleLogger, setGlobalLogger
    from pylibfreenect2 import LoggerLevel
    try:
        from pylibfreenect2 import OpenGLPacketPipeline
        pipeline = OpenGLPacketPipeline()
    except ImportError:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
except ImportError:
    pass


class AppFrame(BaseFrame):

    _WIN_WITH = 1040 + 80
    _WIN_HEIGHT = 585 + 85 + 80

    def __init__(self, root=None):

        BaseFrame.__init__(self, root)

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        w = self._WIN_WITH
        h = self._WIN_HEIGHT
        x = int((ws / 2) - (w / 2))
        y = int((hs / 2) - (h / 2))

        self._root = root
        self._fn = Freenect2()

        self._enabled = {True: tk.ACTIVE, False: tk.DISABLED}

        bottom = BaseFrame(self)
        bottom.pack(fill=tk.X, side=tk.BOTTOM)

        self._label = InfoLabel(bottom, text=' ', relief=tk.GROOVE, anchor=tk.W, justify=tk.LEFT, padding=(2, 2, 2, 2))
        self._label.pack(fill=tk.X, side=tk.BOTTOM)

        self._text = tk.Text(bottom, height=5)
        self._text.bind('<Key>', lambda e: 'break')
        self._text.pack(fill=tk.BOTH, side=tk.TOP)

        self._tab = TabFrame(self, tk.TOP, lambda o, n : self.switch(o, n))
        self._tab.pack(fill=tk.BOTH, expand=True)

        menu_main = tk.Menu(root)

        root.config(menu=menu_main)
        root.geometry('{}x{}+{}+{}'.format(w, h, x, y))
        root.minsize(width=400, height=300)

        menu_file = tk.Menu(menu_main, tearoff=0)
        menu_main.add_cascade(label="File", menu=menu_file)
        menu_file.add_command(label="Exit", command=self.quit)

        num_devs = self._fn.enumerateDevices()
        devs = [self._fn.getDeviceSerialNumber(i) for i in range(num_devs)]

        self._frames = []
        for srl in devs:
            self._frames.append(DeviceFrame(self._tab, self._fn, srl, lambda msg : self.show_info(msg)))
            self._tab.add(srl.decode('utf-8'), self._frames[-1])

        root.bind('<Escape>', lambda event: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.pack(fill=tk.BOTH, expand=True)

    def quit(self):
        self._root.destroy()
        self._root.quit()

    def show_info(self, msg, delay=2500):
        self._label.publish(msg, delay)

    def show_output(self, msg, maxlines=10):
        self._text.insert('end', msg)
        idx = int(self._text.index('end').split('.')[0]) - 2
        self._text.mark_set("insert", "{}.{}".format(idx, 0))
        self._text.see('insert')
        if idx > maxlines: self._text.delete("1.0", "2.0")
        self._text.update()

    @classmethod
    def switch(cls, odev, ndev):
        if odev is not None and ndev is not None:
            if type(odev) == DeviceFrame and type(ndev) == DeviceFrame:
                if odev != ndev:
                    odev.close()
                if not ndev.opened():
                    ndev.open()
                    ndev.play()
                elif not ndev.playing():
                    ndev.play()
                else:
                    ndev.stop()


class VideoUI(BaseFrame):

    def __init__(self, root, device, source):

        BaseFrame.__init__(self, root)

        self.master = device
        self.canvas = tk.Canvas(self)
        self.canvas.tkimg = None
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.source = source
        self.refresh()

    def __draw_grid(self, line_dist_horz, line_dist_vert):
        h = self.canvas.winfo_height()
        w = self.canvas.winfo_width()
        for x in range(line_dist_horz, w, line_dist_horz):
            self.canvas.create_line(x, 0, x, h, fill="lightsteelblue")
        for y in range(line_dist_vert, h, line_dist_vert):
            self.canvas.create_line(0, y, w, y, fill="lightsteelblue")

    def refresh(self):
        if self.winfo_viewable() and self.master.playing():
            img, _, _ = self.source()
            if img is not None:
                tkimg = PIL.ImageTk.PhotoImage(image=img)
                self.canvas.tkimg = tkimg
                self.canvas.config(width=tkimg.width(), height=tkimg.height())
                self.canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
                #self.__draw_grid(tkimg.width()//4+1, tkimg.height()//4+1)
        self.after(33, self.refresh)


class TrackerUI(BaseFrame):
    
    def __init__(self, root, device, source, publish, mode=1):

        BaseFrame.__init__(self, root)

        self.master = device
        self.canvas = tk.Canvas(self)
        self.canvas.tkimg = None
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        self.source = source    

        self._publish = publish        

        if mode == 1:  # "real" tracker, region of interest preset to the center of the screen
            self.tracker = cv.TrackerBoosting_create()
            self.roi = None

        elif mode == 2:  # track a yellow tennis ball and draw a trace
            self.lower = (30, 140, 55)
            self.upper = (60, 220, 255)
            self.pts = collections.deque(maxlen=32)
            dX, dY = 0, 0
            direction = ''
        
        self.refresh(mode)
            
    def __draw_grid(self, line_dist_horz, line_dist_vert):
        h = self.canvas.winfo_height()
        w = self.canvas.winfo_width()
        for x in range(line_dist_horz, w, line_dist_horz):
            self.canvas.create_line(x, 0, x, h, fill="lightsteelblue")
        for y in range(line_dist_vert, h, line_dist_vert):
            self.canvas.create_line(0, y, w, y, fill="lightsteelblue")

    def refresh(self, mode):
        if self.winfo_viewable() and self.master.playing():
            _, img, _ = self.source()
            if img is not None:
                
                if mode == 1:
                    if self.roi is None:        
                        self.roi = (len(img[0]) // 11 * 5, len(img) // 11 * 4.5, len(img[0]) // 11 * 1, len(img) // 11 * 2)            
                        ok = self.tracker.init(img, self.roi) # initialize tracker with first frame and bounding box
                        self._publish('tracker initialization {}'.format(ok))
                    else:
                        timer = cv.getTickCount()
                        ok, self.roi = self.tracker.update(img)
                        fps = cv.getTickFrequency() / (cv.getTickCount() - timer);
                        if ok:
                            p1 = (int(self.roi[0]), int(self.roi[1]))
                            p2 = (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3]))
                            cv.rectangle(img, p1, p2, None, 2)
                        else:
                            self._publish('tracking failure detected') 
                    img = PIL.Image.fromarray(img)
                    #dr = PIL.ImageDraw.Draw(img)
                    #dr.rectangle(((0,0),(10,10)), fill="black", outline = "blue")
                     
                    tkimg = PIL.ImageTk.PhotoImage(image=img)
                    self.canvas.tkimg = tkimg
                    self.canvas.config(width=tkimg.width(), height=tkimg.height())
                    self.canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
                    #self.__draw_grid(tkimg.width()//4+1, tkimg.height()//4+1)
                
                elif mode == 2:
                    bgr = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
                    blurred = cv.GaussianBlur(bgr, (11, 11), 0)
                    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
                    mask = cv.inRange(hsv, self.lower, self.upper)
                    mask = cv.erode(mask, None, iterations=2)
                    mask = cv.dilate(mask, None, iterations=2)
                    
                    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]
     
                    if len(cnts) > 0:                    
                        c = max(cnts, key=cv.contourArea)
                        ((x, y), radius) = cv.minEnclosingCircle(c)
                        M = cv.moments(c)
                        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                        if radius > 10:
                            cv.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                            cv.circle(img, center, 5, (0, 0, 255), -1)
                            self.pts.appendleft(center)
                             
                    for i in np.arange(1, len(self.pts)):
                        if self.pts[i - 1] is None or self.pts[i] is None:
                            continue
                        if len(self.pts) >= 10 and i == 1:
                            dX = self.pts[-10][0] - self.pts[i][0]
                            dY = self.pts[-10][1] - self.pts[i][1]
                            (dirX, dirY) = ('', '')
                            if np.abs(dX) > 20: dirX = 'west' if np.sign(dX) == 1 else 'east'
                            if np.abs(dY) > 20: dirY = 'north' if np.sign(dY) == 1 else 'south'
                            if dirX != '' and dirY != '': direction = "{}-{}".format(dirY, dirX)
                            else: direction = dirX if dirX != '' else dirY
                            self._publish(direction)            
                        thickness = int(np.sqrt(self.pts.maxlen / float(i + 1)) * 2.5)
                        cv.line(img, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)
    
                    img = PIL.Image.fromarray(img) 
                    tkimg = PIL.ImageTk.PhotoImage(image=img)
                    self.canvas.tkimg = tkimg
                    self.canvas.config(width=tkimg.width(), height=tkimg.height())
                    self.canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
                    #self.__draw_grid(tkimg.width()//4+1, tkimg.height()//4+1)
                    
        self.after(33, self.refresh, mode)

    
class DeviceFrame(TabFrame):

    persp = np.asarray([
            [0.75, 0,    120 + 15],
            [0,    0.99,        0],
            [0,    0,           1]], dtype=np.float32)

    def __init__(self, master, freenect, serial, publish):

        TabFrame.__init__(self, master)
        self.pack(fill=tk.BOTH, expand=True)

        self._freenect = freenect

        self._serial = serial
        self._device_index = self.__device_list_index()

        self._device = None
        self._listener = None

        self._opened = False
        self._playing = False

        self.frames = FrameMap()
        self.image_buffer = {'color': (None, None, None), 'ir': (None, None, None), 'depth': (None, None, None)}

        cam = TabFrame(self, tk.TOP)
        self.add(' Camera ', cam)

        color = VideoUI(cam, self, lambda : self.get_image_color())
        cam.add(' RGB ', color)
        color.canvas.bind("<Button-1>", lambda _ : self.capture())
         
        ir = VideoUI(cam, self, lambda : self.get_image_ir())
        cam.add(' IR ', ir)
        ir.canvas.bind('<Motion>', self.motion)
        ir.canvas.bind("<Button-1>", lambda _: self.capture())
 
        depth = VideoUI(cam, self, lambda : self.get_image_depth())
        cam.add(' Depth ', depth)
        depth.canvas.bind('<Motion>', self.motion)
        depth.canvas.bind("<Button-1>", lambda _: self.capture())

        filt = TabFrame(self, tk.TOP)
        self.add(' Filter ', filt)
        
        sob = VideoUI(filt, self, lambda : self.get_image_ir(filters=(sobel,)))
        filt.add(' Sobel ', sob)

        mas = VideoUI(filt, self, lambda : self.get_image_ir(filters=(masking,)))
        filt.add(' Masking ', mas)

        can = VideoUI(filt, self, lambda : self.get_image_ir(filters=(canny,)))
        filt.add(' Canny ', can)

        hou = VideoUI(filt, self, lambda : self.get_image_ir(filters=(hough,)))
        filt.add(' Hough ', hou)

        if 'matplotlib' in sys.modules:
            plot = TabFrame(self, tk.TOP)
            self.add(' 3D ', plot)
            plot_pc = PlotFrame(plot, self, (lambda : self.get_image_depth(), None), (920, 580, (0, 5000)))
            plot.add(' Point Cloud ', plot_pc)
            plot_dm = PlotFrame(plot, self, (lambda : self.get_image_depth(), lambda : self.get_image_ir()), (920, 580, (0, 5000)))
            plot.add(' Depthmap ', plot_dm)
            plot_cm = PlotFrame(plot, self, (lambda : self.get_image_depth(), lambda : self.get_image_color()), (920, 580, (0, 5000)))
            plot.add(' Colormap ', plot_cm)
            #if 'moderngl' in sys.modules:
            #    gl = TabFrame(plot, tk.TOP)
            #    plot.add(' OpenGL ', gl)

        track = TabFrame(self, tk.TOP)
        self.add(' Tracker ', track)

        boos = TrackerUI(track, self, lambda : self.get_image_color(), publish, 1)
        track.add(' Boosting ', boos)

        colo = TrackerUI(track, self, lambda : self.get_image_color(), publish, 2)
        track.add(' Color ', colo)

        #if 'flask' in sys.modules:
        #    stream = TabFrame(self, tk.TOP)
        #    self.add(' Stream ', stream)

        self._publish = publish
        self.refresh()

    def __device_list_index(self):
        num_devs = self._freenect.enumerateDevices()
        devs = [self._freenect.getDeviceSerialNumber(i) for i in range(num_devs)]
        return devs.index(self._serial)

    def motion(self, event):
        msg = " x={}px y={}px ".format(event.x, event.y)
        _, _, buffer = self.get_image_depth()
        if buffer is not None:
            w, h = buffer.shape[::-1]
            if buffer[event.y % h][event.x % w] > 0:
                msg = msg + " d={}mm".format(buffer[event.y % h][event.x % w])
        if self._publish is not None: self._publish(msg)
        return None

    def open(self):
        self._listener = SyncMultiFrameListener(FrameType.Color | FrameType.Ir | FrameType.Depth)
        self._device = self._freenect.openDevice(self._serial, pipeline=pipeline)
        device_index = self.__device_list_index()
        if self._device_index != device_index:  # keep track of changes in the device list
            self._device_index = device_index
            self._device.close()
            self._listener.release(self.frames)
            self.open()
            return
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.start()
        self._opened = True
        self._playing = False

    def opened(self):
        return self._opened

    def play(self):
        if not self._opened: return False
        self._playing = True
        return True

    def playing(self):
        return self._playing

    def stop(self):
        if not self._opened: return False
        self._playing = False
        return True

    def close(self):
        if not self._opened: return
        self._device.close()
        self._opened = False
        self._playing = False

    def refresh(self):
        if self._playing:
            self._listener.release(self.frames)
            self.image_buffer = {key: (None, None, None) for key in self.image_buffer}  # reset image buffer
            self.frames = self._listener.waitForNewFrame()
        self.after(30, self.refresh)

    def get_image_color(self, filters=()):
        color, _, _ = self.image_buffer['color']
        if color is not None: return self.image_buffer['color']
        color = self.frames['color']
        color = color.asarray(dtype=np.uint8)
        color = cv.resize(color, (1920 // 2, 1080 // 2))
        color = cv.cvtColor(color, cv.COLOR_BGRA2RGBA)
        #color = np.rot90(color)
        #color = np.rot90(color, 3)
        for f in filters: color, *_ = f(color) 
        return self.to_image('color', color)

    def get_image_ir(self, filters=()):
        ir, _, _ = self.image_buffer['ir']
        if ir is not None: return self.image_buffer['ir']
        ir = self.frames['ir']
        ir = ir.asarray(dtype=np.float32)[32:-32]  # remove 32 lines top and bottom (in rgb out of view)
        dsize = (1920 // 2, 1080 // 2)
        ir = cv.resize(ir, dsize)        
        ir = ir / 65535
        ir = np.asarray(ir * 255, dtype=np.uint8)
        ir = cv.warpPerspective(ir, DeviceFrame.persp, None, borderMode=cv.BORDER_CONSTANT, borderValue=255)
        for f in filters: ir, *_ = f(ir)    
        return self.to_image('ir', ir)

    def get_image_depth(self, d_min=0, d_max=5000, filters=()):
        depth, _, _ = self.image_buffer['depth']
        if depth is not None: return self.image_buffer['depth']
        depth = self.frames['depth']
        depth = depth.asarray(dtype=np.float32)[32:-32]  # remove 32 lines top and bottom (in rgb out of view)
        depth = cv.resize(depth, (1920 // 2, 1080 // 2))
        depth = cv.warpPerspective(depth, DeviceFrame.persp, None, borderMode=cv.BORDER_CONSTANT, borderValue=d_max)    
        buffer = depth.astype(int)
        buffer[buffer == d_max], buffer[buffer == d_min] = -1, -1
        depth = (depth - d_min) / (d_max - d_min)
        depth = np.asarray(depth * 255, dtype=np.uint8)   
        for f in filters: depth, *_ = f(depth)
        return self.to_image('depth', depth, buffer)

    def to_image(self, key, arr, arg=None):
        self.image_buffer[key] = (PIL.Image.fromarray(arr), arr, arg)
        return self.image_buffer[key]

    def capture(self):
        timezone = (int(-time.timezone / 3600) + time.daylight) % 25
        tz_abbr = 'ZABCDEFGHIKLMNOPQRSTUVWXY'
        timestamp = time.strftime('%Y%m%d' + tz_abbr[timezone] + '%H%M%S')
        self.save(self.get_image_color()[0], timestamp, 'color')
        self.save(self.get_image_ir()[0], timestamp, 'ir')
        self.save(self.get_image_depth()[0], timestamp, 'depth')

    @classmethod
    def save(cls, img, timestamp, name, ext='tiff'):
        scriptdir = os.path.dirname(__file__)
        timedir = os.path.join(scriptdir, 'capture', str(timestamp))
        if not os.path.exists(timedir): os.makedirs(timedir)
        file = os.path.join(timedir, name + '.' + ext)
        img.save(file, ext)


class PlotFrame(BaseFrame):

    _RES = 6

    def __init__(self, parent, root, source, xyz=(1, 1, (0, 1)), view=(5, 20)):
        
        BaseFrame.__init__(self, parent)

        self.master = root
        
        self.source = source[0]
        self.colors = source[1]  
        
        self.xyz = xyz
        self.view = view        

        self.fig = mplplt.figure()
        self.ax = mpl3d.Axes3D(self.fig)    
    
        self.canvas = tk.Canvas(master=self)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_canvas = mplbetkagg.FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)           
        
        self._interrupt = False
        self.bind("<Configure>", self.__configure)

        self.data = None
        self.colmap = None
        
        self.__init()
        self.__refresh()

    def __init(self):

        self.ax.remove()

        self.ax = mpl3d.Axes3D(self.fig)
        self.ax.view_init(azim=self.view[0], elev=self.view[1])
        
        self.ax.set_zlabel("y [px]")
        self.ax.set_ylabel("x [px]")
        self.ax.set_xlabel("z [mm]")

        self.ax.set_xlim(self.xyz[2][::-1])
        self.ax.set_ylim((0, self.xyz[0] - 1))
        self.ax.set_zlim((self.xyz[1] - 1, 0))

        if self.colors is None:
            self.ax.plot(xs=[], ys=[], zs=[], marker='.', linestyle='')
        else:
            self.ax.scatter(xs=[], ys=[], zs=[], marker='.', cmap='jet')
                    
    def __configure(self, event):
        self._interrupt = True

    def __refresh(self, draw=False):
 
        r = PlotFrame._RES
        d = -75 * r + 500
        s = 0.75 * r
 
        if self.winfo_viewable() and self.master.playing():
 
            if not draw:
                 
                _, _, depmap = self.source()

                self.colmap = depmap // 255  # gray
                if self.colors is not None:
                    _, self.colmap, _ = self.colors()
                    self.colmap = self.colmap / 255  # rgb
                
                a = len(depmap) // r
                b = len(depmap[0]) // r
                i = -1
                
                x, y, z, c = [None]*a*b, [None]*a*b, [None]*a*b, [None]*a*b
                for py in range(a):
                    for px in range(b):
                        i += 1
                        iy, ix = py * r, px * r
                        if depmap[iy][ix] < 0: continue  # value in valid range?
                        x[i] = ix
                        y[i] = iy
                        z[i] = depmap[iy][ix]
                        c[i] = self.colmap[iy][ix]

                result = (item for item in zip(z, x, y, c) if item[0] is not None)
                z, x, y, c = zip(*result)
                
                result = sorted(zip(z, x, y, c), reverse=True)  # sort by color                                
                self.data = zip(*result)  # z, x, y, c         
                
            else:
 
                if self._interrupt:
                    self._interrupt = False
                    self.after(d // 2, self.__refresh)
                    return

                z, x, y, c = self.data                
                
                #x, y, z = np.broadcast_arrays(*[np.ravel(np.ma.filled(t, np.nan)) for t in [x, y, z]])    
                #points._offsets3d = (z, x, y)  # positions
                #points._sizes = [s] * len(c)   # sizes set_sizes()
                #points.set_array(np.array(c))  # colors setFacecolor(), set_edgecolor()

                if self.colors is None:
                    for lines in self.ax.lines:
                        lines.remove()
                    self.ax.plot(xs=z, ys=x, zs=y, marker='.', linestyle='', c='black', markersize=s)
                else:
                    for child in self.ax.get_children():
                        if isinstance(child, mpl3dart.Path3DCollection):
                            child.remove()
                    self.ax.scatter(xs=z, ys=x, zs=y, marker='.', cmap='jet', s=s, c=c)
                    mpl.colors._colors_full_map.cache.clear()  # avoid memory leak by clearing the cache
                
                self.__draw()  
            
            self.after(d, self.__refresh, not draw)
            return

        self.after(d, self.__refresh, False)

    def __draw(self):      
        self.fig_canvas.draw()
        fx, fy, fw, fh = self.fig.bbox.bounds
        img = tk.PhotoImage(master=self.canvas, width=int(fw), height=int(fh))
        self.canvas.create_image(0, 0, image=img)
        mpltkagg.blit(img, self.fig_canvas.get_renderer()._renderer, colormode=2)
