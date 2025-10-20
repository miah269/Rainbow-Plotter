import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from datetime import datetime as dt
from skimage import color as skcolor
from pathlib import Path
import xml.etree.ElementTree as ET
import shapes as shapes_pkg
from shapes import point_generator
import cubicsuperpath
import cspsubdiv
import simpletransform
import re
from optimise import optimise_path, get_total_distance
from subprocess import Popen
import copy
import psutil
import tracemalloc

APP_TITLE = "SVG to GCode Converter"
APP_SIZE = "600x400"

debug = False

 ## CONFIGURATION PARAMETERS ## from config.py
"""G-code emitted at the start of processing the SVG file"""
preamble = "G90"
"""G-code emitted at the end of processing the SVG file"""
postamble = "(postamble)"
"""G-code emitted before processing a SVG shape"""
shape_preamble = "(shape preamble)"
#shape_preamble = "Z0"
"""G-code emitted after processing a SVG shape"""
shape_postamble = "(shape postamble)"
#shape_postamble = "Z100)"
"""Print bed width in mm"""
bed_max_x = 345
"""Print bed height in mm"""
bed_max_y = 215
""" 
Used to control the smoothness/sharpness of the curves.
Smaller the value greater the sharpness. Make sure the
value is greater than 0.1
"""
smoothness = 0.02
""" height that the z axis will use to travel between strokes """
zTravel = 550
""" height that the z axis will use to draw """
zDraw = 0
""" feed rate """
feed_rate = 1000
""" decimal precision of gcode"""
precision = 2
""" scale gcode to fit bed size"""
auto_scale = False
""" optimize path - slow for large files"""
do_optimise = True
# How closely to approximate curves, in millimetres on paper.
# 0.25–0.5 mm is a good default for pen plotters.
curve_flatness_mm = 0.35

PENS = [
        {"name": "Black",   "rgb": (0, 0, 0),        "tool": 1},
        {"name": "Red",     "rgb": (255, 0, 0),      "tool": 2},
        {"name": "Blue",    "rgb": (0, 0, 255),     "tool": 3},
        {"name": "Green",   "rgb": (0, 255, 0),    "tool": 4},
        {"name": "Orange",  "rgb": (255, 165, 0),   "tool": 5},
        {"name": "Purple", "rgb": (128, 0, 128),      "tool": 6},
    ]

_PROC = psutil.Process()

def _rss_mib() -> float:
    """Current process RSS in MiB."""
    return _PROC.memory_info().rss / (1024**2)

def parse_inline_style(style: str):
    styles = {}
    if not style:
        return styles
    for item in style.split(';'):
        if ':' in item:
            k, v = item.split(':', 1)
            styles[k.strip().lower()] = v.strip()
    return styles

def parse_colour(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip().lower()
    if s == "none":
        return None
    if s.startswith('#'):
        hx = s[1:]
        if len(hx) == 3:
            hx = ''.join(c*2 for c in hx)
        if len(hx) == 6:
            return '#' + hx
        return None
    if s.startswith('rgb(') and s.endswith(')'):
        nums = re.findall(r'[-+]?\d+\.?\d*', s)
        if len(nums) >= 3:
            r, g, b = (max(0, min(255, int(float(v)))) for v in nums[:3])
            return f'#{r:02x}{g:02x}{b:02x}'
        return None
    basic = {
        'black':'#000000','white':'#ffffff','red':'#ff0000','blue':'#0000ff',
        'green':'#00ff00','orange':'#ff8000','purple':'#800080','gray':'#808080',
        'grey':'#808080','yellow':'#ffff00','cyan':'#00ffff','magenta':'#ff00ff'
    }
    return basic.get(s)

def _hex_to_rgb255(hexstr: str) -> tuple[int,int,int]:
    hexstr = hexstr.lstrip('#')
    return int(hexstr[0:2],16), int(hexstr[2:4],16), int(hexstr[4:6],16)

def _rgb_to_hex(rgb):  # (r,g,b) -> '#rrggbb'
    return '#%02x%02x%02x' % tuple(int(v) for v in rgb)

def _rgb255_to_lab(rgb: tuple[int,int,int]) -> np.ndarray:
    """(0..255,0..255,0..255) -> Lab(3,) float"""
    arr = np.array(rgb, dtype=float)[None, None, :] / 255.0
    return skcolor.rgb2lab(arr)[0, 0, :]  # 3-vector

def timer(t, label):
    duration = dt.now() - t
    duration = duration.total_seconds()
    print("{} took {}".format(label, duration))
    return duration



def get_shapes(svg_text: str, use_fill_if_no_stroke: bool=False):
        t1 = dt.now()
        root = ET.fromstring(svg_text)

        parent_map = {child: parent for parent in root.iter() for child in parent}

        def cumulative_transform(el, parent_map):
            """
            Compose transforms from all ancestors (including el),
            parent-first order.
            """
            m = [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]]
            chain = []
            cur = el
            while cur is not None:
                t = cur.get('transform')
                if t:
                    chain.append(t)
                cur = parent_map.get(cur)

            for t in reversed(chain):
                if isinstance(t, str):
                   m = simpletransform.parseTransform(t, m)
                else:
                    m = simpletransform.composeTransform(m, t)
            return m

        def cascaded_style(el):
            """
            Return a dict of computed style properties for `el`,
            taking inline style="" and attributes from el and its ancestors.
            Closest (child) wins.
            """
            out = {}
            cur = el
            while cur is not None:
                st = parse_inline_style(cur.get('style', ''))
                for key in ('stroke','fill','opacity','stroke-opacity','fill-opacity','display','visibility'):
                    if key not in out:
                        val = st.get(key)
                        if val is None:
                            val = cur.get(key)
                        if val is not None:
                            out[key] = val
                cur = parent_map.get(cur)
            return out

        def parse_len(s: str):
            m = re.match(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([a-z%]*)\s*$', s or '')
            if not m:
                return None, ''
            return float(m.group(1)), m.group(2).lower()
        
        def unit2mm(unit: str) -> float:
            if unit in ('mm', ''): return 1.0
            if unit == 'cm': return 10.0
            if unit == 'in': return 25.4
            if unit == 'px': return 25.4 / 96.0  # assume 96dpi
            if unit == 'pt': return 25.4 / 72.0
            return 1.0 # default to mm if unknown
        
        w_attr = root.get('width') 
        h_attr = root.get('height')
        vb_attr = root.get('viewBox')

        w_val, w_unit = parse_len(w_attr) if w_attr else (None, '')
        h_val, h_unit = parse_len(h_attr) if h_attr else (None, '')
        vb_x = vb_y = 0.0
        vb_w = vb_h = None
        if vb_attr:
            parts = [float(p) for p in re.split(r'[,\s]+', vb_attr.strip()) if p]
            if len(parts) == 4:
                vb_x, vb_y, vb_w, vb_h = parts

        w_mm = w_val * unit2mm(w_unit) if w_val is not None else None
        h_mm = h_val * unit2mm(h_unit) if h_val is not None else None

        if vb_w and vb_h and (w_mm is not None) and (h_mm is not None):
        # viewBox present: user_units -> mm via mm_per_unit
            mm_per_u_x = w_mm / vb_w
            mm_per_u_y = h_mm / vb_h
        elif vb_w and vb_h and (w_mm is None or h_mm is None):
            # viewBox but no physical size: assume 96 dpi (px)
            px2mm = 25.4/96.0
            mm_per_u_x = px2mm
            mm_per_u_y = px2mm
        elif (w_mm is not None) and (h_mm is not None) and not vb_w:
            # no viewBox: assume coordinates are px @ 96dpi
            px2mm = 25.4/96.0
            mm_per_u_x = px2mm
            mm_per_u_y = px2mm
        else:
            raise ValueError("SVG missing usable width/height/viewBox for scaling")
        
        # choose the smaller mm->user scale so the tolerance isn’t overly tight
        mm_per_u_min = max(min(mm_per_u_x, mm_per_u_y), 1e-9)
        flat_user = max(0.1, curve_flatness_mm / mm_per_u_min)  # never let it go below ~0.1 user-unit

        svg_shapes = {'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'path'}
        shapes = []
        
        for elem in root.iter():
            try:
                _, tag_suffix = elem.tag.split('}')
            except ValueError:
                continue
            if tag_suffix not in svg_shapes:
                continue
            
            style_map = cascaded_style(elem)
            disp = (style_map.get('display', '') or '').strip().lower()
            vis = (style_map.get('visibility', '') or '').strip().lower()
            if disp == 'none' or vis == 'hidden':
                continue

            #style = parse_inline_style(elem.get('style', ''))
            #stroke = parse_colour(style.get('stroke', elem.get('stroke')))
            #fill = parse_colour(style.get('fill', elem.get('fill')))
            stroke = parse_colour(style_map.get('stroke'))
            fill = parse_colour(style_map.get('fill'))

            #if stroke is None and use_fill_if_no_stroke:
                #stroke = fill
            
            #opacity
            def _f(x, default=1.0):
                try:
                    return float(x)
                except:
                    return default
                
            #stroke_opac = _f(style.get('stroke-opacity', elem.get('stroke-opacity', 1)))
            #overall_opac = _f(style.get('opacity', elem.get('opacity', 1)))
            stroke_opac = _f(style_map.get('stroke-opacity'), 1.0)
            fill_opac = _f(style_map.get('fill-opacity'), 1.0)
            overall_opac = _f(style_map.get('opacity'), 1.0)

            # if stroke is None or stroke_opac <= 0 or overall_opac <= 0:
              #  continue #Not visible
            from_fill = False
            if (stroke is None or stroke_opac <= 0 or overall_opac <= 0) and use_fill_if_no_stroke and (fill is not None) and (fill_opac > 0):
                stroke = fill
                from_fill = True

            if (stroke is None) or (overall_opac <= 0):
                continue

            shape_class = getattr(shapes_pkg, tag_suffix)
            shape_obj = shape_class(elem)
            d = shape_obj.d_path()
            m = cumulative_transform(elem, parent_map)
            if not d:
                continue
            polys = path_to_polyline(d, m, flat_user)
            for poly in polys:
                coords = [((x - vb_x) * mm_per_u_x, (y - vb_y) * mm_per_u_y) for (x, y) in poly]
                if coords:
                    shapes.append({'coords': coords, 'stroke': stroke})
            #coords = []
            #for x, y in point_generator(d, m, smoothness):
             #   coords.append((x * mm_per_u_x, y * mm_per_u_y))
            #if coords:
            #    shapes.append({'coords': coords, 'stroke': stroke})              

        timer(t1, "parsing gcode")
        return shapes

def path_to_polyline(d: str, mat, flatness_user: float):
    if not d:
        return []
    
    # Parse path into CubicSuperPath (list of subpaths)
    p = cubicsuperpath.parsePath(d)

    # Apply transform if present
    if mat is not None:
        simpletransform.applyTransformToPath(mat, p)

    polylines = []
    for subpath in p:
        original = copy.deepcopy(subpath)
        tol = float(flatness_user)

        for _ in range(5):
            sp = copy.deepcopy(original)
            try:
                cspsubdiv.subdiv(sp, tol)
                # Build a clean polyline from the subpath's nodes
                poly = []
                # first point of subpath (the "move-to")
                sx, sy = sp[0][0]
                poly.append((float(sx), float(sy)))
                # end points of each curve segment
                for seg in sp:
                    ex, ey = seg[2]
                    pt = (float(ex), float(ey))
                    if pt != poly[-1]:
                        poly.append(pt)
                if len(poly) >= 2:
                    polylines.append(poly)
                break
            except RecursionError:
                tol *= 2.0
    return polylines



def scale_shapes(shapes, target_w_mm, target_h_mm, margin_mm=0.0, invert_y=True):
    if not shapes:
        return []
    
    xs = [p[0] for poly in shapes for p in poly['coords']]
    ys = [p[1] for poly in shapes for p in poly['coords']]
    if not xs or not ys:
        return []
    
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    bw, bh = maxx - minx, maxy - miny
    if bw <= 0 or bh <= 0:
        return []
    
    avail_w = max(1e-9, target_w_mm - 2*margin_mm)
    avail_h = max(1e-9, target_h_mm - 2*margin_mm)
    scale = min(avail_w / bw, avail_h / bh)

    out = []
    for poly in shapes:
        new_poly = []
        for x, y in poly['coords']:
            X = (x - minx) * scale
            Y = (y - miny) * scale
            if invert_y:
                Y = (bh * scale) - Y
            X += margin_mm
            Y += margin_mm
            new_poly.append((X, Y))
        out.append({'coords': new_poly, 'stroke': poly['stroke']})
    return out

def assign_to_pens(shapes, pens=PENS, tolerance=100.0):
    #pens in lab space
    pen_labs = []
    for p in pens:
        pen_labs.append(_rgb255_to_lab(tuple(p['rgb'])))
    pen_labs = np.stack(pen_labs, axis=0) 

    by_pen = {p['tool']: [] for p in pens}
    unmapped = []

    lab_cache = {}

    #assign shapes to pens
    for shape in shapes:
        hexcol = shape['stroke']
        if hexcol not in lab_cache:
            lab_cache[hexcol] = _rgb255_to_lab(_hex_to_rgb255(hexcol))
        lab = lab_cache[hexcol]
    
        euclidian = np.linalg.norm(pen_labs - lab[None, :], axis=1)
        best_i = int(np.argmin(euclidian))
        best_d = float(euclidian[best_i])

        if best_d > tolerance:
            unmapped.append(shape)
            continue

        tool = pens[best_i]['tool']
        by_pen[tool].append(shape['coords'])

    return by_pen, unmapped

# optimise paths for each pen separately 
def optimise_per_pen(by_pen: dict) -> dict:
    if not do_optimise:
        return by_pen
    out = {}
    for tool, shapes in by_pen.items():
        if not shapes:
            out[tool] = []
            continue
        try:
            pre = get_total_distance(shapes.copy())
        except Exception:
            pre = None
        new_order = optimise_path(shapes.copy())
        try:
            post = get_total_distance(new_order.copy())
        except Exception:
            post = None
        if pre and post:
            print(f"Tool {tool}: optimise {pre:.1f} -> {post:.1f} mm")
        out[tool] = new_order
    return out

def g_string(x, y, prefix="G1", p=3):
    return f"{prefix} X{x:.{p}f} Y{y:.{p}f}"
    
def pen_up_down(z_up=True):
    if z_up:
        return f"M3 S{zTravel}"
    else:
        return f"M3 S{zDraw}"
    
def shapes_2_gcode(by_pen: dict, pens=PENS, feed_rate=1000):
    t1 = dt.now()
    try:
        with open("header.txt", "r", encoding="utf-8") as h:
            header = h.read().strip()
    except FileNotFoundError:
        header = preamble

    commands = [header, f"F{feed_rate}", pen_up_down(z_up=True)]

    def tool_change(tool_num, name):
        return [f"(TOOL {tool_num} - {name})", f"M6 T{tool_num}"]
    
    for p in pens:
        tool = p['tool']
        shapes = by_pen.get(tool, [])
        if not shapes:
            continue
        commands.append(pen_up_down(z_up=True))
        commands += [""] + tool_change(tool, p['name']) + [""]

        for i in shapes:
            if not i:
                continue
            x0, y0 = i[0]
            commands += ['', shape_preamble, ""] 
            commands.append(g_string(x0, y0, "G0", p=precision))
            commands.append(pen_up_down(z_up=False))
            commands.append(f"G4 P1")  # wait for pen down
            for x, y in i:
                #commands.append(pen_up_down(z_up=False))
                commands.append(g_string(x, y, "G1", p=precision))
            commands.append(pen_up_down(z_up=True))
            commands.append(f"G4 P1")  # wait for pen up
            commands.append(g_string(i[-1][0], i[-1][1], "G0", p=precision))
            commands += ["", shape_postamble, ""]

    commands += [" ",postamble, ""]
    commands += ["(home)", g_string(0, 0, "G0", p=precision), "G0 X0 Y0"]

    timer(t1, "shapes_2_gcode   ")
    return commands


def _secs(t0): 
    return (dt.now() - t0).total_seconds()

def image_to_gcode(
    svg_text: str,
    height_mm: float = 60,
    width_mm: float = 40,
    feed_xy: float = 1000.0,
) -> str:
    
    #Start timer and memory tracking
    t_total = dt.now()
    tracemalloc.start()
    peak_rss = _rss_mib()

    #Parse svg to find coloured paths in mm
    t = dt.now()
    shapes = get_shapes(svg_text, use_fill_if_no_stroke=True)
    peak_rss = max(peak_rss, _rss_mib())
    print(f"[timing] get_shapes: {_secs(t):.3f}s")
    if not shapes:
        raise ValueError("No shapes found in SVG")
    
    #Scale to requested size
    t = dt.now()
    shapes = scale_shapes(shapes, width_mm, height_mm, margin_mm=0.0, invert_y=True)
    peak_rss = max(peak_rss, _rss_mib())
    print(f"[timing] scale_shapes: {_secs(t):.3f}s")

    #Map shapes to pens
    t = dt.now()
    by_pen, unmapped = assign_to_pens(shapes, pens=PENS, tolerance=100.0)
    peak_rss = max(peak_rss, _rss_mib())
    print(f"[timing] assign_to_pens: {_secs(t):.3f}s")
    if unmapped:
        print(f"Warning: {len(unmapped)} shapes not mapped to any pen")

    #Optimise paths per pen
    t = dt.now()
    by_pen = optimise_per_pen(by_pen)
    peak_rss = max(peak_rss, _rss_mib())
    print(f"[timing] optimise_per_pen: {_secs(t):.3f}s")
    
    #Write gcode per pen
    t = dt.now()
    commands = shapes_2_gcode(by_pen, pens=PENS, feed_rate=feed_xy)
    peak_rss = max(peak_rss, _rss_mib())
    print(f"[timing] shapes_2_gcode: {_secs(t):.3f}s")

    cur_rss = _rss_mib()
    _, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"[timing] TOTAL (SVG -> G-code): {_secs(t_total):.3f}s")
    print(f"[memory] Peak RSS during conversion: {peak_rss:.1f} MiB")
    print(f"[memory] Final RSS: {cur_rss:.1f} MiB")
    print(f"[memory] Peak Python heap (tracemalloc): {peak_tracemalloc/(1024**2):.1f} MiB")
    return "\n".join(commands)  

## ------------------------------ USER INTERFACE -------------------------------------------- ##

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(APP_SIZE)
        self.img = None
        self.preview = None
        self._build_ui()

    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_open = ttk.Button(toolbar, text="Upload Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT)

        self.btn_save = ttk.Button(toolbar, text="Convert & Save GCode", command=self.save_gcode, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=6)

        ttk.Button(toolbar, text="Quit", command=self.destroy).pack(side=tk.RIGHT)

        # Parameters panel
        params = ttk.LabelFrame(self, text="Parameters")
        params.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.var_height = tk.DoubleVar(value=60.0)
        self.var_width = tk.DoubleVar(value=40.0)
        self.var_feed = tk.DoubleVar(value=1000.0)

        row = 0
        def add_row(label, widget):
            nonlocal row
            ttk.Label(params, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            widget.grid(row=row, column=1, sticky="ew", padx=6, pady=4)
            row += 1

        params.columnconfigure(1, weight=1)

        add_row("Height (mm)", ttk.Entry(params, textvariable=self.var_height))
        add_row("Width (mm)", ttk.Entry(params, textvariable=self.var_width))
        add_row("Feed XY (mm/min)", ttk.Entry(params, textvariable=self.var_feed))
        

        row += 1
        btn_grbl = ttk.Button(params, text="Open GRBL Plotter", command=self.open_grbl)
        btn_grbl.grid(row=row, column=0, columnspan=2, sticky="ew", padx=6, pady=10)

        # Preview panel
        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.preview_canvas = tk.Canvas(right, bg="white", highlightthickness=1, highlightbackground="#ddd")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas.bind("<Configure>", lambda e: self.update_preview())

    def open_grbl(self):
        Popen("C:\Program Files (x86)\GRBL-Plotter\GRBL-Plotter.exe")

    def open_image(self):
        fn = filedialog.askopenfilename(
            title="Choose an SVG image",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")],
        )
        if not fn:
            return
        
        #ext = os.path.splitext(fn)[1].lower()
        try:
            self.svg_text = Path(fn).read_text(encoding="utf-8")
            self.img = None
            self.update_preview()
            self.btn_save.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{e}")
            return

    def update_preview(self):
        #Check for loaded image
        if not getattr(self, "svg_text", None):
            if hasattr(self, "preview_canvas"):
                self.preview_canvas.delete("all")
                self.preview_canvas.create_text(10,10, anchor="nw", text="No svg loaded", fill="#666")
            return

        canvasw = max(1, self.preview_canvas.winfo_width())
        canvash = max(1, self.preview_canvas.winfo_height())
        pad = 10

        #Parse shapes
        try:
            shapes = get_shapes(self.svg_text, use_fill_if_no_stroke=True)
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(10, 10, anchor="nw", text=f"Parse failed: {e}", fill="red")
            return
        if not shapes:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(10, 10, anchor="nw", text="No drawable geometry", fill="#666")
            return
        
        #Scale shapes to fit canvas
        target_h = float(self.var_height.get())
        target_w = float(self.var_width.get())
        shapes = scale_shapes(shapes, target_w, target_h, margin_mm=0.0, invert_y=False)

        #Match colours to pens
        by_pen, unmapped = assign_to_pens(shapes, pens=PENS, tolerance=100.0)

        #mm to px mapping for display
        sp = min((canvasw - 2*pad) / max(target_w, 1e-9), (canvash - 2*pad) / max(target_h, 1e-9))
        def mm2px(x, y):
            return pad + x * sp, pad + y * sp
        
        #draw on canvas
        self.preview_canvas.delete("all")
        self.preview_canvas.create_rectangle(pad, pad, pad + target_w*sp, pad + target_h*sp, outline="#ddd")
        pen_hex = {p['tool']: _rgb_to_hex(p['rgb']) for p in PENS} #convert PENS to hex

        for p in PENS:
            tool = p['tool']
            shapes = by_pen.get(tool, [])
            if not shapes:
                continue
            colour = pen_hex[tool]

            for shape in shapes:
                if len(shape) < 2:
                    continue
                flat = []
                for x, y in shape:
                    X, Y = mm2px(x, y)
                    flat.extend((X, Y))
                try:
                    self.preview_canvas.create_line(*flat, fill=colour, width=2.0)
                except Exception:
                    for i in range(len(shape)-1):
                        x1, y1 = mm2px(*shape[i])
                        x2, y2 = mm2px(*shape[i+1])
                        self.preview_canvas.create_line(x1, y1, x2, y2, fill=colour, width=2)

        if unmapped:
            self.preview_canvas.create_text(
                canvasw - 10, canvash - 10, anchor="se",
                text=f"{len(unmapped)} shape(s) unmapped",
                fill="#b00", font=("TkDefaultFont", 8)
            )

    def save_gcode(self):
        if not getattr(self, "svg_text", None):
            messagebox.showwarning("No SVG", "Load an SVG first.")
            return

        height = float(self.var_height.get())
        width = float(self.var_width.get())
        feed = float(self.var_feed.get())

        try:
            gcode = image_to_gcode(
                self.svg_text,
                height_mm=height,
                width_mm=width,
                feed_xy=feed,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate G-code:\n{e}")
            return

        ts = dt.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"image2gcode_{ts}.gcode"
        out_fn = filedialog.asksaveasfilename(
            title="Save G-code",
            defaultextension=".gcode",
            initialfile=default_name,
            filetypes=[("G-code", "*.gcode;*.nc;*.ngc;*.txt"), ("All files", "*.*")],
        )
        if not out_fn:
            return

        try:
            with open(out_fn, "w", encoding="utf-8") as f:
                f.write(gcode)
        except Exception as e:
            messagebox.showerror("Error", f"Could not write file:\n{e}")
            return

        messagebox.showinfo("Done", f"G-code saved to:\n{out_fn}")


def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
