#!/usr/bin/env python

# import logging
# import traceback
# import xml.etree.ElementTree as ET
# from . import simplepath
# from . import simpletransform
# from . import cubicsuperpath
# from . import cspsubdiv
# from .bezmisc import beziersplitatt

import logging
import traceback
import xml.etree.ElementTree as ET
import simplepath
import simpletransform
import cubicsuperpath
import cspsubdiv
from bezmisc import beziersplitatt
import re

def _f(x: str) -> float:
    if x is None:
        return 0.0
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', x)
    return float(m.group(0)) if m else 0.0

class svgshape(object):
    
    def __init__(self, xml_node):
        self.xml_node = xml_node 
 
    def d_path(self):
        raise NotImplementedError

    def transformation_matrix(self):
        t = self.xml_node.get('transform')
        return simpletransform.parseTransform(t) if t is not None else None

    def svg_path(self):
        return "<path d=\"" + self.d_path() + "\"/>"

    def __str__(self):
        return self.xml_node        

class path(svgshape):
     def __init__(self, xml_node):
        super(path, self).__init__(xml_node)

        if not self.xml_node == None:
            path_el = self.xml_node
            self.d = path_el.get('d')
        else:
            self.d = None 
            logging.error("path: Unable to get the attributes for %s", self.xml_node)

     def d_path(self):
        return self.d     

class rect(svgshape):
  
    def __init__(self, xml_node):
        super(rect, self).__init__(xml_node)

        if not self.xml_node == None:
            rect_el = self.xml_node
            self.x  = _f(rect_el.get('x')) if rect_el.get('x') else 0
            self.y  = _f(rect_el.get('y')) if rect_el.get('y') else 0
            self.rx = _f(rect_el.get('rx')) if rect_el.get('rx') else 0
            self.ry = _f(rect_el.get('ry')) if rect_el.get('ry') else 0
            self.width = _f(rect_el.get('width')) if rect_el.get('width') else 0
            self.height = _f(rect_el.get('height')) if rect_el.get('height') else 0
        else:
            self.x = self.y = self.rx = self.ry = self.width = self.height = 0
            logging.error("rect: Unable to get the attributes for %s", self.xml_node)

    def d_path(self):
        a = list()
        a.append( ['M ', [self.x, self.y]] )
        a.append( [' l ', [self.width, 0]] )
        a.append( [' l ', [0, self.height]] )
        a.append( [' l ', [-self.width, 0]] )
        a.append( [' Z', []] )
        return simplepath.formatPath(a)     

class ellipse(svgshape):

    def __init__(self, xml_node):
        super(ellipse, self).__init__(xml_node)

        if not self.xml_node == None:
            ellipse_el = self.xml_node
            self.cx  = _f(ellipse_el.get('cx')) if ellipse_el.get('cx') else 0
            self.cy  = _f(ellipse_el.get('cy')) if ellipse_el.get('cy') else 0
            self.rx = _f(ellipse_el.get('rx')) if ellipse_el.get('rx') else 0
            self.ry = _f(ellipse_el.get('ry')) if ellipse_el.get('ry') else 0
        else:
            self.cx = self.cy = self.rx = self.ry = 0
            logging.error("ellipse: Unable to get the attributes for %s", self.xml_node)

    def d_path(self):
        x1 = self.cx - self.rx
        x2 = self.cx + self.rx
        p = 'M %f,%f ' % ( x1, self.cy ) + \
            'A %f,%f ' % ( self.rx, self.ry ) + \
            '0 1 0 %f,%f ' % ( x2, self.cy ) + \
            'A %f,%f ' % ( self.rx, self.ry ) + \
            '0 1 0 %f,%f' % ( x1, self.cy )
        return p

class circle(ellipse):
    def __init__(self, xml_node):
        super(circle, self).__init__(xml_node)

        if not self.xml_node == None:
            circle_el = self.xml_node
            self.cx  = _f(circle_el.get('cx')) if circle_el.get('cx') else 0
            self.cy  = _f(circle_el.get('cy')) if circle_el.get('cy') else 0
            self.rx = _f(circle_el.get('r')) if circle_el.get('r') else 0
            self.ry = self.rx
        else:
            self.cx = self.cy = self.rx = self.ry = 0
            logging.error("Circle: Unable to get the attributes for %s", self.xml_node)

class line(svgshape):

    def __init__(self, xml_node):
        super(line, self).__init__(xml_node)

        if not self.xml_node == None:
            line_el = self.xml_node
            self.x1  = _f(line_el.get('x1')) if line_el.get('x1') else 0
            self.y1  = _f(line_el.get('y1')) if line_el.get('y1') else 0
            self.x2 = _f(line_el.get('x2')) if line_el.get('x2') else 0
            self.y2 = _f(line_el.get('y2')) if line_el.get('y2') else 0
        else:
            self.x1 = self.y1 = self.x2 = self.y2 = 0
            logging.error("line: Unable to get the attributes for %s", self.xml_node)

    def d_path(self):
        a = []
        a.append( ['M ', [self.x1, self.y1]] )
        a.append( ['L ', [self.x2, self.y2]] )
        return simplepath.formatPath(a)

class polycommon(svgshape):

    def __init__(self, xml_node, polytype):
        super(polycommon, self).__init__(xml_node)
        self.points = []
        self.polytype = polytype

        if self.xml_node is None:
            return
        raw = self.xml_node.get('points') or ""
        nums = re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', raw)

        if len(nums) < 2:
            return  # nothing useful

        # Pair them up: (x0,y0), (x1,y1), ...
        it = iter(map(float, nums))
        pts = list(zip(it, it))  # pairs 0-1, 2-3, ...

        # Remove consecutive duplicates to avoid zero-length moves
        dedup = []
        for xy in pts:
            if not dedup or xy != dedup[-1]:
                dedup.append(xy)

        self.points = dedup


class polygon(polycommon):

    def __init__(self, xml_node):
         super(polygon, self).__init__(xml_node, 'polygon')

    def d_path(self):
        if not self.points:
            return ""
        d = f"M {self.points[0][0]} {self.points[0][1]}"
        for x, y in self.points[1:]:
            d += f" L {x} {y}"
        d += " Z"
        return d

class polyline(polycommon):

    def __init__(self, xml_node):
         super(polyline, self).__init__(xml_node, 'polyline')

    def d_path(self):
        if not self.points:
            return "" 
        d = f"M {self.points[0][0]} {self.points[0][1]}"
        for x, y in self.points[1:]:
            d += f" L {x} {y}"
        return d

def point_generator(path, mat, flatness):

        if len(simplepath.parsePath(path)) == 0:
                return
       
        simple_path = simplepath.parsePath(path)
        startX,startY = float(simple_path[0][1][0]), float(simple_path[0][1][1])
        yield startX, startY

        p = cubicsuperpath.parsePath(path)
        
        if mat:
            simpletransform.applyTransformToPath(mat, p)

        for sp in p:
                cspsubdiv.subdiv( sp, flatness)
                for csp in sp:
                    ctrl_pt1 = csp[0]
                    ctrl_pt2 = csp[1]
                    end_pt = csp[2]
                    yield end_pt[0], end_pt[1],    
