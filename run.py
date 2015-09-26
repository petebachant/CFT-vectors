#!/usr/bin/env python
"""
This script generates a force and velocity vector diagram for a cross-flow
turbine.
"""

from __future__ import division, print_function
import gizeh as gz
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cairocffi as cairo


class PDFSurface:
    """
    Simple class to allow gizeh to create pdf figures
    
    from https://gist.github.com/cosmonaut/31ec5f6370eb891581f1
    """
    def __init__(self, name, width, height, bg_color=None):
        self.width = width
        self.height = height
        self._cairo_surface = cairo.PDFSurface(name, width, height)

    def get_new_context(self):
        """Returns a new context for drawing on the surface."""
        return cairo.Context(self._cairo_surface)

    def get_npimage(self, transparent=False, y_origin="top"):
        """Returns a WxHx[3-4] numpy array representing the RGB picture.
        
        If `transparent` is True the image is WxHx4 and represents a RGBA 
        picture,
        i.e. array[i,j] is the [r,g,b,a] value of the pixel at position [i,j].
        If `transparent` is false, a RGB array is returned.

        Parameter y_origin ("top" or "bottom") decides whether point (0,0) lies in
        the top-left or bottom-left corner of the screen.
        """

        im = 0+np.frombuffer(self._cairo_surface.get_data(), np.uint8)
        im.shape = (self.height, self.width, 4)
        im = im[:,:,[2,1,0,3]] # put RGB back in order
        if y_origin== "bottom":
            im = im[::-1]
        return im if transparent else im[:,:, :3]

    def flush(self):
        """Write the file"""
        self._cairo_surface.flush()

    def finish(self):
        """Close the surface"""
        self._cairo_surface.finish()


def mag(v):
    """
    Return magnitude of 2-D vector (input as a tuple, list, or NumPy array).
    """
    return np.sqrt(v[0]**2 + v[1]**2)
    

def rotate(v, rad):
    """Rotate a 2-D vector by rad radians."""
    dc, ds = np.cos(rad), np.sin(rad)
    x, y = v[0], v[1]
    x, y = dc*x - ds*y, ds*x + dc*y
    return np.array((x, y))


def gen_naca_points(naca="0020", c=100, npoints=100, tuples=True):
    """Generate points for a NACA foil."""
    x = np.linspace(0, 1, npoints)*c
    t = float(naca[2:])/100.0
    y = 5.0*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 \
            + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
    y = np.append(y, -y[::-1])
    x = np.append(x, x[::-1])
    if tuples:
        return np.array([(x0, y0) for x0, y0 in zip(x, y)])
    else:
        return x, y


def test_gen_naca_points():
    points = gen_naca_points()
    x = []
    y = []
    for p in points:
        x.append(p[0])
        y.append(p[1])
    fig, ax = plt.subplots()
    ax.plot(x, y, "o")
    ax.set_aspect(1)
    plt.show()


def make_foil(naca="0020", c=100, xy=(350, 350), angle=np.pi/2, **kwargs):
    """Make a NACA foil."""
    points = gen_naca_points(naca, c)
    kwargs["xy"] = xy
    kwargs["angle"] = angle
    line = gz.polyline(points, close_path=True, stroke_width=2,
                       fill=(0.6, 0.6, 0.6), **kwargs)
    return line


def make_arrow(xy_start, xy_end, head_angle_deg=25, head_len=40, label="",
               color=(0, 0, 0), stroke_width=2, **kwargs):
    """Make an arrow."""
    head_angle = np.deg2rad(head_angle_deg)
    xy_start = np.array(xy_start, dtype=float)
    xy_end = np.array(xy_end, dtype=float)
    # Create shaft
    shaft = gz.polyline((xy_start, xy_end), stroke_width=stroke_width, 
                        stroke=color, **kwargs)
    # Create direction vector
    direction = xy_end - xy_start
    direction /= mag(direction)
    # Head direction 2 is direction rotated by half the head angle
    head_dir2 = rotate(direction, head_angle/2)
    # Hypotenuse of arrow head
    head_hypot = head_len*np.cos(head_angle/2)    
    # Head points are created by moving along head directions
    head_point2 = xy_end - head_hypot*head_dir2
    head_dir1 = rotate(direction, -head_angle/2)
    head_point1 = xy_end - head_hypot*head_dir1
    head = gz.polyline((xy_end, head_point2, head_point1), 
                       stroke_width=stroke_width, close_path=True, 
                       stroke=color, fill=color)
    return gz.Group((shaft, head))
    

def make_label(text, **kwargs):
    """Create a label."""
    label = gz.text(text, fontfamily="times", fontslant="italic", **kwargs)
    return label
    
    
def make_diagram(tsr, theta):
    """Create entire turbine diagram as a Group."""
    pass
    
    
def plot_surface(surface):
    """Plot a Gizeh Surface with matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(surface.get_npimage())
    
    
def make_pdf():
    s = PDFSurface("test.pdf", 600, 600, bg_color=(1, 1, 1))
    arrow = make_arrow((0, 0), (100, 100), stroke_width=2, head_len=20)
    arrow.draw(s)
    s.flush()
    s.finish()


def main():
    c = 100
    r = 200
    canvas = gz.Surface(width=700, height=700, bg_color=(1, 1, 1))
    origin = canvas.width/2, canvas.height/2
    origin_x, origin_y = origin
    radius = gz.polyline([origin, (origin_x + r, origin_y)], stroke_width=2)
    radius.draw(canvas)
    foil = make_foil(c=c, xy=(origin_x + r, origin_y - c/4))
    foil.draw(canvas)
    arrow = make_arrow((100, 100), (300, 300))
    arrow2 = make_arrow((400, 400), (400, 600), 
                        color=(0.5, 0.5, 0.5), stroke_width=5)
    arrow.draw(canvas)
    arrow2.draw(canvas)
    
    origin_label = make_label(chr(945), fontsize=50, xy=(350, 300))
    origin_label.draw(canvas)

    canvas.write_to_png("cft-vectors.png")    
    plot_surface(canvas)


def plot_radius(ax, theta_deg=0):
    """Plot radius at given azimuthal angle."""
    r = 0.495
    theta_rad = np.deg2rad(theta_deg)
    x2, y2 = r*np.cos(theta_rad), r*np.sin(theta_rad)
    ax.plot((0, x2), (0, y2), "k", linewidth=2)


def make_naca_path(c=0.3, theta_deg=0.0):
    verts = gen_naca_points(c=c)
    verts = np.array([rotate(v, -np.pi/2) for v in verts])
    verts += (0.5, c/4)
    theta_rad = np.deg2rad(theta_deg)
    verts = np.array([rotate(v, theta_rad) for v in verts])
    p = matplotlib.path.Path(verts, closed=True)
    return p
        
    
def plot_foil(ax, c=0.3, theta_deg=0.0):
    """Plot the foil shape using a matplotlib patch."""
    p = matplotlib.patches.PathPatch(make_naca_path(c, theta_deg), 
                                     facecolor="gray")
    ax.add_patch(p)
    

def plot_diagram(theta_deg=0.0, tsr=2.0):
    """Plot full vector diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    plot_foil(ax, c=0.3, theta_deg=theta_deg)
    plot_radius(ax, theta_deg)

    
    # Make blade velocity vector
    ax.arrow(0.5, 0.5, 0, -0.5, head_width=0.06, head_length=0.15, 
             length_includes_head=True, color="black", linewidth=1.5)
    ax.text(0.5, 0.5, r"$\omega r$")

    # Figure formatting    
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_aspect(1)
    ax.axis("off")
    
    # Save figure
    fig.savefig("test.pdf")


if __name__ == "__main__":
    plt.rcParams["font.size"] = 18
    plot_diagram(0)
