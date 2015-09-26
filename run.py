#!/usr/bin/env python
"""
This script generates a force and velocity vector diagram for a cross-flow
turbine.
"""

from __future__ import division, print_function
import gizeh as gz
import numpy as np
import matplotlib.pyplot as plt


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


def gen_naca_points(naca="0020", c=100, npoints=100):
    """Generate points for a NACA foil."""
    x = np.linspace(0, 1, npoints)*c
    t = float(naca[2:])/100.0
    y = 5.0*t*c*(0.2969*np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 \
            + 0.2843*(x/c)**3 - 0.1015*(x/c)**4)
    y = np.append(y, -y[::-1])
    x = np.append(x, x[::-1])
    points = [(x0, y0) for x0, y0 in zip(x, y)]
    return points


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
    
    
def make_diagram(tsr, theta):
    """Create entire turbine diagram as a Group."""
    pass
    
    
def plot_surface(surface):
    """Plot a Gizeh Surface with matplotlib."""
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(surface.get_npimage())


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
    canvas.write_to_png("cft-vectors.png")
    
    plot_surface(canvas)


if __name__ == "__main__":
    main()
