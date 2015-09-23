#!/usr/bin/env python
"""
This script generates a force and velocity vector diagram for a cross-flow
turbine.
"""

import gizeh as gz
import numpy as np
import matplotlib.pyplot as plt


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


def make_arrow(xy_start, xy_end, label=""):
    """Make an arrow."""
    line = gz.polyline((xy_start, xy_end), stroke_width=2)
    head_point2 = (xy_end[0] + 50, xy_end[1] + 150)
    head = gz.polyline((xy_end, head_point2), stroke_width=2, close_path=False)
    return gz.Group((line, head))


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
    arrow.draw(canvas)
    canvas.write_to_png("cft-vectors.png")


if __name__ == "__main__":
    main()
