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


def draw_foil(naca="0020", c=100):
    """Draw NACA 0020 foil."""
    points = gen_naca_points(naca, c)
    line = gz.polyline(points, close_path=False, stroke_width=2, xy=(300, 300))
    return line


def main():
    canvas = gz.Surface(width=700, height=700)
    foil = draw_foil()
    foil.draw(canvas)
    canvas.write_to_png("cft-vectors.png")


if __name__ == "__main__":
    main()
