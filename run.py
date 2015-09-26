#!/usr/bin/env python
"""
This script generates a force and velocity vector diagram for a cross-flow
turbine.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib
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


def plot_radius(ax, theta_deg=0):
    """Plot radius at given azimuthal angle."""
    r = 0.495
    theta_rad = np.deg2rad(theta_deg)
    x2, y2 = r*np.cos(theta_rad), r*np.sin(theta_rad)
    ax.plot((0, x2), (0, y2), "gray", linewidth=2)
    

def plot_center(ax, length=0.07, linewidth=1.75):
    """Plot centermark at origin."""
    ax.plot((0, 0), (-length/2, length/2), lw=linewidth, color="black")
    ax.plot((-length/2, length/2), (0, 0), lw=linewidth, color="black")


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
                                     facecolor="gray", linewidth=1,
                                     edgecolor="gray")
    ax.add_patch(p)


def plot_velocities(ax, theta_deg=0.0, tsr=2.0, label=False):
    """Plot blade velocity, free stream velocity, and relative velocity."""
    r = 0.5
    u_infty = 0.27
    theta_rad = np.deg2rad(theta_deg)
    blade_xy = r*np.cos(theta_rad), r*np.sin(theta_rad)
    head_width = 0.05
    head_length = 0.12
    linewidth = 1.5
    
    # Define some colors (some from the Seaborn deep palette)
    blue = "#4C72B0"
    green = "#55A868"
    dark_gray = (0.3, 0.3, 0.3)
    
    # Make blade velocity vector
    x1, y1 = rotate((0.5, tsr*u_infty), np.deg2rad(theta_deg))
    dx, dy = np.array(blade_xy) - np.array((x1, y1))
    blade_vel = np.array((dx, dy))
    ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length, 
             length_includes_head=True, color=dark_gray, linewidth=linewidth)
    if label:
        ax.text(x1 + 0.01, y1 + 0.05*np.sign(y1), r"$\omega r$")
    
    # Make free stream velocity vector
    y1 += u_infty
    ax.arrow(x1, y1, 0, -u_infty, head_width=head_width, 
             head_length=head_length, length_includes_head=True, 
             color=blue, linewidth=linewidth)
    u_infty = np.array((0, -u_infty))
             
    # Make relative velocity vector
    dx, dy = np.array(blade_xy) - np.array((x1, y1))
    rel_vel = u_infty + blade_vel
    ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length, 
             length_includes_head=True, color=green, linewidth=linewidth)
             
    return {"u_infty": u_infty, "blade_vel": blade_vel, "rel_vel": rel_vel}


def plot_diagram(theta_deg=0.0, tsr=2.0, label=False, save=False):
    """Plot full vector diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    plot_foil(ax, c=0.3, theta_deg=theta_deg)
    plot_radius(ax, theta_deg)
    plot_center(ax)
    vels = plot_velocities(ax, theta_deg, tsr, label=label)
    print(vels)

    # Figure formatting    
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_aspect(1)
    ax.axis("off")
    
    if save:
        fig.savefig("cft-vectors.pdf")


if __name__ == "__main__":
    plt.rcParams["font.size"] = 18
    plot_diagram(120)
