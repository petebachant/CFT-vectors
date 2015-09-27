#!/usr/bin/env python
"""
This script generates a force and velocity vector diagram for a cross-flow
turbine.
"""

from __future__ import division, print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d


def load_foildata():
    """Loads NACA 0020 airfoil data at Re = 2.1 x 10^5."""
    Re = 2.1e5
    foil = "0020"
    fname = "NACA {}_T1_Re{:.3f}_M0.00_N9.0.dat".format(foil, Re/1e6)
    fpath = "data/{}".format(fname)
    alpha, cl, cd = np.loadtxt(fpath, skiprows=14, unpack=True)
    if alpha[0] != 0.0:
        alpha = np.append([0.0], alpha[:-1])
        cl = np.append([0.0], cl[:-1])
        cd = np.append(cd[0.0], cd[:-1])
    df = pd.DataFrame()
    df["alpha_deg"] = alpha
    df["cl"] = cl
    df["cd"] = cd
    return df
    
    
def lookup_foildata(alpha_deg):
    """Lookup foil characteristics at given angle of attack."""
    alpha_deg = np.asarray(alpha_deg)
    df = load_foildata()
    df["alpha_rad"] = np.deg2rad(df.alpha_deg)
    f_cl = interp1d(df.alpha_deg, df.cl)
    f_cd = interp1d(df.alpha_deg, df.cd)
    f_ct = interp1d(df.alpha_deg, df.cl*np.sin(df.alpha_rad) \
         - df.cd*np.cos(df.alpha_rad))
    cl, cd, ct = f_cl(alpha_deg), f_cd(alpha_deg), f_ct(alpha_deg)
    return {"cl": cl, "cd": cd, "ct": ct}
    
    
def calc_cft_ctorque(tsr=2.0, chord=0.14, R=0.5):
    """Calculate the geometric torque coefficient for a CFT."""
    U_infty = 1.0
    omega = tsr*U_infty/R
    theta_blade_deg = np.arange(0, 361)
    theta_blade_rad = np.deg2rad(theta_blade_deg)
    blade_vel_mag = omega*R
    blade_vel_x = blade_vel_mag*np.cos(theta_blade_rad)
    blade_vel_y = blade_vel_mag*np.sin(theta_blade_rad)
    u = U_infty # No induction
    rel_vel_mag = np.sqrt((blade_vel_x + u)**2 + blade_vel_y**2)
    rel_vel_x = u + blade_vel_x
    rel_vel_y = blade_vel_y
    relvel_dot_bladevel = (blade_vel_x*rel_vel_x + blade_vel_y*rel_vel_y)
    alpha_rad = np.arccos(relvel_dot_bladevel/(rel_vel_mag*blade_vel_mag))
    alpha_deg = np.rad2deg(alpha_rad)
    foil_coeffs = lookup_foildata(alpha_deg)
    ctorque = foil_coeffs["ct"]*chord/(2*R)*rel_vel_mag**2/U_infty**2
    cdx = -foil_coeffs["cd"]*np.sin(np.pi/2 - alpha_rad + theta_blade_rad)
    clx = foil_coeffs["cl"]*np.cos(np.pi/2 - alpha_rad - theta_blade_rad)
    df = pd.DataFrame()
    df["theta"] = theta_blade_deg
    df["alpha_deg"] = alpha_deg
    df["rel_vel_mag"] = rel_vel_mag
    df["ctorque"] = ctorque
    df["cdrag"] = clx + cdx
    return df


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
    red = "#C44E52"
    purple = "#8172B2"
    
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
