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
import seaborn as sns
from pxl.styleplot import set_sns


# Define some colors (some from the Seaborn deep palette)
blue = sns.color_palette()[0]
green = sns.color_palette()[1]
dark_gray = (0.3, 0.3, 0.3)
red = sns.color_palette()[2]
purple = sns.color_palette()[3]
tan = sns.color_palette()[4]
light_blue = sns.color_palette()[5]


def load_foildata():
    """Loads NACA 0020 airfoil data at Re = 2.1 x 10^5."""
    Re = 2.1e5
    foil = "0020"
    fname = "NACA {}_T1_Re{:.3f}_M0.00_N9.0.dat".format(foil, Re/1e6)
    fpath = "data/{}".format(fname)
    alpha, cl, cd = np.loadtxt(fpath, skiprows=14, unpack=True)
    if alpha[0] != 0.0:
        alpha = np.append([0.0], alpha[:-1])
        cl = np.append([1e-12], cl[:-1])
        cd = np.append(cd[0], cd[:-1])
    # Mirror data about 0 degrees AoA since it's a symmetrical foil
    alpha = np.append(-np.flipud(alpha), alpha)
    cl = np.append(-np.flipud(cl), cl)
    cd = np.append(np.flipud(cd), cd)
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
    alpha_rad[theta_blade_deg > 180] *= -1
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


def plot_vectors(ax, theta_deg=0.0, tsr=2.0, label=False):
    """
    Plot blade velocity, free stream velocity, relative velocity,
    lift, and drag vectors.
    """
    r = 0.5
    u_infty = 0.26
    theta_rad = np.deg2rad(theta_deg)
    blade_xy = r*np.cos(theta_rad), r*np.sin(theta_rad)
    head_width = 0.04
    head_length = 0.11
    linewidth = 1.5
    
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
             length_includes_head=True, color=tan, linewidth=linewidth)
    
    # Calculate angle between blade vel and rel vel
    alpha_deg = np.rad2deg(np.arccos(np.dot(blade_vel/mag(blade_vel), 
                                            rel_vel/mag(rel_vel))))
    if theta_deg > 180:
        alpha_deg *= -1
    
    # Make drag vector
    drag_amplify = 3.0
    data = lookup_foildata(alpha_deg)
    drag = data["cd"]*mag(rel_vel)**2*drag_amplify
    if drag < 0.4/drag_amplify:
        hs = 0.5
    else:
        hs = 1
    dx, dy = drag*np.array((dx, dy))/mag((dx, dy))
    ax.arrow(blade_xy[0], blade_xy[1], dx, dy, head_width=head_width*hs, 
             head_length=head_length*hs, length_includes_head=True, color=red, 
             linewidth=linewidth)
    
    # Make lift vector
    lift_amplify = 1.5
    lift = data["cl"]*mag(rel_vel)**2*lift_amplify
    dx, dy = rotate((dx, dy), -np.pi/2)/mag((dx, dy))*lift
    if lift < 0.5/lift_amplify:
        hs = 0.5
    else:
        hs = 1
    ax.arrow(blade_xy[0], blade_xy[1], dx, dy, head_width=head_width*hs, 
             head_length=head_length*hs, length_includes_head=True, color=green, 
             linewidth=linewidth)

    return {"u_infty": u_infty, "blade_vel": blade_vel, "rel_vel": rel_vel}


def plot_alpha(ax=None, tsr=2.0, theta=None, alpha_ss=None, **kwargs):
    """Plot angle of attack versus azimuthal angle."""
    if ax is None:
        fig, ax = plt.subplots()
    df = calc_cft_ctorque(tsr=tsr)
    ax.plot(df.theta, df.alpha_deg, **kwargs)
    ax.set_ylabel(r"$\alpha$ (degrees)")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_xlim((0, 360))
    ylim = np.round(df.alpha_deg.max() + 5)
    ax.set_ylim((-ylim, ylim))
    if theta is not None:
        ax.plot(theta, df.alpha_deg[df.theta==theta].iloc[0], "ok")
    if alpha_ss is not None:
        ax.hlines((alpha_ss, -alpha_ss), 0, 360, linestyles="dashed")
        
        
def plot_rel_vel_mag(ax=None, tsr=2.0, theta=None, **kwargs):
    """Plot relative velocity magnitude versus azimuthal angle."""
    if ax is None:
        fig, ax = plt.subplots()
    df = calc_cft_ctorque(tsr=tsr)
    ax.plot(df.theta, df.rel_vel_mag, **kwargs)
    ax.set_ylabel(r"$|\vec{U}_\mathrm{rel}|$")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_xlim((0, 360))
    if theta is not None:
        ax.plot(theta, df.rel_vel_mag[df.theta==theta].iloc[0], "ok")
        
        
def plot_ctorque(ax=None, tsr=2.0, theta=None, **kwargs):
    """Plot torque coefficient versus azimuthal angle."""
    if ax is None:
        fig, ax = plt.subplots()
    df = calc_cft_ctorque(tsr=tsr)
    ax.plot(df.theta, df.ctorque, **kwargs)
    ax.set_ylabel("Torque coefficient")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_xlim((0, 360))
    if theta is not None:
        ax.plot(theta, df.ctorque[df.theta==theta].iloc[0], "ok")


def plot_diagram(ax=None, theta_deg=0.0, tsr=2.0, label=False, save=False,
                 axis="on"):
    """Plot full vector diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    plot_foil(ax, c=0.3, theta_deg=theta_deg)
    plot_radius(ax, theta_deg)
    plot_center(ax)
    plot_vectors(ax, theta_deg, tsr, label=label)

    # Figure formatting    
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis(axis)
    
    if save:
        fig.savefig("cft-vectors.pdf")
        
        
def plot_all(theta_deg=0.0, tsr=2.0):
    """Create diagram and plots of kinematics in a single figure."""
    fig = plt.figure(figsize=(7.5, 4.75))
    # Draw vector diagram
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plot_diagram(ax1, theta_deg, tsr, axis="on")
    # Plot angle of attack
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    plot_alpha(ax2, tsr=tsr, theta=theta_deg, alpha_ss=18, color=light_blue)
    # Plot relative velocity magnitude
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    plot_rel_vel_mag(ax3, tsr=tsr, theta=theta_deg, color=tan)
    # Plot torque coefficient
    ax4 = plt.subplot2grid((3, 3), (2, 2))
    plot_ctorque(ax4, tsr=tsr, theta=theta_deg, color=purple)
    
    fig.tight_layout()
    

if __name__ == "__main__":
    set_sns(font_scale=1.25)
    plt.rcParams["axes.grid"] = True
    plot_all(theta_deg=45, tsr=2.0)
