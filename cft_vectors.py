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
import os


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
    f_cl = interp1d(df.alpha_deg, df.cl, bounds_error=False)
    f_cd = interp1d(df.alpha_deg, df.cd, bounds_error=False)
    f_ct = interp1d(df.alpha_deg, df.cl*np.sin(df.alpha_rad) \
         - df.cd*np.cos(df.alpha_rad), bounds_error=False)
    cl, cd, ct = f_cl(alpha_deg), f_cd(alpha_deg), f_ct(alpha_deg)
    return {"cl": cl, "cd": cd, "ct": ct}


def calc_cft_ctorque(tsr=2.0, chord=0.14, R=0.5):
    """Calculate the geometric torque coefficient for a CFT."""
    U_infty = 1.0
    omega = tsr*U_infty/R
    theta_blade_deg = np.arange(0, 721)
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


def plot_center(ax, length=0.07, linewidth=1.2):
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


def plot_blade_path(ax, R=0.5):
    """Plot blade path as a dashed line."""
    p = plt.Circle((0, 0), R, linestyle="dashed", edgecolor="black",
                   facecolor="none", linewidth=1)
    ax.add_patch(p)


def plot_vectors(fig, ax, theta_deg=0.0, tsr=2.0, c=0.3, label=False):
    """Plot blade velocity, free stream velocity, relative velocity, lift, and
    drag vectors.
    """
    r = 0.5
    u_infty = 0.26
    theta_deg %= 360
    theta_rad = np.deg2rad(theta_deg)
    blade_xy = r*np.cos(theta_rad), r*np.sin(theta_rad)
    head_width = 0.04
    head_length = 0.11
    linewidth = 1.5

    # Function for plotting vector labels
    def plot_label(text, x, y, dx, dy, text_width=0.09, text_height=0.03,
                   sign=-1, dist=1.0/3.0):
        text_width *= plt.rcParams["font.size"]/12*6/fig.get_size_inches()[1]
        text_height *= plt.rcParams["font.size"]/12*6/fig.get_size_inches()[1]
        dvec = np.array((dx, dy))
        perp_vec = rotate(dvec, np.pi/2)
        perp_vec /= mag(perp_vec)
        if theta_deg > 270:
            diag = text_height
        else:
            diag = np.array((text_width, text_height))
        # Projection of text diagonal vector onto normal vector
        proj = np.dot(diag, perp_vec)
        if sign != -1:
            proj = 0 # Text is on right side of vector
        if theta_deg > 180:
            sign *= -1
        dxlab, dylab = perp_vec*(np.abs(proj) + .01)*sign
        xlab, ylab = x + dx*dist + dxlab, y + dy*dist + dylab
        ax.text(xlab, ylab, text)

    # Make blade velocity vector
    x1, y1 = rotate((0.5, tsr*u_infty), np.deg2rad(theta_deg))
    dx, dy = np.array(blade_xy) - np.array((x1, y1))
    blade_vel = np.array((dx, dy))
    ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length,
             length_includes_head=True, color=dark_gray, linewidth=linewidth)
    if label:
        plot_label(r"$-\omega r$", x1, y1, dx*0.25, dy*0.5)
        # Make chord line vector
        x1c, y1c = np.array((x1, y1)) - np.array((dx, dy))*0.5
        x2c, y2c = np.array((x1, y1)) + np.array((dx, dy))*2
        ax.plot([x1c, x2c], [y1c, y2c], marker=None, color="k", linestyle="-.",
                zorder=1)

    # Make free stream velocity vector
    y1 += u_infty
    ax.arrow(x1, y1, 0, -u_infty, head_width=head_width,
             head_length=head_length, length_includes_head=True,
             color=blue, linewidth=linewidth)
    u_infty = np.array((0, -u_infty))
    if label:
        dy = -mag(u_infty)
        plot_label(r"$U_\mathrm{in}$", x1, y1, 0, dy, text_width=0.1)

    # Make relative velocity vector
    dx, dy = np.array(blade_xy) - np.array((x1, y1))
    rel_vel = u_infty + blade_vel
    ax.plot((x1, x1 + dx), (y1, y1 + dy), lw=0)
    ax.arrow(x1, y1, dx, dy, head_width=head_width, head_length=head_length,
             length_includes_head=True, color=tan, linewidth=linewidth)
    if label:
        plot_label(r"$U_\mathrm{rel}$", x1, y1, dx, dy, sign=1,
                   text_width=0.11)

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
    if label:
        plot_label(r"$F_d$", blade_xy[0], blade_xy[1], dx, dy, sign=-1,
                   dist=0.66)

    # Make lift vector
    lift_amplify = 1.5
    lift = data["cl"]*mag(rel_vel)**2*lift_amplify
    dx, dy = rotate((dx, dy), -np.pi/2)/mag((dx, dy))*lift
    if np.abs(lift) < 0.4/lift_amplify:
        hs = 0.5
    else:
        hs = 1
    ax.plot((blade_xy[0], blade_xy[0] + dx), (blade_xy[1], blade_xy[1] + dy),
            linewidth=0)
    ax.arrow(blade_xy[0], blade_xy[1], dx, dy, head_width=head_width*hs,
             head_length=head_length*hs, length_includes_head=True,
             color=green, linewidth=linewidth)
    if label:
        plot_label(r"$F_l$", blade_xy[0], blade_xy[1], dx, dy, sign=-1,
                   text_width=0.12, text_height=0.02, dist=0.66)

    # Label radius
    if label:
        plot_label("$r$", 0, 0, blade_xy[0], blade_xy[1], text_width=0.04,
                   text_height=0.04)

    # Label angle of attack
    if label:
        ast = "simple,head_width={},tail_width={},head_length={}".format(
                head_width*8, linewidth/16, head_length*8)
        xy = blade_xy - rel_vel/mag(rel_vel)*0.2
        ax.annotate(r"$\alpha$", xy=xy, xycoords="data",
                    xytext=(37.5, 22.5), textcoords="offset points",
                    arrowprops=dict(arrowstyle=ast,
                                    ec="none",
                                    connectionstyle="arc3,rad=0.1",
                                    color="k"))
        xy = blade_xy - blade_vel/mag(blade_vel)*0.2
        ax.annotate("", xy=xy, xycoords="data",
                    xytext=(-15, -30), textcoords="offset points",
                    arrowprops=dict(arrowstyle=ast,
                                    ec="none",
                                    connectionstyle="arc3,rad=-0.1",
                                    color="k"))

    # Label azimuthal angle
    if label:
        xy = np.array(blade_xy)*0.6
        ast = "simple,head_width={},tail_width={},head_length={}".format(
                head_width*5.5, linewidth/22, head_length*5.5)
        ax.annotate(r"$\theta$", xy=xy, xycoords="data",
                    xytext=(0.28, 0.12), textcoords="data",
                    arrowprops=dict(arrowstyle=ast,
                                    ec="none",
                                    connectionstyle="arc3,rad=0.1",
                                    color="k"))
        ax.annotate("", xy=(0.41, 0), xycoords="data",
                    xytext=(0.333, 0.12), textcoords="data",
                    arrowprops=dict(arrowstyle=ast,
                                    ec="none",
                                    connectionstyle="arc3,rad=-0.1",
                                    color="k"))

    # Label pitching moment
    if label:
        xy = np.array(blade_xy)*1.1 - blade_vel/mag(blade_vel) * c/4
        ast = "simple,head_width={},tail_width={},head_length={}".format(
                head_width*8, linewidth/16, head_length*8)
        ax.annotate(r"", xy=xy, xycoords="data",
                    xytext=(25, -15), textcoords="offset points",
                    arrowprops=dict(arrowstyle=ast,
                                    ec="none",
                                    connectionstyle="arc3,rad=0.6",
                                    color="k"))
        plot_label(r"$M$", xy[0], xy[1], 0.1, 0.1, sign=-1, dist=0.66)

    return {"u_infty": u_infty, "blade_vel": blade_vel, "rel_vel": rel_vel}


def plot_alpha(ax=None, tsr=2.0, theta=None, alpha_ss=None, **kwargs):
    """Plot angle of attack versus azimuthal angle."""
    if theta is not None:
        theta %= 360
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
        f = interp1d(df.theta, df.alpha_deg)
        ax.plot(theta, f(theta), "ok")
    if alpha_ss is not None:
        ax.hlines((alpha_ss, -alpha_ss), 0, 360, linestyles="dashed")


def plot_rel_vel_mag(ax=None, tsr=2.0, theta=None, **kwargs):
    """Plot relative velocity magnitude versus azimuthal angle."""
    if theta is not None:
        theta %= 360
    if ax is None:
        fig, ax = plt.subplots()
    df = calc_cft_ctorque(tsr=tsr)
    ax.plot(df.theta, df.rel_vel_mag, **kwargs)
    ax.set_ylabel(r"$|\vec{U}_\mathrm{rel}|$")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_xlim((0, 360))
    if theta is not None:
        f = interp1d(df.theta, df.rel_vel_mag)
        ax.plot(theta, f(theta), "ok")


def plot_alpha_relvel_all(tsrs=np.arange(1.5, 6.1, 1.0), save=False):
    """Plot angle of attack and relative velocity magnitude for a list of TSRs.

    Figure will have two subplots in a single row.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(7.5, 3.0))
    cm = plt.cm.get_cmap("Reds")
    for tsr in tsrs:
        color = cm(tsr/np.max(tsrs))
        plot_alpha(ax=ax1, tsr=tsr, label=r"$\lambda = {}$".format(tsr),
                   color=color)
        plot_rel_vel_mag(ax=ax2, tsr=tsr, color=color)
    [a.set_xticks(np.arange(0, 361, 60)) for a in (ax1, ax2)]
    ax1.legend(loc=(0.17, 1.1), ncol=len(tsrs))
    ax1.set_ylim((-45, 45))
    ax1.set_yticks(np.arange(-45, 46, 15))
    ax2.set_ylabel(r"$|\vec{U}_\mathrm{rel}|/U_\infty$")
    fig.tight_layout()
    if save:
        fig.savefig("figures/alpha_deg_urel_geom.pdf", bbox_inches="tight")


def plot_ctorque(ax=None, tsr=2.0, theta=None, **kwargs):
    """Plot torque coefficient versus azimuthal angle."""
    theta %= 360
    if ax is None:
        fig, ax = plt.subplots()
    df = calc_cft_ctorque(tsr=tsr)
    ax.plot(df.theta, df.ctorque, **kwargs)
    ax.set_ylabel("Torque coeff.")
    ax.set_xlabel(r"$\theta$ (degrees)")
    ax.set_xlim((0, 360))
    if theta is not None:
        f = interp1d(df.theta, df.ctorque)
        ax.plot(theta, f(theta), "ok")


def plot_diagram(fig=None, ax=None, theta_deg=0.0, tsr=2.0, label=False,
                 save=False, axis="on", full_view=True):
    """Plot full vector diagram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    plot_blade_path(ax)
    if label:
        # Create dashed line for x-axis
        ax.plot((-0.5, 0.5), (0, 0), linestyle="dashed", color="k",
                zorder=1)
    plot_foil(ax, c=0.3, theta_deg=theta_deg)
    plot_radius(ax, theta_deg)
    plot_center(ax)
    plot_vectors(fig, ax, theta_deg, tsr, label=label)

    # Figure formatting
    if full_view:
        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
    ax.set_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis(axis)

    if save:
        fig.savefig("figures/cft-vectors.pdf")


def plot_all(theta_deg=0.0, tsr=2.0, scale=1.0, full_view=True):
    """Create diagram and plots of kinematics in a single figure."""
    fig = plt.figure(figsize=(7.5*scale, 4.75*scale))
    # Draw vector diagram
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
    plot_diagram(fig, ax1, theta_deg, tsr, axis="on", full_view=full_view)
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
    return fig


def make_frame(t):
    """Make a frame for a movie."""
    sec_per_rev = 5.0
    deg = t/sec_per_rev*360
    return mplfig_to_npimage(plot_all(deg, scale=2.0))


def make_animation(filetype="mp4", fps=30):
    """Make animation video."""
    if not os.path.isdir("videos"):
        os.mkdir("videos")
    animation = VideoClip(make_frame, duration=5.0)
    if "mp4" in filetype.lower():
        animation.write_videofile("videos/cft-animation.mp4", fps=fps)
    elif "gif" in filetype.lower():
        animation.write_gif("videos/cft-animation.gif", fps=fps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create cross-flow turbine \
                                     vector diagrams.")
    parser.add_argument("create", choices=["figure", "diagram", "animation"],
                        help="Either create a static figure or animation")
    parser.add_argument("--angle", type=float, default=60.0,
                        help="Angle (degrees) to create figure")
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--save", "-s", action="store_true", default=False,
                        help="Save figure")
    args = parser.parse_args()

    if args.save:
        if not os.path.isdir("figures"):
            os.mkdir("figures")

    if args.create == "diagram":
        set_sns(font_scale=2)
        plot_diagram(theta_deg=args.angle, label=True, axis="off",
                     save=args.save)
    elif args.create == "figure":
        set_sns()
        plot_alpha_relvel_all(save=args.save)
    elif args.create == "animation":
        set_sns(font_scale=2)
        from moviepy.editor import VideoClip
        from moviepy.video.io.bindings import mplfig_to_npimage
        make_animation()

    if args.show:
        plt.show()
