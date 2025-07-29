#vis.py

# Taken from
#https://github.com/ML-KULeuven/soccermix/blob/master/vis.py
#SoccerMix. T. Decroos

# Adapted for my use

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotsoccer as mps
import softclustering as sc
import numpy as np

# grab the default color cycle as a list of hex‐colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def dual_axes(figsize=4):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches((figsize * 3, figsize))
    return axs[0], axs[1]


def loc_angle_axes(figsize=4):
    fig, _axs = plt.subplots(1, 2)
    fig.set_size_inches((figsize * 3, figsize))

    axloc = plt.subplot(121)
    axloc = field(axloc)
    axpol = plt.subplot(122, projection="polar")
    # axpol.set_rticks(np.linspace(0, 2, 21))
    return axloc, axpol


def field(ax):
    ax = mps.field(ax=ax, show=False)
    ax.set_xlim(-1, 105 + 1)
    ax.set_ylim(-1, 68 + 1)
    return ax


def movement(ax):
    plt.axis("on")
    plt.axis("scaled")
    ax.spines["left"].set_position("center")
    ax.spines["bottom"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    return ax


def polar(ax):
    plt.axis("on")
    ax.set_xlim(-3.2, 3.2)
    ax.spines["left"].set_position("center")
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    return ax


##################################
# MODEL-BASED VISUALIZATION
#################################


def show_location_model(loc_model: sc.MixtureModel, show=True, figsize=6):
    ax = mps.field(show=False, figsize=figsize)

    norm_strengths = loc_model.get_weights() / np.max(loc_model.get_weights()) * 0.8
    for strength, gauss, color in zip(norm_strengths, loc_model.get_components(), colors * 10):
        mean, cov = gauss.get_params()
        add_ellips(ax, mean, cov, color=color, alpha=strength)
    if show:
        plt.show()
    else:
        return ax
        

def show_direction_model(gauss: sc.MultivariateGaussian, dir_models: sc.MixtureModel, show=True, figsize=6):
    ax = mps.field(show=False, figsize=figsize)
    # for gauss in loc_model.submodels:
    mean, cov = gauss.get_params()
    add_ellips(ax, mean, cov, alpha=0.5)
    x, y = mean
    for vonmises in dir_models.get_components():
        loc, kappa = vonmises.get_params()
        r = vonmises.get_mean_length()
        dx = np.cos(loc)
        dy = np.sin(loc)
        add_arrow(ax, x, y, 10*dx, 10*dy,
            linewidth=0.5)
    if show:
        plt.show()


def show_location_models(loc_models: list[sc.MixtureModel], figsize=6):
    """
    Model-based visualization
    """
    for model in loc_models:
        print(model)
        show_location_model(model, figsize=6)


def show_all_models(loc_model: sc.MixtureModel,
                    dir_models: list[sc.MixtureModel],
                    figsize: float = 6,
                    arrow_scale: float = 12.0):
    """
    Plot every (Gaussian + VonMises arrows) on one shared Axes,
    using a different color per cluster, and arrow lengths ∝ mean length r.
    """
    ax = mps.field(show=False, figsize=figsize)

    n = loc_model.n_clusters
    palette = colors * ((n // len(colors)) + 1)

    for i, (gauss, vmm) in enumerate(zip(loc_model.get_components(),
                                         dir_models)):
        col = palette[i]
        mean, cov = gauss.get_params()
        add_ellips(ax, mean, cov, color=col, alpha=0.5)
        x0, y0 = mean

        for vonm in vmm.get_components():
            loc, _ = vonm.get_params()
            r = vonm.get_mean_length()  # in [0, 1]
            length = arrow_scale * r  # scale accordingly
            dx, dy = np.cos(loc), np.sin(loc)
            add_arrow(ax,
                      x0, y0,
                      length * dx,
                      length * dy,
                      linewidth=0.8)
    plt.show()
        

def show_direction_models(loc_models, dir_models, figsize=8):
    """
    Model-based visualization
    """
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        ax = mps.field(show=False, figsize=figsize)

        norm_strengths = loc_model.priors / np.max(loc_model.priors) * 0.8
        for i, (strength, gauss) in enumerate(zip(norm_strengths, loc_model.submodels)):
            add_ellips(ax, gauss.mean, gauss.cov, alpha=strength)

            x, y = gauss.mean
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:
                    print(dir_model.name, dir_model.n_components)
                    dir_norm_strengths = (
                        dir_model.priors / np.max(dir_model.priors) * 0.8
                    )
                    for strength, vonmises in zip(
                        dir_norm_strengths, dir_model.submodels
                    ):
                        dx = np.cos(vonmises.loc)[0]
                        dy = np.sin(vonmises.loc)[0]
                        r = vonmises.R[0]
                        add_arrow(
                            ax,
                            x,
                            y,
                            10 * r * dx,
                            10 * r * dy,
                            alpha=strength,
                            threshold=0,
                        )
        plt.show()


def add_ellips(ax, mean, covar, color=None, alpha=0.7):
    # eigen‑decomposition
    eigvals, eigvecs = linalg.eigh(covar)
    # 2sigma lengths along principal axes
    lengths = 2.0 * np.sqrt(2.0) * np.sqrt(eigvals)

    # principal direction = first eigenvector
    direction = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])

    # compute angle in degrees
    angle = np.degrees(np.arctan2(direction[1], direction[0]))

    # enforce a minimum size so it’s always visible
    width, height = max(lengths[0], 3), max(lengths[1], 3)

    ell = mpl.patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,    # or edgecolor=color
        alpha=alpha
    )
    ax.add_patch(ell)
    return ax


def add_arrow(ax, x, y, dx, dy,
              arrowsize=2.5,
              linewidth=2.0,
              threshold=1.8,
              alpha=1.0,
              fc='grey',
              ec='grey'):
    """
    Draw an arrow only if its dx or dy exceed the threshold,
    with both facecolor and edgecolor set to grey by default.
    """
    if np.sqrt(dx**2 + dy**2) > threshold:
        return ax.arrow(
            x, y, dx, dy,
            head_width=arrowsize,
            head_length=arrowsize,
            linewidth=linewidth,
            fc=fc,
            ec=ec,
            length_includes_head=True,
            alpha=alpha,
            zorder=3,
        )



######################################################
# PROBABILITY-DENSITY-FUNCTION BASED VISUALIZATION
######################################################

colors = [
    "#377eb8",
    "#e41a1c",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
]

def show_direction_models_pdf(loc_models, dir_models):
    """
    Probability-density function based visualization
    """
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        for i, gauss in enumerate(loc_model.submodels):
            # axloc, axpol = dual_axes()
            # # vis.add_ellips(axloc,gauss.mean,gauss.cov)
            # draw_contour(axloc, gauss, cmap="Blues")
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:
                    print(dir_model.name, dir_model.n_components)

                    axcol, axpol = loc_angle_axes()
                    draw_contour(axcol, gauss, cmap="Blues")
                    draw_vonmises_pdfs(dir_model, axpol)
                    plt.show()


def draw_contour(ax, gauss, n=100, cmap="Blues"):
    x = np.linspace(0, 105, n)
    y = np.linspace(0, 105, n)
    xx, yy = np.meshgrid(x, y)
    zz = gauss.pdf(np.array([xx.flatten(), yy.flatten()]).T)
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=cmap)
    return ax


def draw_vonmises_pdfs(model, ax=None,figsize=4,projection="polar",n=200,show=True):
    if ax is None:
        ax = plt.subplot(111, projection=projection)
        plt.gcf().set_size_inches((figsize, figsize))
    x = np.linspace(-np.pi, np.pi, n)
    total = np.zeros(x.shape)
    for i, (prior, vonmises) in enumerate(zip(model.priors, model.submodels)):
        p = prior * vonmises.pdf(x)
        p = np.nan_to_num(p)
        ax.plot(x, p, linewidth=2, color=(colors * 10)[i],label = f"Component {i}")
        total += p
#     ax.plot(x, total, linewidth=3, color="black")
    return ax


#################################
# DATA-BASED VISUALIZATION
#################################


def scatter_location_model(
    loc_model, actions, W, samplefn="max", tol=0.1, figsize=6, alpha=0.5, show=True
):
    X = actions[["x", "y"]]
    probs = loc_model.predict_proba(X, W[loc_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    x = X[pos_prob_idx]
    w = probs[pos_prob_idx]

    if loc_model.n_components > len(colors):
        means = [m.mean for m in loc_model.submodels]
        good_colors = color_submodels(means, colors)
    else:
        good_colors = colors
    c = scattercolors(w, good_colors, samplefn=samplefn)

    ax = mps.field(show=False, figsize=figsize)
    ax.scatter(x.x, x.y, c=c, alpha=alpha)
    if show:
        plt.show()

def scatter_location_model_black(
    loc_model, actions, W, samplefn="max", tol=0.1, figsize=6, alpha=0.5, show=True
):
    X = actions[["x", "y"]]
    probs = loc_model.predict_proba(X, W[loc_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    x = X[pos_prob_idx]
    w = probs[pos_prob_idx]

    if loc_model.n_components > len(colors):
        means = [m.mean for m in loc_model.submodels]
        good_colors = color_submodels(means, colors)
    else:
        good_colors = colors
    c = scattercolors(w, good_colors, samplefn=samplefn)

    ax = mps.field(show=False, figsize=figsize)
    ax.scatter(x.x, x.y, c="black", alpha=alpha)
    if show:
        plt.show()
        

def scatter_location_models(
    loc_models, actions, W, samplefn="max", tol=0.1, figsize=8, alpha=0.5
):
    """
    Data-based visualization
    """
    for model in loc_models:
        print(model.name, model.n_components)
        X = actions[["x", "y"]]
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        x = X[pos_prob_idx]
        w = probs[pos_prob_idx]

        if model.n_components > len(colors):
            means = [m.mean for m in model.submodels]
            good_colors = color_submodels(means, colors)
        else:
            good_colors = colors
        c = scattercolors(w, good_colors, samplefn=samplefn)

        ax = mps.field(show=False, figsize=figsize)
        ax.scatter(x.x, x.y, c=c, alpha=alpha)
        plt.show()


def scatter_direction_models(
    dir_models, actions, X, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for model in dir_models:
        print(model.name, model.n_components)
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        w = probs[pos_prob_idx]
        c = scattercolors(w, samplefn=samplefn)

        axloc, axmov = dual_axes()
        field(axloc)
        movement(axmov)

        x = actions[pos_prob_idx]
        axloc.scatter(x.x, x.y, c=c, alpha=alpha)
        axmov.scatter(x.dx, x.dy, c=c, alpha=alpha)
        plt.show()


def hist_direction_model(
    dir_model,
    actions,
    W,
    samplefn="max",
    tol=0.1,
    figsize=4,
    alpha=0.5,
    projection="polar",
    bins=20,
    show=False,
):
    X = actions["mov_angle_a0"]
    probs = dir_model.predict_proba(X, W[dir_model.name].values)
    probs = np.nan_to_num(probs)
    pos_prob_idx = probs.sum(axis=1) > tol
    w = probs[pos_prob_idx]
    c = scattercolors(w, samplefn=samplefn)

    axpol = plt.subplot(111, projection=projection)
    plt.gcf().set_size_inches((figsize, figsize))

    x = actions[pos_prob_idx]
    for p, c in zip(w.T, colors):
        p = p.flatten()
        axpol.hist(x.mov_angle_a0, weights=p.flatten(), color=c, alpha=alpha, bins=bins)
    if show:
        plt.show()




def hist_direction_models(
    dir_models, actions, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for model in dir_models:
        print(model.name, model.n_components)
        X = actions["mov_angle_a0"]
        probs = model.predict_proba(X, W[model.name].values)
        probs = np.nan_to_num(probs)
        pos_prob_idx = probs.sum(axis=1) > tol
        w = probs[pos_prob_idx]
        c = scattercolors(w, samplefn=samplefn)

        # axloc, axmov = dual_axes()
        # field(axloc)
        # movement(axmov)
        axloc, axpol = loc_angle_axes()

        x = actions[pos_prob_idx]
        axloc.scatter(x.x, x.y, c=c, alpha=alpha)
        for p, c in zip(w.T, colors):
            p = p.flatten()
            axpol.hist(
                x.mov_angle_a0, weights=p.flatten(), color=c, alpha=alpha, bins=100
            )
        # axpol.hist(x.mov_angle_a0, c=c, alpha=alpha)
        plt.show()


def model_vs_data(
    dir_models, loc_models, actions, W, samplefn="max", tol=0.1, figsize=4, alpha=0.5
):
    for loc_model in loc_models:
        print(loc_model.name, loc_model.n_components)
        for i, gauss in enumerate(loc_model.submodels):
            # axloc, axpol = dual_axes()
            # # vis.add_ellips(axloc,gauss.mean,gauss.cov)
            # draw_contour(axloc, gauss, cmap="Blues")
            for dir_model in dir_models:
                if f"{loc_model.name}_{i}" == dir_model.name:

                    print(dir_model.name, dir_model.n_components)
                    axcol, axpol = loc_angle_axes()
                    draw_contour(axcol, gauss, cmap="Blues")
                    draw_vonmises_pdfs(axpol, dir_model)
                    plt.show()

                    X = actions["mov_angle_a0"]
                    probs = dir_model.predict_proba(X, W[dir_model.name].values)
                    probs = np.nan_to_num(probs)
                    pos_prob_idx = probs.sum(axis=1) > tol
                    w = probs[pos_prob_idx]
                    c = scattercolors(w, samplefn=samplefn)

                    # axloc, axmov = dual_axes()
                    # field(axloc)
                    # movement(axmov)
                    axloc, axpol = loc_angle_axes()

                    x = actions[pos_prob_idx]
                    axloc.scatter(x.x, x.y, c=c, alpha=alpha)
                    for p, c in zip(w.T, colors):
                        p = p.flatten()
                        axpol.hist(
                            x.mov_angle_a0,
                            weights=p.flatten(),
                            color=c,
                            alpha=alpha,
                            bins=100,
                        )
                    # axpol.hist(x.mov_angle_a0, c=c, alpha=alpha)
                    plt.show()


from scipy.spatial import Delaunay





def sample(probs):
    return np.random.choice(len(probs), p=probs / sum(probs))


def scattercolors(weights, colors=colors, samplefn="max"):
    if samplefn == "max":
        labels = np.argmax(weights, axis=1)
    else:
        labels = np.apply_along_axis(sample, axis=1, arr=weights)

    pcolors = [colors[l % len(colors)] for l in labels]
    return pcolors


#################################
# EXPERIMENTS VISUALIZATION
#################################


def savefigure(figname):
    plt.savefig(figname,dpi=300,
               bbox_inches="tight",
               pad_inches=0.0
               )
    
    
def show_component_differences(loc_models, dir_models, vec_p1, vec_p2, name1, name2, save=True):
    
    # determine colors of dir sub models
    difference = vec_p1 - vec_p2
    cmap = mpl.cm.get_cmap('bwr_r')
    
    for loc_model in loc_models:
        
        mini = min(difference.loc[difference.index.str.contains(f"^{loc_model.name}_")])
        maxi = max(difference.loc[difference.index.str.contains(f"^{loc_model.name}_")])
        ab = max(abs(mini), abs(maxi))
        
        if (ab == 0):
            ab = 0.0001
        
        norm = mpl.colors.DivergingNorm(vcenter=0, vmin=-ab,
                                               vmax = ab)
        
        print(loc_model.name, loc_model.n_components)
        ax = mps.field(show=False, figsize=8)
        
        
        am_subclusters = []
        for a, _ in enumerate(loc_model.submodels):
            for dir_model in dir_models:
                if f"{loc_model.name}_{a}" == dir_model.name:
                    am_subclusters.append(dir_model.n_components)
        
        am_subclusters = np.array(am_subclusters)
                
        for i, gauss in enumerate(loc_model.submodels):
                      
            if (am_subclusters == 1).all():
                    add_ellips(ax, gauss.mean, gauss.cov, 
                                   color=cmap(norm(difference.loc[f"{loc_model.name}_{i}_0"])), alpha=1)
            
            else:
                add_ellips(ax, gauss.mean, gauss.cov, color='gainsboro')

                x, y = gauss.mean
                for dir_model in dir_models:
                    if f"{loc_model.name}_{i}" == dir_model.name:
                        print(dir_model.name, dir_model.n_components)

                        for j, vonmises in enumerate(dir_model.submodels):
                                dx = np.cos(vonmises.loc)[0]
                                dy = np.sin(vonmises.loc)[0]
                                add_arrow(ax, x, y, 10*dx, 10*dy, 
                                              fc=cmap(norm(difference.loc[f"{loc_model.name}_{i}_{j}"])), 
                                              arrowsize=4.5, linewidth=1
                                             )
                        
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.065, pad=-0.05, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('bottom')
        cb.ax.tick_params(labelsize=16) 
        plt.axis("scaled")
        
        if save:
            savefigure(f"../figures/{name1}-{name2}-{loc_model.name}.png")
        else:
            plt.show()
