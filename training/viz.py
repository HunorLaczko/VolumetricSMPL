"""Interactive 3D mesh panels for the W&B report.

`wandb.Object3D` draws an OBJ unlit — a flat silhouette with no shading, which made
the surfaces unreadable. These are plotly `Mesh3d` figures with a matte clay material,
so form reads from the shading, plus an overlay whose traces can be toggled from the
legend.

Coordinate note: AMASS SMPL-X bodies here are **z-up**, with y as depth — measured from
a val body's bounding box, x spanned 1.48 m (arms), z 1.67 m (height) and y only 0.39 m.
The camera is placed accordingly; getting this wrong yields a body viewed end-on.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Matte and warm. Low specular with high roughness is what reads as clay rather than
# plastic; the ambient term stops unlit regions going pure black.
LIGHTING = dict(ambient=0.42, diffuse=0.88, specular=0.08, roughness=0.92, fresnel=0.06)
LIGHTPOS = dict(x=140, y=-260, z=220)

CLAY = '#c9b39b'
COLORS = {'gt': '#9a9a9a', 'ours': '#c8763c'}


def _axis():
    return dict(showbackground=False, showgrid=False, showticklabels=False,
                zeroline=False, visible=False)


def _layout(title, legend=False):
    return dict(
        title=dict(text=title, x=0.02, font=dict(size=13)),
        scene=dict(xaxis=_axis(), yaxis=_axis(), zaxis=_axis(),
                   aspectmode='data',
                   # z-up, viewed from the front (-y). See the module docstring.
                   camera=dict(eye=dict(x=0.0, y=-2.3, z=0.15),
                               up=dict(x=0, y=0, z=1))),
        margin=dict(l=0, r=0, t=34, b=0),
        showlegend=legend,
        legend=dict(itemsizing='constant', x=0.01, y=0.99),
        height=620,
    )


def clay(v: np.ndarray, f: np.ndarray, name: str, color: str = CLAY,
         opacity: float = 1.0) -> go.Mesh3d:
    """One mesh under the clay material.

    Coordinates are rounded to 0.01 mm. Full float64 repr would trail 17 significant
    digits into the HTML for each of ~140k vertices, so rounding roughly halves the
    payload.

    5 dp rather than 4 costs about 8 % of the payload and buys little: independent
    +-0.05 mm jitter across a 4 mm edge perturbs a normal by ~0.6 deg RMS, which against
    the 4.63 deg the extraction already carries is ~1 % in quadrature. It is kept
    because the panels still open comfortably at this size, not because the difference
    is large — if payload ever becomes the binding constraint, this is the first thing
    to give up.
    """
    v = np.round(v, 5)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
        flatshading=False, lighting=LIGHTING, lightposition=LIGHTPOS,
        hoverinfo='skip')


def single(v, f, title, color=CLAY) -> go.Figure:
    return go.Figure(data=[clay(v, f, title, color)], layout=_layout(title))


def overlay(meshes: dict, title: str) -> go.Figure:
    """All meshes in one scene, each toggleable from the legend.

    Opaque, not blended. The surfaces coincide to within millimetres, so with
    transparency they muddle into one another and plotly's depth sorting adds artefacts
    on top of that. Opaque means whichever is nearest the camera simply wins, and the
    legend is what you use to see a specific one — which is the intended workflow.
    """
    data = [clay(v, f, name, COLORS.get(name, CLAY), opacity=1.0)
            for name, (v, f) in meshes.items()]
    return go.Figure(data=data, layout=_layout(title, legend=True))


def to_wandb(fig: go.Figure, label: str = ''):
    """Wrap for W&B.

    `wandb.Html` with a real plotly bundle rather than `wandb.Plotly`, because the
    legend toggling and 3D camera are the point here and the HTML path is the one that
    preserves them verbatim.

    Prints the payload size, since that — not the extraction — is what decides whether a
    panel opens smoothly in a browser, and it is not otherwise visible anywhere.
    """
    import wandb
    html = fig.to_html(full_html=True, include_plotlyjs='cdn')
    if label:
        print(f'    panel {label:<28} {len(html) / 1e6:5.1f} MB')
    return wandb.Html(html, inject=False)
