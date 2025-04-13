import re
from collections.abc import Callable
from typing import Any

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from information_flow_routes.graph import Component


def node_type_from_name(name: str) -> tuple[Component, int, int]:
    match = re.match(r"^([A-Z]+)\[(\d+)-(\d+)\]$", name)

    if match:
        component_str, layer_index, token_index = match.groups()
        return Component.from_string(component_str), int(layer_index), int(token_index)

    raise ValueError()


class Renderer:
    COLOR_PALETTE: str = "muted"

    def __init__(
        self,
        num_layers: int,
        string_tokens: list[str],
        target_token_index: int,
    ) -> None:
        self.string_tokens: list[str] = self._process_tokens(string_tokens)
        self.target_token_index: int = target_token_index
        self.num_layers: int = num_layers
        self.num_tokens: int = len(string_tokens)

        self._init_style()
        self._init_colors()
        self._init_render_params()

    def _process_tokens(self, tokens: list[str]) -> list[str]:
        if tokens:
            tokens = tokens.copy()
            tokens[0] = "<BOS>"

        return tokens

    def _init_style(self) -> None:
        sns.set_style("whitegrid", {"grid.linestyle": ":"})
        sns.set_context("paper", font_scale=1.1)
        plt.rcParams.update(
            {
                "font.family": "IBM Plex Mono",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def _init_colors(self) -> None:
        palette = sns.color_palette(self.COLOR_PALETTE, 6)
        self.colors: dict[str, Any] = {
            "background": "#ffffff",
            "layer_highlight": "#f8f9fa",
            "attention_edge": to_rgba(palette[0], 1.0),
            "feed_forward_edge": to_rgba(palette[1], 1.0),
            "attention_node": to_rgba(palette[2], 1.0),
            "feed_forward_node": to_rgba(palette[3], 1.0),
            "inactive_node": "#e0e0e0",
            "selected_token": to_rgba(palette[5], 1.0),
            "text": "#444444",
            "edge_text": "#666666",
        }

        self.component_active_colors: dict[Component, Any] = {
            Component.FEED_FORWARD: self.colors["feed_forward_node"],
            Component.POST_ATTENTION_RESIDUAL: self.colors["attention_node"],
            Component.POST_FEED_FORWARD_RESIDUAL: self.colors["attention_node"],
            Component.TOKEN: self.colors["attention_node"],
        }

        self.component_inactive_colors: dict[Component, Any] = {
            Component.FEED_FORWARD: self.colors["inactive_node"],
            Component.POST_ATTENTION_RESIDUAL: self.colors["inactive_node"],
            Component.POST_FEED_FORWARD_RESIDUAL: self.colors["inactive_node"],
            Component.TOKEN: self.colors["inactive_node"],
        }

    def _init_render_params(self) -> None:
        scale_factor = max(0.7, min(1.2, 10 / max(self.num_layers, self.num_tokens)))

        self.render_params: dict[str, float] = {
            "cell_h": 1.0 * scale_factor,
            "cell_w": 1.0 * scale_factor,
            "node_size": 0.18 * scale_factor,
            "layer_corner_radius": 0.1 * scale_factor,
            "edge_alpha": 0.85,
            "node_alpha": 0.9,
            "inactive_alpha": 0.3,
            "font_size": 10 * scale_factor,
            "title_font_size": 12 * scale_factor,
            "label_font_size": 10 * scale_factor,
            "edge_width_base": 0.8 + 2.0 * scale_factor,
        }

    def _calculate_figsize(self) -> tuple[float, float]:
        width: float = max(7, 4 + 0.5 * self.num_tokens)
        height: float = max(6, 3 + 0.5 * self.num_layers)

        aspect_ratio: float = width / height
        if aspect_ratio > 1.5:
            height = width / 1.5
        elif aspect_ratio < 0.67:
            width = height * 0.67

        return width, height

    def _get_all_node_names(self) -> list[str]:
        node_names: list[str] = []

        for layer_index in range(self.num_layers):
            for token_index in range(self.num_tokens):
                post_attention = Component.POST_ATTENTION_RESIDUAL.name(
                    token_index, layer_index
                )
                node_names.append(post_attention)

                feed_forward = Component.FEED_FORWARD.name(token_index, layer_index)
                node_names.append(feed_forward)

                post_feed_forward = Component.POST_FEED_FORWARD_RESIDUAL.name(
                    token_index, layer_index
                )
                node_names.append(post_feed_forward)

        for token_index in range(self.num_tokens):
            token_node = Component.TOKEN.name(token_index)
            node_names.append(token_node)

        return node_names

    def _calculate_node_positions(
        self, node_names: list[str]
    ) -> dict[str, tuple[float, float]]:
        node_positions: dict[str, tuple[float, float]] = {}

        for name in node_names:
            node_type, layer_index, token_index = node_type_from_name(name)

            match node_type:
                case Component.POST_ATTENTION_RESIDUAL:
                    x = self._x_scale(token_index)
                    y = self._y_scale(layer_index) + self.render_params["cell_h"] / 4
                case Component.FEED_FORWARD:
                    x = (
                        self._x_scale(token_index)
                        + 5 * self.render_params["cell_w"] / 16
                    )
                    y = self._y_scale(layer_index)
                case Component.POST_FEED_FORWARD_RESIDUAL:
                    x = self._x_scale(token_index)
                    y = self._y_scale(layer_index) - self.render_params["cell_h"] / 4
                case Component.TOKEN:
                    x = self._x_scale(token_index)
                    y = self._y_scale(-0.5) + self.render_params["cell_h"] / 4
                case _:
                    raise ValueError(f"Unknown node type for name {name}")

            node_positions[name] = (x, y)

        return node_positions

    def _x_scale(self, token_index: int | float) -> float:
        return (token_index + 1.5) * self.render_params["cell_w"]

    def _y_scale(self, layer_index: int | float) -> float:
        return (self.num_layers - layer_index + 1.5 - 1) * self.render_params["cell_h"]

    def _draw_circle_node(
        self, axis: Axes, pos: tuple[float, float], active: bool = True
    ) -> None:
        component = Component.POST_ATTENTION_RESIDUAL
        if active:
            facecolor = self.component_active_colors[component]
            linewidth = 0.75
            alpha = self.render_params["node_alpha"]
        else:
            facecolor = self.component_inactive_colors[component]
            linewidth = 0.5
            alpha = self.render_params["inactive_alpha"]
        circle = Circle(
            pos,
            radius=self.render_params["node_size"] / 2,
            facecolor=facecolor,
            edgecolor="#444444",
            linewidth=linewidth,
            zorder=4,
            alpha=alpha,
        )
        axis.add_patch(circle)

    def _draw_rectangle_node(
        self, axis: Axes, center: tuple[float, float], active: bool = True
    ) -> None:
        component = Component.FEED_FORWARD
        if active:
            facecolor = self.component_active_colors[component]
            linewidth = 0.75
            alpha = self.render_params["node_alpha"]
        else:
            facecolor = self.component_inactive_colors[component]
            linewidth = 0.5
            alpha = self.render_params["inactive_alpha"]
        size = self.render_params["node_size"]
        rectangle = Rectangle(
            (center[0] - size / 2, center[1] - size / 2),
            width=size,
            height=size,
            facecolor=facecolor,
            edgecolor="#444444",
            linewidth=linewidth,
            zorder=1,
            alpha=alpha,
        )
        axis.add_patch(rectangle)

    def _draw_layer_backgrounds(self, axis: Axes) -> None:
        for layer in range(self.num_layers):
            if layer % 2 == 1:
                y_pos: float = self._y_scale(layer)
                rect = patches.FancyBboxPatch(
                    (self._x_scale(-0.75), y_pos - self.render_params["cell_h"] / 2),
                    width=self._x_scale(self.num_tokens - 0.25) - self._x_scale(-0.75),
                    height=self.render_params["cell_h"],
                    boxstyle=(
                        f"round,pad=0,rounding_size={self.render_params['layer_corner_radius']}"
                    ),
                    facecolor=self.colors["layer_highlight"],
                    edgecolor="none",
                    zorder=1,
                )
                axis.add_patch(rect)
            text = axis.text(
                self._x_scale(-0.75),
                self._y_scale(layer),
                f"L{layer}",
                ha="center",
                va="center",
                fontweight="semibold",
                fontsize=self.render_params["label_font_size"],
                color=self.colors["text"],
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )

    def _draw_edges(
        self, axis: Axes, node_positions: dict[str, tuple[float, float]], graph: Any
    ) -> None:
        if not graph or not graph.edges:
            return

        edges = list(graph.edges(data=True))
        weights = np.array([data["weight"] for _, _, data in edges])
        max_weight = max(weights.max(), 1e-6)
        norm_weights = weights / max_weight

        attention_color_map = LinearSegmentedColormap.from_list(
            "attention_color_map",
            [
                to_rgba(self.colors["attention_edge"], 0.3),
                to_rgba(self.colors["attention_edge"], 1.0),
            ],
        )
        feed_forward_color_map = LinearSegmentedColormap.from_list(
            "feed_forward_color_map",
            [
                to_rgba(self.colors["feed_forward_edge"], 0.3),
                to_rgba(self.colors["feed_forward_edge"], 1.0),
            ],
        )

        sorted_edges = sorted(zip(edges, norm_weights), key=lambda x: x[1])
        for (u, v, data), norm_weight in sorted_edges:
            if u not in node_positions or v not in node_positions:
                continue
            start_pos: tuple[float, float] = node_positions[u]
            end_pos: tuple[float, float] = node_positions[v]
            u_type, _, _ = node_type_from_name(u)
            v_type, _, _ = node_type_from_name(v)
            is_feed_forward_edge: bool = (
                u_type == Component.FEED_FORWARD or v_type == Component.FEED_FORWARD
            )
            color = (
                feed_forward_color_map(norm_weight)
                if is_feed_forward_edge
                else attention_color_map(norm_weight)
            )
            width: float = 0.8 + 2.0 * np.sqrt(norm_weight) * (
                self.render_params["edge_width_base"] / 2
            )
            axis.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                color=color,
                linewidth=width,
                zorder=2,
                alpha=min(0.6 + 0.4 * norm_weight, self.render_params["edge_alpha"]),
                solid_capstyle="round",
            )

    def _draw_nodes(
        self,
        axis: Axes,
        node_positions: dict[str, tuple[float, float]],
        graph: Any,
        node_names: list[str],
    ) -> None:
        for name in node_names:
            node_type, _, _ = node_type_from_name(name)
            pos: tuple[float, float] = node_positions[name]
            is_active: bool = bool(graph and (name in graph.nodes))

            if node_type == Component.FEED_FORWARD:
                continue

            self._draw_circle_node(axis, pos, active=is_active)

        for name in node_names:
            node_type, _, _ = node_type_from_name(name)

            if node_type != Component.FEED_FORWARD:
                continue

            pos: tuple[float, float] = node_positions[name]
            is_active: bool = bool(graph and (name in graph.nodes))

            if is_active:
                self._draw_rectangle_node(axis, center=pos, active=is_active)

    def _draw_labels(self, axis: Axes) -> None:
        for token_index, token_text in enumerate(self.string_tokens):
            display_text: str = token_text.replace(" ", "_")
            text = axis.text(
                self._x_scale(token_index),
                self._y_scale(-1.5),
                display_text,
                ha="center",
                va="top",
                fontsize=self.render_params["font_size"],
                fontweight=(
                    "medium" if token_index == self.target_token_index else "normal"
                ),
                color=self.colors["text"],
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )

        x: float = self._x_scale(self.target_token_index)
        y: float = self._y_scale(self.num_layers)
        size: float = 0.15 * (self.render_params["node_size"] / 0.18)
        triangle = plt.Polygon(
            [[x, y + size], [x + size, y - size / 2], [x - size, y - size / 2]],
            closed=True,
            facecolor=self.colors["selected_token"],
            edgecolor="black",
            linewidth=0.75,
            zorder=5,
            alpha=0.9,
        )

        axis.add_patch(triangle)

    def _draw_legend(self, axis: Axes) -> None:
        legend_elements = [
            patches.Patch(
                facecolor=self.colors["attention_node"],
                edgecolor="#444444",
                linewidth=0.75,
                label="Attention Output",
            ),
            patches.Patch(
                facecolor=self.colors["feed_forward_node"],
                edgecolor="#444444",
                linewidth=0.75,
                label="Feed-Forward Network",
            ),
            patches.Patch(
                facecolor=self.colors["attention_edge"],
                edgecolor=None,
                label="Attention Path",
            ),
            patches.Patch(
                facecolor=self.colors["feed_forward_edge"],
                edgecolor=None,
                label="Feed-Forward Path",
            ),
        ]

        legend = axis.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            frameon=True,
            fontsize=self.render_params["font_size"],
            fancybox=True,
            framealpha=0.9,
        )

        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#cccccc")

    def _finalize_plot(self, axis: Axes) -> None:
        axis.set_xlim(self._x_scale(-1.5), self._x_scale(self.num_tokens + 0.5))
        axis.set_ylim(self._y_scale(self.num_layers + 1.5), self._y_scale(-1 - 0.5))
        axis.set_xticks([])
        axis.set_yticks([])

        for spine in axis.spines.values():
            spine.set_visible(False)

        axis.set_title(
            f'Information Flow for Token "{self.string_tokens[self.target_token_index]}" '
            f"(position {self.target_token_index})",
            fontsize=self.render_params["title_font_size"],
            fontweight="bold",
            pad=15,
        )

        for layer in range(self.num_layers):
            axis.axhline(
                y=self._y_scale(layer),
                color="#f0f0f0",
                linestyle="-",
                linewidth=0.5,
                zorder=0,
            )

        for token in range(self.num_tokens):
            axis.axvline(
                x=self._x_scale(token),
                color="#f0f0f0",
                linestyle="-",
                linewidth=0.5,
                zorder=0,
            )

        plt.tight_layout()

    def _setup_figure(self) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(
            figsize=self._calculate_figsize(), facecolor=self.colors["background"]
        )

        axis.set_facecolor(self.colors["background"])
        axis.set_aspect("equal", adjustable="box")

        return figure, axis

    def _create_error_plot(self, message: str) -> tuple[Figure, Axes]:
        figure, axis = plt.subplots(figsize=(8, 6))

        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        plt.tight_layout()

        return figure, axis

    def plot(
        self,
        graph: Any,
        export_pdf: bool = False,
        filename: str = "contribution_graph.pdf",
    ) -> tuple[Figure, Axes]:
        figure, axis = self._setup_figure()

        all_node_names: list[str] = self._get_all_node_names()
        node_positions = self._calculate_node_positions(all_node_names)

        self._draw_layer_backgrounds(axis)
        self._draw_edges(axis, node_positions, graph)
        self._draw_nodes(axis, node_positions, graph, all_node_names)
        self._draw_labels(axis)
        self._draw_legend(axis)
        self._finalize_plot(axis)

        if export_pdf:
            figure.savefig(filename, bbox_inches="tight", dpi=300)

        return figure, axis
