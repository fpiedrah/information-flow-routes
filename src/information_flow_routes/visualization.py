import re

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.patches import Circle, Rectangle

from information_flow_routes.graph import Component, find_prediction_paths


def node_type_from_name(name):
    match = re.match(r"^([A-Z]+)\[(\d+)-(\d+)\]$", name)
    if match:
        type_code, layer, token = match.groups()
        mapping = {
            Component.TOKEN.value: "original",
            Component.FEED_FORWARD.value: "ffn",
            Component.POST_ATTENTION_RESIDUAL.value: "after_attn",
            Component.POST_FEED_FORWARD_RESIDUAL.value: "after_ffn",
        }
        return mapping.get(type_code, None), int(layer), int(token)
    return None, None, None


class InformationFlowGraphRenderer:
    def __init__(
        self,
        graph,
        tokens_str,
        start_token,
        threshold=0.04,
        figsize=None,
        color_palette="muted",
    ):
        self.graph = graph
        self.tokens_str = self._process_tokens(tokens_str)
        self.start_token = start_token
        self.threshold = threshold
        self.figsize = figsize
        self.color_palette = color_palette

        self.fig = None
        self.ax = None
        self.node_positions = {}
        self.sub_graph = None
        self.n_layers = 0
        self.n_tokens = len(tokens_str)
        self.valid_layers = []

        self._init_style()
        self._init_colors()
        self._init_render_params()
        self._create_subgraph()
        self._validate_graph()

    def _process_tokens(self, tokens):
        if tokens and len(tokens) > 0:
            tokens = tokens.copy()
            tokens[0] = "<BOS>"
        return tokens

    def _init_style(self):
        sns.set_style("whitegrid", {"grid.linestyle": ":"})
        sns.set_context("paper", font_scale=1.1)
        plt.rcParams.update(
            {
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def _init_colors(self):
        palette = sns.color_palette(self.color_palette, 6)
        self.colors = {
            "background": "#ffffff",
            "layer_highlight": "#f8f9fa",
            "attn_edge": to_rgba(palette[0], 1.0),
            "ffn_edge": to_rgba(palette[1], 1.0),
            "attn_node": to_rgba(palette[2], 1.0),
            "ffn_node": to_rgba(palette[3], 1.0),
            "inactive_node": "#e0e0e0",
            "selected_token": to_rgba(palette[5], 1.0),
            "text": "#444444",
            "edge_text": "#666666",
        }

    def _init_render_params(self):
        scale_factor = max(0.7, min(1.2, 10 / max(self.n_layers, self.n_tokens)))
        self.render_params = {
            "cellH": 1.0 * scale_factor,
            "cellW": 1.0 * scale_factor,
            "nodeSize": 0.18 * scale_factor,
            "layerCornerRadius": 0.1 * scale_factor,
            "edgeAlpha": 0.85,
            "nodeAlpha": 0.9,
            "inactiveAlpha": 0.3,
            "fontsize": 10 * scale_factor,
            "title_fontsize": 12 * scale_factor,
            "label_fontsize": 10 * scale_factor,
            "edge_width_base": 0.8 + 2.0 * scale_factor,
        }

    def _create_subgraph(self):
        sub_graphs = find_prediction_paths(self.graph, self.start_token, self.threshold)
        self.sub_graph = sub_graphs[0] if sub_graphs else None

    def _validate_graph(self):
        nodes_with_layers = [node_type_from_name(node) for node in self.graph.nodes]
        self.valid_layers = [
            layer for _, layer, _ in nodes_with_layers if layer is not None
        ]
        if not self.valid_layers:
            raise ValueError("No valid layers found in the graph!")
        self.n_layers = max(self.valid_layers) + 1
        self.n_tokens = len(self.tokens_str)
        self._init_render_params()

    def _calculate_figsize(self):
        width = max(7, 4 + 0.5 * self.n_tokens)
        height = max(6, 3 + 0.5 * self.n_layers)
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            height = width / 1.5
        elif aspect_ratio < 0.67:
            width = height * 0.67
        return (width, height)

    def _calculate_node_positions(self):
        self.node_positions = {}
        for node in self.graph.nodes:
            node_type, layer, token = node_type_from_name(node)
            if None in (node_type, layer, token):
                continue
            cx = self._x_scale(token)
            cy = self._y_scale(layer)
            if node_type == "after_attn":
                self.node_positions[node] = (cx, cy + self.render_params["cellH"] / 4)
            elif node_type == "after_ffn":
                self.node_positions[node] = (cx, cy - self.render_params["cellH"] / 4)
            elif node_type == "ffn":
                self.node_positions[node] = (
                    cx + 5 * self.render_params["cellW"] / 16,
                    cy,
                )
            elif node_type == "original":
                cy = self._y_scale(layer - 0.5)
                self.node_positions[node] = (cx, cy + self.render_params["cellH"] / 4)

    def _x_scale(self, token_idx):
        return (token_idx + 1.5) * self.render_params["cellW"]

    def _y_scale(self, layer_idx):
        return (self.n_layers - layer_idx + 1.5 - 1) * self.render_params["cellH"]

    def _draw_layer_backgrounds(self):
        for layer in range(self.n_layers):
            if layer % 2 == 1:
                y_pos = self._y_scale(layer)
                rect = patches.FancyBboxPatch(
                    (self._x_scale(-0.75), y_pos - self.render_params["cellH"] / 2),
                    width=self._x_scale(self.n_tokens - 0.25) - self._x_scale(-0.75),
                    height=self.render_params["cellH"],
                    boxstyle=f"round,pad=0,rounding_size={self.render_params['layerCornerRadius']}",
                    facecolor=self.colors["layer_highlight"],
                    edgecolor="none",
                    zorder=1,
                )
                self.ax.add_patch(rect)
            text = self.ax.text(
                self._x_scale(-0.75),
                self._y_scale(layer),
                f"L{layer}",
                ha="center",
                va="center",
                fontweight="semibold",
                fontsize=self.render_params["label_fontsize"],
                color=self.colors["text"],
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )

    def _draw_edges(self):
        if not self.sub_graph or self.sub_graph.number_of_edges() == 0:
            return
        edges = list(self.sub_graph.edges(data=True))
        weights = np.array([data["weight"] for _, _, data in edges])
        max_weight = max(weights.max(), 1e-6)
        norm_weights = weights / max_weight
        attn_cmap = LinearSegmentedColormap.from_list(
            "attn_cmap",
            [
                to_rgba(self.colors["attn_edge"], 0.3),
                to_rgba(self.colors["attn_edge"], 1.0),
            ],
        )
        ffn_cmap = LinearSegmentedColormap.from_list(
            "ffn_cmap",
            [
                to_rgba(self.colors["ffn_edge"], 0.3),
                to_rgba(self.colors["ffn_edge"], 1.0),
            ],
        )
        sorted_edges = sorted(zip(edges, norm_weights), key=lambda x: x[1])
        for (u, v, data), norm_weight in sorted_edges:
            if u not in self.node_positions or v not in self.node_positions:
                continue
            start_pos = self.node_positions[u]
            end_pos = self.node_positions[v]
            u_type, _, _ = node_type_from_name(u)
            v_type, _, _ = node_type_from_name(v)
            is_ffn_edge = u_type == "ffn" or v_type == "ffn"
            color = ffn_cmap(norm_weight) if is_ffn_edge else attn_cmap(norm_weight)
            width = 0.8 + 2.0 * np.sqrt(norm_weight) * (
                self.render_params["edge_width_base"] / 2
            )
            self.ax.plot(
                [start_pos[0], end_pos[0]],
                [start_pos[1], end_pos[1]],
                color=color,
                linewidth=width,
                zorder=2,
                alpha=min(0.6 + 0.4 * norm_weight, self.render_params["edgeAlpha"]),
                solid_capstyle="round",
            )

    def _draw_nodes(self):
        for node in self.graph.nodes:
            node_type, _, _ = node_type_from_name(node)
            if node not in self.node_positions or node_type not in [
                "after_attn",
                "after_ffn",
                "original",
            ]:
                continue
            pos = self.node_positions[node]
            is_active = node in self.sub_graph.nodes if self.sub_graph else False
            circle = Circle(
                pos,
                radius=self.render_params["nodeSize"] / 2,
                facecolor=(
                    self.colors["attn_node"]
                    if is_active
                    else self.colors["inactive_node"]
                ),
                edgecolor="#444444",
                linewidth=0.75 if is_active else 0.5,
                zorder=4,
                alpha=(
                    self.render_params["nodeAlpha"]
                    if is_active
                    else self.render_params["inactiveAlpha"]
                ),
            )
            self.ax.add_patch(circle)
        for node in self.graph.nodes:
            node_type, _, _ = node_type_from_name(node)
            if node not in self.node_positions or node_type != "ffn":
                continue
            pos = self.node_positions[node]
            is_active = node in self.sub_graph.nodes if self.sub_graph else False
            if is_active:
                size = self.render_params["nodeSize"]
                rectangle = Rectangle(
                    (pos[0] - size / 2, pos[1] - size / 2),
                    width=size,
                    height=size,
                    facecolor=self.colors["ffn_node"],
                    edgecolor="#444444",
                    linewidth=0.75,
                    zorder=4,
                    alpha=self.render_params["nodeAlpha"],
                )
                self.ax.add_patch(rectangle)

    def _draw_labels(self):
        for i, token_text in enumerate(self.tokens_str):
            display_text = token_text.replace(" ", "_")
            text = self.ax.text(
                self._x_scale(i),
                self._y_scale(-1.5),
                display_text,
                ha="center",
                va="top",
                fontsize=self.render_params["fontsize"],
                fontweight="medium" if i == self.start_token else "normal",
                color=self.colors["text"],
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="white"),
                    path_effects.Normal(),
                ]
            )
        x = self._x_scale(self.start_token)
        y = self._y_scale(self.n_layers)
        size = 0.15 * (self.render_params["nodeSize"] / 0.18)
        triangle = plt.Polygon(
            [[x, y + size], [x + size, y - size / 2], [x - size, y - size / 2]],
            closed=True,
            facecolor=self.colors["selected_token"],
            edgecolor="black",
            linewidth=0.75,
            zorder=5,
            alpha=0.9,
        )
        self.ax.add_patch(triangle)

    def _draw_legend(self):
        legend_elements = [
            patches.Patch(
                facecolor=self.colors["attn_node"],
                edgecolor="#444444",
                linewidth=0.75,
                label="Attention Output",
            ),
            patches.Patch(
                facecolor=self.colors["ffn_node"],
                edgecolor="#444444",
                linewidth=0.75,
                label="Feed-Forward Network",
            ),
            patches.Patch(
                facecolor=self.colors["attn_edge"],
                edgecolor=None,
                label="Attention Path",
            ),
            patches.Patch(
                facecolor=self.colors["ffn_edge"], edgecolor=None, label="FFN Path"
            ),
        ]
        leg = self.ax.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=4,
            frameon=True,
            fontsize=self.render_params["fontsize"],
            fancybox=True,
            framealpha=0.9,
        )
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#cccccc")

    def _finalize_plot(self):
        self.ax.set_xlim(self._x_scale(-1.5), self._x_scale(self.n_tokens + 0.5))
        self.ax.set_ylim(self._y_scale(self.n_layers + 1.5), self._y_scale(-1 - 0.5))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_title(
            f'Information Flow for Token "{self.tokens_str[self.start_token]}" '
            f"(position {self.start_token})",
            fontsize=self.render_params["title_fontsize"],
            fontweight="bold",
            pad=15,
        )
        for layer in range(self.n_layers):
            self.ax.axhline(
                y=self._y_scale(layer),
                color="#f0f0f0",
                linestyle="-",
                linewidth=0.5,
                zorder=0,
            )
        for token in range(self.n_tokens):
            self.ax.axvline(
                x=self._x_scale(token),
                color="#f0f0f0",
                linestyle="-",
                linewidth=0.5,
                zorder=0,
            )
        plt.tight_layout()

    def plot(self, export_pdf=False, filename="contribution_graph.pdf"):
        if not self.valid_layers:
            self._create_error_plot("No valid layers found in the graph!")
            return self.fig, self.ax
        if not self.sub_graph or self.sub_graph.number_of_nodes() == 0:
            self._create_error_plot(
                "No significant contributions found with current threshold."
            )
            return self.fig, self.ax
        self._setup_figure()
        self._calculate_node_positions()
        self._draw_layer_backgrounds()
        self._draw_edges()
        self._draw_nodes()
        self._draw_labels()
        self._draw_legend()
        self._finalize_plot()
        if export_pdf:
            self.fig.savefig(filename, bbox_inches="tight", dpi=300)
        return self.fig, self.ax

    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(
            figsize=self.figsize or self._calculate_figsize(),
            facecolor=self.colors["background"],
        )
        self.ax.set_facecolor(self.colors["background"])
        self.ax.set_aspect("equal", adjustable="box")

    def _create_error_plot(self, message):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
        plt.tight_layout()
