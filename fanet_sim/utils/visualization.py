"""
visualization.py — Matplotlib animation of the FANET simulation.

Produces a real-time interactive animation (vis.show()) or saves a GIF
(vis.animate(save_path="episode.gif")).

Layout:
    Left panel  — Network map: drone positions, links, packet flashes.
    Right panel — Live metric bar: PDR and avg delay updated each frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from fanet_sim import config
from fanet_sim.utils.metrics import compute_step_metrics

if TYPE_CHECKING:
    from fanet_sim.envs.fanet_env import FANETEnv


# Use Agg backend when saving, let matplotlib pick for interactive
_COLORS = {
    "M": "#3a86ff",        # blue — mission drones
    "C": "#fb8500",        # orange — communication drones
    "GS": "#06d6a0",       # green — ground station
    "link": "#adb5bd",     # gray — wireless links
    "tx": "#ef233c",       # red — packet transmission flash
}


class FANETVisualizer:
    """Animate or display the FANET simulation.

    Example usage::

        env = FANETEnv()
        env.reset()

        vis = FANETVisualizer(env)
        vis.animate(save_path="episode.gif")   # save GIF
        # or
        vis.show()                              # interactive window

    Args:
        env:         The FANETEnv instance to visualise.  The visualiser
                     calls env.step() internally for each frame.
        interval_ms: Milliseconds between animation frames.
        max_steps:   Number of simulation steps to animate.  Defaults to
                     config.MAX_STEPS.
    """

    def __init__(
        self,
        env: "FANETEnv",
        interval_ms: int = 100,
        max_steps: Optional[int] = None,
    ) -> None:
        self.env = env
        self.interval_ms = interval_ms
        self.max_steps = max_steps or config.MAX_STEPS

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Run the animation in an interactive matplotlib window."""
        self._run(save_path=None)

    def animate(self, save_path: str = "episode.gif") -> None:
        """Run the animation and save it to *save_path* (GIF or MP4).

        Args:
            save_path: File path for the saved animation.
        """
        self._run(save_path=save_path)

    # ------------------------------------------------------------------
    # Core animation logic
    # ------------------------------------------------------------------

    def _run(self, save_path: Optional[str]) -> None:
        """Build and run (or save) the matplotlib animation.

        Args:
            save_path: If given, save the animation here and close the figure.
                       If None, display interactively.
        """
        fig, (ax_net, ax_metrics) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            gridspec_kw={"width_ratios": [3, 1]},
        )
        fig.patch.set_facecolor("#1a1a2e")
        self._setup_network_axis(ax_net)
        self._setup_metric_axis(ax_metrics)

        # We store artist handles so the update function can mutate them.
        artists: dict = {}
        self._init_artists(ax_net, ax_metrics, artists)

        # Step state — carried across frames via a mutable container
        state: dict = {"done": False}

        def update(frame: int) -> list:
            if state["done"]:
                return list(artists.values())

            obs, _, dones, _ = self.env.step()
            state["done"] = all(dones.values())

            metrics = compute_step_metrics(self.env)
            self._draw_network(ax_net, artists, metrics)
            self._draw_metrics(ax_metrics, artists, metrics)
            return list(artists.values())

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=self.max_steps,
            interval=self.interval_ms,
            blit=False,
            repeat=False,
        )

        if save_path:
            writer = self._pick_writer(save_path)
            print(f"[vis] Saving animation to {save_path} …")
            anim.save(save_path, writer=writer, dpi=80)
            plt.close(fig)
            print(f"[vis] Saved {save_path}")
        else:
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # Axis setup
    # ------------------------------------------------------------------

    def _setup_network_axis(self, ax: plt.Axes) -> None:
        """Configure the network map axis."""
        ax.set_facecolor("#16213e")
        ax.set_xlim(0, config.WIDTH)
        ax.set_ylim(0, config.HEIGHT)
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_title("FANET Network", color="white", fontsize=11)

    def _setup_metric_axis(self, ax: plt.Axes) -> None:
        """Configure the live metric panel axis."""
        ax.set_facecolor("#16213e")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Live Metrics", color="white", fontsize=11)

    # ------------------------------------------------------------------
    # Artist initialisation
    # ------------------------------------------------------------------

    def _init_artists(
        self,
        ax_net: plt.Axes,
        ax_metrics: plt.Axes,
        artists: dict,
    ) -> None:
        """Create all matplotlib artists and store them in *artists*.

        Args:
            ax_net:     Network map axis.
            ax_metrics: Metrics panel axis.
            artists:    Dict to populate with named artist references.
        """
        drones = self.env.drones
        gs = self.env.gs_position

        # Ground station
        artists["gs"] = ax_net.plot(
            gs[0], gs[1],
            marker="*", markersize=15,
            color=_COLORS["GS"], zorder=5, label="GS",
        )[0]

        # Drone scatter (two separate for M and C)
        m_pos = np.array([d.position for d in drones if d.drone_type == "M"])
        c_pos = np.array([d.position for d in drones if d.drone_type == "C"])

        artists["m_scatter"] = ax_net.scatter(
            m_pos[:, 0] if len(m_pos) else [],
            m_pos[:, 1] if len(m_pos) else [],
            s=60, c=_COLORS["M"], zorder=4, label="M-drone",
        )
        artists["c_scatter"] = ax_net.scatter(
            c_pos[:, 0] if len(c_pos) else [],
            c_pos[:, 1] if len(c_pos) else [],
            s=60, c=_COLORS["C"], zorder=4, label="C-drone",
            marker="^",
        )

        # Link lines — stored as a list of Line2D, recreated each frame
        artists["link_lines"] = []

        # Transmission flashes — list of Line2D
        artists["tx_lines"] = []

        # Drone ID labels
        artists["labels"] = [
            ax_net.text(
                d.position[0], d.position[1] + 12,
                str(d.drone_id),
                color="white", fontsize=7, ha="center", zorder=6,
            )
            for d in drones
        ]

        # Step counter text
        artists["step_text"] = ax_net.text(
            0.02, 0.97, "Step: 0",
            transform=ax_net.transAxes,
            color="white", fontsize=9, va="top",
        )

        # Legend
        ax_net.legend(
            loc="lower right",
            facecolor="#1a1a2e",
            edgecolor="#444",
            labelcolor="white",
            fontsize=8,
        )

        # Metric text block
        artists["metric_text"] = ax_metrics.text(
            0.1, 0.85,
            self._metric_str({}),
            transform=ax_metrics.transAxes,
            color="white", fontsize=9, va="top", family="monospace",
        )

    # ------------------------------------------------------------------
    # Per-frame drawing
    # ------------------------------------------------------------------

    def _draw_network(
        self,
        ax: plt.Axes,
        artists: dict,
        metrics: dict,
    ) -> None:
        """Update drone positions, links, and transmission flashes.

        Args:
            ax:      Network map axis.
            artists: Shared artist dict.
            metrics: Current-step metrics dict.
        """
        drones = self.env.drones

        # Update drone positions
        m_pos = np.array([d.position for d in drones if d.drone_type == "M"])
        c_pos = np.array([d.position for d in drones if d.drone_type == "C"])

        if len(m_pos):
            artists["m_scatter"].set_offsets(m_pos)
        if len(c_pos):
            artists["c_scatter"].set_offsets(c_pos)

        # Update labels
        for label, drone in zip(artists["labels"], drones):
            label.set_position((drone.position[0], drone.position[1] + 12))

        # Remove old link lines
        for line in artists["link_lines"]:
            line.remove()
        artists["link_lines"] = []

        # Draw active links
        drawn: set = set()
        for drone in drones:
            for nid in drone.neighbors:
                key = tuple(sorted((drone.drone_id, nid)))
                if key in drawn:
                    continue
                drawn.add(key)
                nbr = self.env.get_drone_by_id(nid)
                line, = ax.plot(
                    [drone.position[0], nbr.position[0]],
                    [drone.position[1], nbr.position[1]],
                    color=_COLORS["link"], linewidth=0.6, alpha=0.5, zorder=1,
                )
                artists["link_lines"].append(line)

        # Remove old tx flashes
        for line in artists["tx_lines"]:
            line.remove()
        artists["tx_lines"] = []

        # Draw transmission flashes
        gs = self.env.gs_position
        drone_map = {d.drone_id: d for d in drones}

        for (src_id, dst) in self.env.tx_events:
            src_drone = drone_map.get(src_id)
            if src_drone is None:
                continue
            if dst == "GS":
                end_pos = gs
            else:
                dst_drone = drone_map.get(dst)
                if dst_drone is None:
                    continue
                end_pos = dst_drone.position

            line, = ax.plot(
                [src_drone.position[0], end_pos[0]],
                [src_drone.position[1], end_pos[1]],
                color=_COLORS["tx"], linewidth=1.5, alpha=0.7, zorder=3,
            )
            artists["tx_lines"].append(line)

        # Step counter
        artists["step_text"].set_text(f"Step: {self.env.step_count}")

    def _draw_metrics(
        self,
        ax: plt.Axes,
        artists: dict,
        metrics: dict,
    ) -> None:
        """Update the live metrics panel.

        Args:
            ax:      Metrics panel axis.
            artists: Shared artist dict.
            metrics: Current-step metrics dict.
        """
        artists["metric_text"].set_text(self._metric_str(metrics))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metric_str(metrics: dict) -> str:
        """Format metric dict as a readable multi-line string.

        Args:
            metrics: Dict from compute_step_metrics().

        Returns:
            Formatted string for display.
        """
        if not metrics:
            return "Waiting…"
        return (
            f"Step : {metrics.get('step', 0)}\n\n"
            f"PDR  : {metrics.get('PDR', 0):.3f}\n"
            f"Delay: {metrics.get('avg_delay', 0):.1f} steps\n\n"
            f"Pkts\n"
            f"  gen : {metrics.get('generated', 0)}\n"
            f"  dlv : {metrics.get('delivered', 0)}\n"
            f"  drp : {metrics.get('dropped', 0)}\n\n"
            f"Links: {metrics.get('active_links', 0)}\n"
            f"Conn : {'yes' if metrics.get('connected') else 'no'}"
        )

    @staticmethod
    def _pick_writer(save_path: str):
        """Choose an animation writer based on the file extension.

        Args:
            save_path: Output file path.

        Returns:
            A matplotlib animation writer instance.
        """
        if save_path.endswith(".gif"):
            try:
                return animation.PillowWriter(fps=10)
            except Exception:
                return animation.FFMpegWriter(fps=10)
        return animation.FFMpegWriter(fps=10)
