"""RIR recipe utilities and the optional Habitat-Sim FOA renderer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SCENE_CONFIG_SHA256 = "b56050135d429839ca7066f335de51897d27e9741810bff8e13972fe539b0f5f"
MATERIALS_CONFIG_SHA256 = "77cc3b16f45843b1138020837d4f78aeaf0abda2427f658291b632d84afbb1f4"


def verify_file_sha256(path: str | Path, expected: str) -> None:
    """Require a renderer asset to match the released recipe."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"Unexpected renderer asset checksum for {path}: {actual}")


def time_bin_centers(duration_s: float, interval_s: float) -> np.ndarray:
    """Return normalized center times for all fixed-width render bins."""

    count = int(round(float(duration_s) / float(interval_s)))
    if count < 1:
        raise ValueError("duration_s / interval_s must produce at least one bin")
    starts = np.arange(count, dtype=np.float32) * float(interval_s)
    ends = np.minimum(starts + float(interval_s), float(duration_s))
    return (0.5 * (starts + ends) / float(duration_s)).astype(np.float32)


def _warp_time(alpha: np.ndarray, profile: str, parameters: dict[str, Any]) -> np.ndarray:
    x = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    if profile == "linear":
        return x
    if profile == "ease_in":
        return np.power(x, float(parameters.get("power", 2.0))).astype(np.float32)
    if profile == "ease_out":
        return (1.0 - np.power(1.0 - x, float(parameters.get("power", 2.0)))).astype(
            np.float32
        )
    if profile in {"smoothstep", "ease_in_out"}:
        return (x * x * (3.0 - 2.0 * x)).astype(np.float32)
    if profile == "pause_start":
        pause = float(parameters.get("pause_fraction", 0.18))
        return np.clip((x - pause) / max(1.0 - pause, 1e-6), 0.0, 1.0)
    if profile == "pause_end":
        pause = float(parameters.get("pause_fraction", 0.18))
        return np.clip(x / max(1.0 - pause, 1e-6), 0.0, 1.0)
    raise ValueError(f"Unsupported time profile: {profile}")


def trajectory_positions(recipe: dict[str, Any]) -> np.ndarray:
    """Evaluate one released source trajectory at RIR keyframe centers."""

    alpha = time_bin_centers(
        float(recipe.get("duration_s", 10.0)),
        float(recipe.get("keyframe_interval_s", 0.25)),
    )
    alpha = _warp_time(
        alpha,
        str(recipe.get("time_profile", "linear")),
        dict(recipe.get("time_profile_params") or {}),
    )
    start = np.asarray(recipe["start_source_position"], dtype=np.float32)
    end = np.asarray(recipe["end_source_position"], dtype=np.float32)
    control = np.asarray(
        recipe.get("trajectory_control_points", [start.tolist(), end.tolist()]),
        dtype=np.float32,
    )
    path_type = str(recipe.get("trajectory_path_type", "linear"))
    if path_type == "linear":
        return (1.0 - alpha[:, None]) * start + alpha[:, None] * end
    if path_type == "quadratic_bezier":
        if control.shape != (3, 3):
            raise ValueError("quadratic_bezier requires three xyz control points")
        t = alpha[:, None]
        return (
            ((1.0 - t) ** 2) * control[0]
            + 2.0 * (1.0 - t) * t * control[1]
            + (t**2) * control[2]
        ).astype(np.float32)
    if path_type == "polyline":
        if control.ndim != 2 or control.shape[0] < 2 or control.shape[1] != 3:
            raise ValueError("polyline requires at least two xyz control points")
        segments = control[1:] - control[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        total = float(np.sum(lengths))
        if total <= 1e-8:
            return np.repeat(control[:1], len(alpha), axis=0)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths) / total])
        output = np.empty((len(alpha), 3), dtype=np.float32)
        for index, value in enumerate(alpha):
            segment = int(np.searchsorted(cumulative, value, side="right") - 1)
            segment = min(max(segment, 0), len(lengths) - 1)
            width = max(float(cumulative[segment + 1] - cumulative[segment]), 1e-8)
            local = float((value - cumulative[segment]) / width)
            output[index] = (1.0 - local) * control[segment] + local * control[segment + 1]
        return output
    raise ValueError(f"Unsupported trajectory path type: {path_type}")


def pack_rirs(rirs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack variable-length four-channel RIRs into the release NPZ layout."""

    if not rirs:
        raise ValueError("Cannot pack an empty RIR sequence")
    lengths = np.asarray([np.asarray(rir).shape[0] for rir in rirs], dtype=np.int32)
    packed = np.zeros((len(rirs), int(lengths.max()), 4), dtype=np.float32)
    for index, rir in enumerate(rirs):
        item = np.asarray(rir, dtype=np.float32)
        if item.ndim != 2 or item.shape[1] != 4:
            raise ValueError(f"RIR {index} must have shape [samples, 4], got {item.shape}")
        if not np.isfinite(item).all():
            raise ValueError(f"RIR {index} contains NaN or infinity")
        packed[index, : item.shape[0]] = item
    return packed, lengths


class FOARenderer:
    """Render raw SoundSpaces FOA RIRs for released trajectory recipes.

    Habitat-Sim and the Matterport3D assets are optional and must be installed
    separately. Re-rendering is not expected to be bit-exact across simulator
    builds; use the released exact test RIRs for benchmark reproduction.
    """

    def __init__(
        self,
        *,
        mp3d_root: str | Path,
        scene_dataset_config: str | Path,
        materials_json: str | Path,
        sample_rate: int = 32_000,
        indirect: bool = True,
    ) -> None:
        self.mp3d_root = Path(mp3d_root)
        self.scene_dataset_config = Path(scene_dataset_config)
        self.materials_json = Path(materials_json)
        verify_file_sha256(self.scene_dataset_config, SCENE_CONFIG_SHA256)
        verify_file_sha256(self.materials_json, MATERIALS_CONFIG_SHA256)
        self.sample_rate = int(sample_rate)
        self.indirect = bool(indirect)
        self._scene_id: str | None = None
        self._sensor_offset: tuple[float, float, float] | None = None
        self._sim = None
        self._sensor = None

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
        self._scene_id = None
        self._sensor_offset = None
        self._sim = None
        self._sensor = None

    def __enter__(self) -> "FOARenderer":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def _ensure_scene(self, scene_id: str, sensor_offset: list[float]) -> None:
        try:
            import quaternion  # noqa: F401
            import habitat_sim
            import habitat_sim.sensor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RIR rendering requires a SoundSpaces-compatible Habitat-Sim installation"
            ) from exc
        offset = tuple(float(value) for value in sensor_offset)
        if self._scene_id == scene_id and self._sensor_offset == offset:
            return
        self.close()
        scene = self.mp3d_root / scene_id / f"{scene_id}.glb"
        if not scene.is_file():
            raise FileNotFoundError(scene)
        backend = habitat_sim.SimulatorConfiguration()
        backend.scene_id = str(scene)
        backend.scene_dataset_config_file = str(self.scene_dataset_config)
        backend.load_semantic_mesh = True
        backend.enable_physics = False
        simulator = habitat_sim.Simulator(
            habitat_sim.Configuration(backend, [habitat_sim.AgentConfiguration()])
        )
        layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
        layout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
        layout.channelCount = 4
        specification = habitat_sim.AudioSensorSpec()
        specification.uuid = "audio_sensor"
        specification.enableMaterials = True
        specification.position = list(offset)
        specification.channelLayout = layout
        specification.acousticsConfig.sampleRate = self.sample_rate
        specification.acousticsConfig.indirect = self.indirect
        simulator.add_sensor(specification)
        sensor = simulator.get_agent(0)._sensors["audio_sensor"]
        sensor.setAudioMaterialsJSON(str(self.materials_json))
        self._scene_id = scene_id
        self._sensor_offset = offset
        self._sim = simulator
        self._sensor = sensor

    def render_ir(
        self,
        *,
        scene_id: str,
        agent_position: list[float],
        sensor_offset: list[float],
        source_position: np.ndarray,
    ) -> np.ndarray:
        self._ensure_scene(scene_id, sensor_offset)
        assert self._sim is not None and self._sensor is not None
        self._sensor.setAudioSourceTransform(np.asarray(source_position, dtype=np.float32))
        agent = self._sim.get_agent(0)
        state = agent.get_state()
        state.position = np.asarray(agent_position, dtype=np.float32)
        state.sensor_states = {}
        agent.set_state(state, True)
        observation = np.asarray(
            self._sim.get_sensor_observations()["audio_sensor"], dtype=np.float32
        )
        if observation.ndim != 2:
            raise RuntimeError(f"Unexpected audio observation shape: {observation.shape}")
        if observation.shape[0] == 4:
            observation = observation.T
        if observation.shape[1] != 4:
            raise RuntimeError(f"Expected four FOA channels, got {observation.shape}")
        return observation.astype(np.float32, copy=False)
