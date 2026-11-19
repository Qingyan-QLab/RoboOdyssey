"""Isaac Sim RGB-D/pose publishing for HoloAgent's ROS perception stack.

The benchmark normally keeps Isaac/ROS optional.  This module is imported and
constructed only when ``--holoagent-sensors`` is supplied.  RGB, depth and
camera-info are published by Isaac Sim's native ROS 2 camera-helper graph;
agent pose is published as ``geometry_msgs/msg/PoseStamped`` from the same
benchmark pose used for collision and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any


@dataclass(frozen=True)
class HoloAgentSensorTopics:
    """ROS topic contract consumed by HoloAgent's online mapper/perception."""

    rgb: str = "/zed/zed_node/rgb/color/rect/image"
    depth: str = "/zed/zed_node/depth/depth_registered"
    camera_info: str = "/zed/zed_node/rgb/color/rect/image/camera_info"
    pose: str = "/zed/zed_node/pose"
    frame_id: str = "zed_left_camera_frame_optical"
    pose_frame_id: str = "map"
    depth_encoding: str = "32FC1"
    depth_unit: str = "m"
    width: int = 640
    height: int = 360
    fx: float = 263.3348388671875
    fy: float = 263.3348388671875
    cx: float = 329.8558349609375
    cy: float = 180.2921600341797
    horizontal_fov_degrees: float = 101.09661114039531

    def as_dict(self) -> dict[str, Any]:
        """Serialize the complete cross-process sensor contract."""
        return {
            "rgb_topic": self.rgb,
            "depth_topic": self.depth,
            "camera_info_topic": self.camera_info,
            "pose_topic": self.pose,
            "frame_id": self.frame_id,
            "pose_frame_id": self.pose_frame_id,
            "depth_encoding": self.depth_encoding,
            "depth_unit": self.depth_unit,
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "horizontal_fov_degrees": float(self.horizontal_fov_degrees),
        }


def pose_message_fields(pose: dict[str, Any], *, frame_id: str = "map",
                        stamp: Any = None) -> dict[str, Any]:
    """Return the ROS PoseStamped fields without importing ROS in CPU tests."""
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        raise ValueError("agent pose requires position[x,y,z]")
    if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
        raise ValueError("agent pose requires orientation[x,y,z,w]")
    values = [float(item) for item in (*position[:3], *orientation)]
    if not all(math.isfinite(item) for item in values):
        raise ValueError("agent pose must contain finite values")
    return {
        "header": {"frame_id": str(frame_id), "stamp": stamp},
        "position": {"x": values[0], "y": values[1], "z": values[2]},
        "orientation": {"x": values[3], "y": values[4],
                        "z": values[5], "w": values[6]},
    }


class IsaacSimHoloAgentSensorBridge:
    """Publish an Isaac Sim camera and benchmark pose on HoloAgent topics.

    Isaac Sim's ``ROS2CameraHelper`` owns the RGB/depth/camera-info publishers,
    avoiding a CPU copy through Python.  A small rclpy publisher is used only
    for PoseStamped because the benchmark avatar is kinematic rather than an
    Isaac articulation with a ready-made odometry node.
    """

    def __init__(self, camera: Any, resolution: tuple[int, int],
                 topics: HoloAgentSensorTopics | None = None,
                 graph_path: str = "/World/HoloAgentSensorBridge") -> None:
        self.camera = camera
        self.resolution = tuple(int(item) for item in resolution)
        self.topics = topics or HoloAgentSensorTopics()
        # OmniGraph paths live in the current USD stage.  Derive a stable,
        # camera-specific path so a second bridge cannot accidentally reuse a
        # graph belonging to another camera while keeping repeated session
        # loads deterministic.
        camera_path = getattr(camera, "prim_path", None) or getattr(camera, "_prim_path", None)
        if graph_path == "/World/HoloAgentSensorBridge" and camera_path:
            digest = hashlib.sha1(str(camera_path).encode("utf-8")).hexdigest()[:10]
            self.graph_path = f"/World/HoloAgentSensorBridge_{digest}"
        else:
            self.graph_path = graph_path
        self._graph = None
        self._node = None
        self._rclpy = None
        self._rclpy_owned = False
        self._setup_camera_graph()
        try:
            self._setup_pose_publisher()
        except Exception:
            # Do not leave a live camera graph behind when ROS is unavailable.
            self.close()
            raise

    def _setup_camera_graph(self) -> None:
        try:
            import carb
            import isaacsim.core.experimental.utils.app as app_utils
        except Exception as exc:  # pragma: no cover - requires Isaac Sim
            raise RuntimeError(
                "HoloAgent ROS sensors require Isaac Sim's isaacsim.ros2.bridge "
                f"extension and OmniGraph imports: {exc}") from exc

        app_utils.enable_extension("isaacsim.ros2.bridge")
        # Isaac Sim exposes its bundled Python ROS 2 modules only after the
        # extension has completed one Kit update.  The benchmark runs inside
        # ``python.sh`` (Python 3.12), so importing the system/Conda rclpy here
        # is incorrect; let Kit load the extension first.
        try:
            import omni.kit.app
            omni.kit.app.get_app().update()
        except Exception:
            # The graph imports below provide the actionable error if this is
            # not an Isaac Sim process (for example, in CPU unit tests).
            pass
        try:
            import omni.graph.core as og
            import usdrt.Sdf
        except Exception as exc:  # pragma: no cover - requires Isaac Sim
            raise RuntimeError(
                "HoloAgent ROS sensors require OmniGraph/usdrt after enabling "
                f"isaacsim.ros2.bridge: {exc}") from exc
        keys = og.Controller.Keys
        width, height = self.resolution
        camera_path = getattr(self.camera, "prim_path", None) or getattr(self.camera, "_prim_path", None)
        if not camera_path:
            raise RuntimeError("Isaac Sim camera does not expose a prim_path")
        try:
            graph, _, _, _ = og.Controller.edit(
                {
                    "graph_path": self.graph_path,
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("PublishRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                        ("PublishDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "PublishRgb.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "PublishCameraInfo.inputs:execIn"),
                        ("CreateRenderProduct.outputs:execOut", "PublishDepth.inputs:execIn"),
                        ("CreateRenderProduct.outputs:renderProductPath", "PublishRgb.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "PublishCameraInfo.inputs:renderProductPath"),
                        ("CreateRenderProduct.outputs:renderProductPath", "PublishDepth.inputs:renderProductPath"),
                    ],
                    keys.SET_VALUES: [
                        ("CreateRenderProduct.inputs:cameraPrim",
                         [usdrt.Sdf.Path(camera_path)]),
                        ("CreateRenderProduct.inputs:width", width),
                        ("CreateRenderProduct.inputs:height", height),
                        ("PublishRgb.inputs:frameId", self.topics.frame_id),
                        ("PublishRgb.inputs:topicName", self.topics.rgb),
                        ("PublishRgb.inputs:type", "rgb"),
                        ("PublishCameraInfo.inputs:frameId", self.topics.frame_id),
                        ("PublishCameraInfo.inputs:topicName", self.topics.camera_info),
                        ("PublishDepth.inputs:frameId", self.topics.frame_id),
                        ("PublishDepth.inputs:topicName", self.topics.depth),
                        ("PublishDepth.inputs:type", "depth"),
                    ],
                },
            )
            self._graph = graph
            og.Controller.evaluate_sync(graph)
        except Exception as exc:  # pragma: no cover - requires Isaac Sim
            carb.log_error(f"HoloAgent ROS camera graph setup failed: {exc}")
            raise RuntimeError(f"could not create HoloAgent ROS camera graph: {exc}") from exc

    def _setup_pose_publisher(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
        except Exception as exc:  # pragma: no cover - requires ROS 2
            raise RuntimeError(
                "HoloAgent ROS sensors require rclpy and geometry_msgs; "
                f"import failed: {exc}") from exc
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
            self._rclpy_owned = True
        self._node = rclpy.create_node("longterm_benchmark_holoagent_sensor_bridge")
        self._pose_type = PoseStamped
        self._pose_pub = self._node.create_publisher(PoseStamped, self.topics.pose, 10)

    def publish_pose(self, pose: dict[str, Any]) -> None:
        """Publish the current benchmark avatar pose and service ROS callbacks."""
        if self._node is None:
            return
        message = self._pose_type()
        now = self._node.get_clock().now().to_msg()
        message.header.stamp = now
        message.header.frame_id = self.topics.pose_frame_id
        position = pose.get("position") or []
        orientation = pose.get("orientation") or []
        if len(position) < 3 or len(orientation) != 4:
            raise ValueError("agent pose must contain position and quaternion orientation")
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.position.z = float(position[2])
        message.pose.orientation.x = float(orientation[0])
        message.pose.orientation.y = float(orientation[1])
        message.pose.orientation.z = float(orientation[2])
        message.pose.orientation.w = float(orientation[3])
        self._pose_pub.publish(message)
        # Keep the optional ROS context responsive without taking ownership of
        # Isaac Sim's main loop or blocking the benchmark step.
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None and self._rclpy_owned and self._rclpy.ok():
            self._rclpy.shutdown()
            self._rclpy = None
