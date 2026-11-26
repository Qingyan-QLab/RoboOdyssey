"""Isaac viewport binding, imported without loading the simulator."""
from __future__ import annotations
def _bind_viewport_camera(viewport, camera_path: str) -> None:
    """Bind a Kit viewport to a USD camera using the Isaac Sim 6 API."""
    if viewport is None:
        return
    if str(getattr(viewport, 'camera_path', '')) == camera_path:
        return
    # ``camera_path`` is the public API in Isaac Sim 6.  Older Kit builds
    # exposed ``set_active_camera`` on the same object, so retain that small
    # fallback for local installations.
    try:
        # ViewportManager updates both the viewport API and its render product.
        # This is important when the viewport was created after the camera
        # annotator: assigning ``camera_path`` alone can leave a stale product.
        from isaacsim.core.rendering_manager import ViewportManager
        ViewportManager.set_camera(camera_path, render_product_or_viewport=viewport)
        # Some Kit 6.x builds keep a separate render-product relationship.  Set
        # it explicitly as well, otherwise the visible viewport may continue to
        # render the camera that was selected when the product was created.
        try:
            from pxr import Sdf, Usd
            product = ViewportManager.get_render_product(viewport)
            if product is not None:
                stage = __import__("omni.usd", fromlist=["get_context"]).get_context().get_stage()
                with Usd.EditContext(stage, stage.GetSessionLayer()):
                    product.GetCameraRel().SetTargets([Sdf.Path(camera_path)])
        except Exception:
            pass
    except Exception:
        try:
            viewport.camera_path = camera_path
        except Exception:
            setter = getattr(viewport, "set_active_camera", None)
            if setter is None:
                raise
            setter(camera_path)
    try:
        viewport.updates_enabled = True
    except Exception:
        pass
    # USD pose updates already drive the renderer. Calling the viewport's
    # internal viewport_changed hook rebuilds products and can corrupt Kit's
    # backbuffers when camera annotators are attached.


def _place_viewport_window(window) -> None:
    """Place the overview window in the top-right without assuming 1920px UI."""
    if window is None:
        return
    try:
        import omni.ui as ui
        width = int(ui.Workspace.get_main_window_width())
        window.position_x = max(0, width - 440)
        window.position_y = 42
    except Exception:
        # A fixed fallback is still preferable to leaving the window off-screen.
        try:
            window.position_x = 840
            window.position_y = 42
        except Exception:
            pass
    window.visible = True
    try:
        window.viewport_api.updates_enabled = True
    except Exception:
        pass


def _sync_viewer(viewport, overview_window, front_camera: str, overview_camera: str | None,
                 app) -> None:
    """Rebind and advance viewer products after an environment action.

    Isaac Sim can recreate a viewport render product when a stage is opened or
    a window is moved.  Reasserting both camera relationships after each step
    keeps the displayed cameras tied to the same USD prims used by the agent.
    """
    try:
        _bind_viewport_camera(viewport, front_camera)
        if overview_camera and overview_window is not None:
            overview_viewport = getattr(overview_window, "viewport_api", overview_window)
            _bind_viewport_camera(overview_viewport, overview_camera)
            _place_viewport_window(overview_window)
        app.update()
    except Exception as exc:
        # Rendering must not abort an otherwise valid benchmark run.
        print(f"[viewer] warning: viewport sync failed: {exc}")
