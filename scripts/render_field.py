"""Interactive 3D reconstruction of a soccer frame.

Pipeline: image -> keypoints -> homography -> player positions on the bird's-eye
field (player_homography.py) -> humanoids spawned on a full-size 3D pitch.

Usage (from anywhere):
    python scripts/render_field.py [image_path]
    python scripts/render_field.py --demo        # random players, no models loaded

Controls:
    Right mouse drag  orbit            Middle mouse drag  pan
    Mouse wheel       zoom             W/A/S/D  move      Q/E  down/up
    Shift             move faster      1-4  camera presets    R  reset    Esc  quit
"""
import math
import os
import random
import sys

import numpy as np

# torch must be imported before ursina/panda3d: loading Panda3D's DLLs first makes
# torch's c10.dll fail to initialise (WinError 1114).
try:
    import torch  # noqa: F401
except (ImportError, OSError):
    pass  # only needed for the detection pipeline, not --demo

from ursina import (
    AmbientLight, Entity, DirectionalLight, Mesh, Sky, Text, Ursina, Vec3,
    application, camera, clamp, color, held_keys, lerp, mouse, time, window,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "dataset", "images", "2e57b9_5_7.png")

# FIFA / IFAB dimensions in metres. x runs along the pitch, z across it.
L, W = 105.0, 68.0
LINE_W = 0.12
PENALTY_DEPTH, PENALTY_WIDTH = 16.5, 40.32
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5, 18.32
PENALTY_SPOT = 11.0
CIRCLE_R = 9.15
CORNER_R = 1.0
GOAL_WIDTH, GOAL_HEIGHT, GOAL_DEPTH = 7.32, 2.44, 2.0

GRASS_LIGHT = color.rgb32(52, 140, 52)
GRASS_DARK = color.rgb32(44, 124, 44)
TEAM_COLORS = {0: color.rgb32(30, 70, 220), 1: color.rgb32(220, 40, 40)}  # matches player_homography overlay


# ---------------------------------------------------------------------------
# Player positions from player_homography.py
# ---------------------------------------------------------------------------

def image_to_pitch(points):
    """Map bird's-eye image pixels (the homography's output space) to metres."""
    import player_homography as ph

    lo, hi = ph.field_keypoints.min(axis=0), ph.field_keypoints.max(axis=0)
    centre = (lo + hi) / 2
    scale = np.array([L, W]) / (hi - lo)
    x = (points[:, 0] - centre[0]) * scale[0]
    z = -(points[:, 1] - centre[1]) * scale[1]  # image y points down, scene z points up-field
    return np.stack([x, z], axis=1)


def compute_player_positions(image_path):
    """Run the player_homography pipeline; return (N, 2) pitch coords in metres and (N,) team ids."""
    os.chdir(PROJECT_ROOT)  # SAM() looks for sam_b.pt relative to the cwd
    import cv2
    import player_homography as ph

    # Same steps as ph.detect_keypoints, minus its result.show() popup.
    result = ph.keypoint_detector(image_path)[0].cpu()
    if result.keypoints is None or len(result.keypoints) == 0:
        raise RuntimeError("No pitch detected in the image")
    valid = result.keypoints.conf.numpy()[0] >= 0.75
    if valid.sum() < 4:
        raise RuntimeError(f"Only {int(valid.sum())} confident keypoints; a homography needs at least 4")

    h, _ = ph.get_homography(result.keypoints.xy.numpy()[0][valid], valid)
    if h is None:
        raise RuntimeError("Homography estimation failed")

    players, _, _ = ph.detect_players(image_path)
    if len(players) == 0:
        raise RuntimeError("No players detected in the image")

    if len(players) >= 2:
        teams = ph.detect_teams(image_path, ph.segment_players(image_path, players))
    else:
        teams = np.zeros(len(players), dtype=int)

    feet = np.array([ph.get_player_coords(p.tolist()) for p in players], dtype=np.float32).reshape(-1, 1, 2)
    on_field = cv2.perspectiveTransform(feet, h).reshape(-1, 2)
    return image_to_pitch(on_field), np.asarray(teams)


def demo_positions(n=22):
    rng = np.random.default_rng(0)
    pos = np.stack([rng.uniform(-L / 2 + 3, L / 2 - 3, n), rng.uniform(-W / 2 + 3, W / 2 - 3, n)], axis=1)
    return pos, np.arange(n) % 2


# ---------------------------------------------------------------------------
# Pitch
# ---------------------------------------------------------------------------

def _rect(x0, z0, x1, z1):
    """Line segment as a LINE_W-wide rectangle, extended half a width at both ends so corners close."""
    d = np.array([x1 - x0, z1 - z0], dtype=float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) * LINE_W / 2
    a = np.array([x0, z0]) - d * LINE_W / 2
    b = np.array([x1, z1]) + d * LINE_W / 2
    return [a + n, a - n, b - n, b + n]


def _arc(cx, cz, r, a0, a1, steps=48):
    """Arc ribbon from angle a0 to a1 (degrees); returns quads."""
    angles = np.radians(np.linspace(a0, a1, steps + 1))
    inner, outer = r - LINE_W / 2, r + LINE_W / 2
    pts = [(np.array([cx + np.cos(t) * inner, cz + np.sin(t) * inner]),
            np.array([cx + np.cos(t) * outer, cz + np.sin(t) * outer])) for t in angles]
    return [[i0, o0, o1, i1] for (i0, o0), (i1, o1) in zip(pts[:-1], pts[1:])]


def _quads_to_mesh(quads, y):
    verts, tris = [], []
    for q in quads:
        base = len(verts)
        verts += [Vec3(p[0], y, p[1]) for p in q]
        tris.append((base, base + 1, base + 2, base + 3))
    return Mesh(vertices=verts, triangles=tris, mode='triangle')


def pitch_markings():
    quads = []
    hl, hw = L / 2, W / 2

    # Touchlines, goal lines, halfway line
    quads += [_rect(-hl, -hw, hl, -hw), _rect(-hl, hw, hl, hw),
              _rect(-hl, -hw, -hl, hw), _rect(hl, -hw, hl, hw),
              _rect(0, -hw, 0, hw)]

    # Centre circle and centre mark
    quads += _arc(0, 0, CIRCLE_R, 0, 360, steps=96)
    quads += _arc(0, 0, 0.1, 0, 360, steps=16) + _arc(0, 0, 0.2, 0, 360, steps=16)

    for s in (-1, 1):
        gx = s * hl
        # Penalty area and goal area
        for depth, width in ((PENALTY_DEPTH, PENALTY_WIDTH), (GOAL_AREA_DEPTH, GOAL_AREA_WIDTH)):
            fx, half = gx - s * depth, width / 2
            quads += [_rect(fx, -half, fx, half), _rect(gx, -half, fx, -half), _rect(gx, half, fx, half)]

        # Penalty mark and the part of the penalty arc outside the box
        px = gx - s * PENALTY_SPOT
        quads += _arc(px, 0, 0.1, 0, 360, steps=16) + _arc(px, 0, 0.2, 0, 360, steps=16)
        theta = math.degrees(math.acos((PENALTY_DEPTH - PENALTY_SPOT) / CIRCLE_R))
        centre = 180 if s > 0 else 0  # arc bulges towards the middle of the pitch
        quads += _arc(px, 0, CIRCLE_R, centre - theta, centre + theta, steps=32)

        # Corner arcs
        for sz in (-1, 1):
            start = {(1, 1): 180, (1, -1): 90, (-1, 1): 270, (-1, -1): 0}[(s, sz)]
            quads += _arc(gx, sz * hw, CORNER_R, start, start + 90, steps=12)

    return Entity(model=_quads_to_mesh(quads, 0.03), color=color.white,
                  unlit=True, double_sided=True)


def build_goal(side):
    gx = side * L / 2
    hw = GOAL_WIDTH / 2
    back = gx + side * GOAL_DEPTH
    post = color.white
    for z in (-hw, hw):
        Entity(model='cube', position=(gx, GOAL_HEIGHT / 2, z), scale=(0.12, GOAL_HEIGHT, 0.12), color=post)
        Entity(model='cube', position=((gx + back) / 2, 0.01 + 0.0, z), scale=(GOAL_DEPTH, 0.02, 0.05), color=post)
    Entity(model='cube', position=(gx, GOAL_HEIGHT, 0), scale=(0.12, 0.12, GOAL_WIDTH + 0.12), color=post)

    verts = [
        Vec3(back, 0, -hw), Vec3(back, 0, hw), Vec3(back, GOAL_HEIGHT, hw), Vec3(back, GOAL_HEIGHT, -hw),      # back
        Vec3(gx, 0, -hw), Vec3(back, 0, -hw), Vec3(back, GOAL_HEIGHT, -hw), Vec3(gx, GOAL_HEIGHT, -hw),        # side
        Vec3(gx, 0, hw), Vec3(back, 0, hw), Vec3(back, GOAL_HEIGHT, hw), Vec3(gx, GOAL_HEIGHT, hw),            # side
        Vec3(gx, GOAL_HEIGHT, -hw), Vec3(back, GOAL_HEIGHT, -hw), Vec3(back, GOAL_HEIGHT, hw), Vec3(gx, GOAL_HEIGHT, hw),  # roof
    ]
    tris = [(i, i + 1, i + 2, i + 3) for i in range(0, len(verts), 4)]
    Entity(model=Mesh(vertices=verts, triangles=tris, mode='triangle'), color=color.rgba32(255, 255, 255, 60),
           double_sided=True, unlit=True)


def build_pitch():
    # Mowing stripes across the playing surface, plus a darker run-off area around it
    Entity(model='plane', position=(0, -0.02, 0), scale=(L + 24, 1, W + 24), color=color.rgb32(34, 100, 34),
           unlit=True)
    n_stripes = 21
    stripe = L / n_stripes
    for i in range(n_stripes):
        Entity(model='plane', position=(-L / 2 + stripe * (i + 0.5), 0, 0), scale=(stripe, 1, W),
               color=GRASS_LIGHT if i % 2 == 0 else GRASS_DARK, unlit=True)

    pitch_markings()
    build_goal(-1)
    build_goal(1)

    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * L / 2, sz * W / 2
            Entity(model='cube', position=(x, 0.75, z), scale=(0.05, 1.5, 0.05), color=color.white)
            Entity(model='cube', position=(x - sx * 0.15, 1.35, z), scale=(0.3, 0.25, 0.02), color=color.yellow)


# ---------------------------------------------------------------------------
# Humanoids
# ---------------------------------------------------------------------------

def create_humanoid(x, z, kit, facing=0, skin=None, height=1.0):
    """Blocky ~1.8 m humanoid built from primitives, feet on y=0, facing +z before `facing` degrees of yaw."""
    skin = skin or color.rgb32(224, 180, 146)
    root = Entity(position=(x, 0, z), rotation_y=facing, scale=height)

    def part(model, pos, scale, col):
        return Entity(parent=root, model=model, position=pos, scale=scale, color=col)

    for sx in (-1, 1):
        part('cube', (sx * 0.1, 0.04, 0.03), (0.12, 0.08, 0.28), color.black)          # boots
        part('cube', (sx * 0.1, 0.30, 0), (0.11, 0.44, 0.11), kit)                      # socks
        part('cube', (sx * 0.1, 0.68, 0), (0.17, 0.32, 0.18), color.white)              # shorts
        part('cube', (sx * 0.29, 1.35, 0), (0.1, 0.18, 0.12), kit)                      # sleeves
        part('cube', (sx * 0.29, 1.08, 0), (0.08, 0.36, 0.09), skin)                    # forearms
    part('cube', (0, 1.17, 0), (0.46, 0.56, 0.24), kit)                                  # torso
    part('sphere', (0, 1.65, 0), (0.24, 0.26, 0.24), skin)                               # head
    part('sphere', (0, 1.72, -0.02), (0.26, 0.17, 0.26), color.rgb32(50, 35, 25))          # hair
    part('cube', (0, 1.63, 0.12), (0.04, 0.05, 0.04), skin)                              # nose, shows facing
    return root


def spawn_players(positions, teams):
    skins = [color.rgb32(224, 180, 146), color.rgb32(198, 134, 96), color.rgb32(141, 85, 54), color.rgb32(240, 200, 170)]
    rng = random.Random(1)
    players = []
    for (x, z), team in zip(positions, teams):
        # Teams attack in opposite directions: team 0 faces +x, team 1 faces -x
        facing = 90 if team == 0 else -90
        players.append(create_humanoid(float(x), float(z), TEAM_COLORS.get(int(team), color.gray),
                                       facing=facing, skin=rng.choice(skins), height=rng.uniform(0.95, 1.06)))
    return players


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class FlyOrbitCamera(Entity):
    """Orbit/pan/zoom camera that can also fly around with WASD + Q/E.

    The rig sits at the focus point; the camera hangs `distance` metres behind it.
    """
    PRESETS = {
        '1': ((0, 0, -4), 0, 38, 95),       # broadcast, from the main stand
        '2': ((0, 0, 0), 0, 89, 120),       # top-down
        '3': ((-L / 2, 1, 0), 90, 18, 35),  # behind the left goal
        '4': ((0, 0, 0), 180, 25, 80),      # opposite touchline
    }

    def __init__(self):
        super().__init__(name='fly_orbit_camera')
        camera.parent = self
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.clip_plane_near = 0.3
        camera.clip_plane_far = 1500
        camera.fov = 60
        self.rotation_speed = 200
        self.distance = self.target_distance = 95
        self.apply_preset('1')

    def apply_preset(self, key):
        pos, yaw, pitch, dist = self.PRESETS[key]
        self.position = Vec3(*pos)
        self.rotation_y, self.rotation_x = yaw, pitch
        self.distance = self.target_distance = dist

    def input(self, key):
        if key == 'scroll up':
            self.target_distance *= 0.88
        elif key == 'scroll down':
            self.target_distance /= 0.88
        elif key in self.PRESETS:
            self.apply_preset(key)
        elif key == 'r':
            self.apply_preset('1')
        elif key == 'escape':
            application.quit()
        self.target_distance = clamp(self.target_distance, 2, 300)

    def update(self):
        if held_keys['right mouse']:
            self.rotation_y += mouse.velocity[0] * self.rotation_speed
            self.rotation_x = clamp(self.rotation_x - mouse.velocity[1] * self.rotation_speed, 2, 89)

        yaw = math.radians(self.rotation_y)
        forward = Vec3(math.sin(yaw), 0, math.cos(yaw))
        right = Vec3(math.cos(yaw), 0, -math.sin(yaw))

        if mouse.middle:
            k = self.distance * 0.9
            self.position -= right * mouse.velocity[0] * k
            self.position -= forward * mouse.velocity[1] * k

        speed = max(8, self.distance * 0.5) * (3 if held_keys['shift'] else 1)
        move = (forward * (held_keys['w'] - held_keys['s'])
                + right * (held_keys['d'] - held_keys['a'])
                + Vec3(0, held_keys['e'] - held_keys['q'], 0))
        self.position += move * speed * time.dt
        self.position = Vec3(clamp(self.x, -L, L), clamp(self.y, 0, 80), clamp(self.z, -W, W))

        self.distance = lerp(self.distance, self.target_distance, min(1, time.dt * 10))
        camera.z = -self.distance


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_scene(positions, teams):
    Sky()
    build_pitch()
    spawn_players(positions, teams)
    DirectionalLight(shadows=False).look_at(Vec3(0.4, -1, 0.3))
    AmbientLight(color=color.rgba32(140, 140, 140, 255))
    cam = FlyOrbitCamera()
    Text(text=__doc__.split('Controls:')[1].strip(), position=window.top_left, origin=(-0.5, 0.5), scale=0.8,
         background=True)
    return cam


def main(argv):
    demo = '--demo' in argv
    paths = [a for a in argv if not a.startswith('--')]
    image_path = paths[0] if paths else DEFAULT_IMAGE

    # Compute positions before opening the window so model errors show up in the console
    positions, teams = demo_positions() if demo else compute_player_positions(image_path)
    print(f"Spawning {len(positions)} players")

    app = Ursina(title="Soccer 3D Reconstruction", borderless=False)
    window.color = color.rgb32(30, 30, 30)
    build_scene(positions, teams)
    app.run()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1:])
