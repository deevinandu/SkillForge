"""Serializable specs: what a task is, what the scene holds, how it is randomized.

These are the frozen Phase 0 contracts. Change them only by a PR that touches
`contracts/` alone and updates every track that reads them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]
# Named colours an object can have. The env maps each to an RGBA; perception and success use the name.
ColorName = Literal["red", "blue", "green", "yellow", "orange", "purple", "white", "grey", "black"]
Quat = tuple[float, float, float, float]  # w, x, y, z (MuJoCo order)


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Pose(Spec):
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)


class Range(Spec):
    """Closed interval [lo, hi]. lo == hi means no randomization."""

    lo: float
    hi: float

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.lo > self.hi:
            raise ValueError(f"Range lo={self.lo} > hi={self.hi}")
        return self

    @classmethod
    def fixed(cls, v: float) -> Range:
        return cls(lo=v, hi=v)


class RobotSpec(Spec):
    """A robot arm + gripper. `source` names where its model comes from."""

    name: str
    source: str = Field(
        description="'menagerie:<dir>' for mujoco_menagerie, 'mjcf:<path>' or 'urdf:<path>' for local files, 'stub' for the toy robot"
    )
    base_pose: Pose = Pose()
    ee_site: str = "tcp"
    gripper: Literal["parallel", "suction", "none"] = "parallel"
    max_gripper_width: float = Field(0.08, gt=0, description="metres")
    # "foam": the fingertip pads are covered in foam: high friction, soft (compliant) contact and
    # torsional grip. "rigid": the model's bare pads. Only the sim reads it (the real gripper has
    # what it has).
    fingertips: Literal["rigid", "foam"] = "rigid"


class MugHandle(Spec):
    """The handle of a `shape="mug"` object: a closed D-shaped loop on the mug's +x side (object
    frame), in the mug's x-z plane, so the hole through it runs along the mug's y axis. Made of
    round bars of radius `radius`; the loop leaves the wall at heights `top` and `bottom` (bar
    centrelines, metres from the mug's centre) and reaches `reach` beyond the outer wall (outer
    edge). The clear opening is then (reach - 2 radius) wide and (top - bottom - 2 radius) tall."""

    top: float = 0.025
    bottom: float = -0.02
    reach: float = Field(0.03, gt=0)
    radius: float = Field(0.004, gt=0)

    @model_validator(mode="after")
    def _open(self) -> MugHandle:
        if self.reach <= 2 * self.radius or self.top - self.bottom <= 2 * self.radius:
            raise ValueError("mug handle has no opening: reach and top - bottom must exceed 2 * radius")
        return self


class ObjectSpec(Spec):
    name: str
    shape: Literal["box", "cylinder", "sphere", "mesh", "mug"]
    size: Vec3 = Field(
        description="box: half-extents; cylinder: (radius, half-height, 0); sphere: (radius, 0, 0); mesh: scale; "
        "mug: (outer radius, half-height, wall thickness), open at +z, with a `handle` on its +x side"
    )
    mesh: str | None = None
    handle: MugHandle | None = Field(None, description="mug only; None on a mug means MugHandle()")
    mass: float = Field(0.1, gt=0, description="kg")
    friction: float = Field(1.0, gt=0)
    pose: Pose = Pose()
    fixed: bool = False
    color: ColorName | None = Field(None, description="None: the env picks one by the object's index")

    @model_validator(mode="after")
    def _mesh_needs_path(self) -> ObjectSpec:
        if (self.shape == "mesh") != (self.mesh is not None):
            raise ValueError(f"object '{self.name}': mesh path is required for shape 'mesh' and only for it")
        if self.handle is not None and self.shape != "mug":
            raise ValueError(f"object '{self.name}': only a mug has a handle")
        if self.shape == "mug":
            r, h, w = self.size
            hd = self.handle or MugHandle()
            if not (0 < w < r and h > 0):
                raise ValueError(f"mug '{self.name}': size must be (outer radius, half-height, wall) with 0 < wall < radius")
            if not (-h < hd.bottom < hd.top < h):
                raise ValueError(f"mug '{self.name}': handle must leave the wall between the mug's bottom and rim")
        return self

    def mug_handle(self) -> MugHandle:
        """This mug's handle (the default one if the spec leaves it out)."""
        if self.shape != "mug":
            raise ValueError(f"object '{self.name}' is not a mug")
        return self.handle or MugHandle()


class Region(Spec):
    """Axis-aligned box in world frame, used for goals and placement."""

    name: str
    center: Vec3
    half_extents: Vec3
    # Drawn in the sim (a translucent green box) when True. Set False for regions that only exist
    # in the task logic (spots, lanes): nothing marks them on a real table, and a camera should
    # not see them either.
    visible: bool = True


class Knockdown(Spec):
    """Someone knocks objects over mid-episode (a disturbance to recover from).

    At the control step drawn from `at_step`, `count` movable objects (drawn from the seed, never
    the one in the gripper) are knocked over, and so is everything resting on them (in a stack,
    knocking a lower object brings down what stands on it). Each knocked object is laid on its
    side on the table, at rest, at a free spot within `radius` of where it stood. Both ranges are
    rounded to integers."""

    at_step: Range
    count: Range = Range.fixed(1.0)
    radius: float = Field(0.08, gt=0, description="metres")

    @model_validator(mode="after")
    def _non_negative(self) -> Knockdown:
        if self.at_step.lo < 0 or self.count.lo < 0:
            raise ValueError("knockdown at_step and count must be >= 0")
        return self


class RandomizationSpec(Spec):
    """Domain randomization. Every field is sampled per episode from its Range."""

    # Jitter fields are magnitudes: per episode m ~ U(lo, hi), then the offset ~ U(-m, m).
    object_xy_jitter: Range = Range.fixed(0.0)  # metres, per axis (a square)
    object_yaw_jitter: Range = Range.fixed(0.0)  # radians, about world z
    friction_scale: Range = Range.fixed(1.0)
    mass_scale: Range = Range.fixed(1.0)
    actuator_gain_scale: Range = Range.fixed(1.0)
    obs_noise_std: Range = Range.fixed(0.0)  # metres, added to object positions
    action_latency_steps: Range = Range.fixed(0.0)  # rounded to int
    # Permute the colours among the movable objects that have one, per episode (from the env seed):
    # which object is which colour is then only knowable by looking.
    shuffle_colors: bool = False
    knockdown: Knockdown | None = None  # None: nobody knocks anything over


# How a box rests on the table: "upright" on its own z axis (its spawn convention), "flat" on its
# largest face (thinnest axis vertical, a domino lying face down), "side" on the remaining face
# (middle axis vertical, a domino lying on its long edge).
RestFace = Literal["upright", "flat", "side"]


class Scatter(Spec):
    """Random initial poses, drawn from the seed at reset: each of `objects` (empty = every movable
    one) rests on one of `faces` (uniform), at a yaw ~ U(-max_yaw, max_yaw), at a uniform xy in one of `regions`
    (uniform), at least `min_dist` (centre to centre) from every other object. Faces other than
    "upright" need box objects. Overrides the spec pose and the xy/yaw jitter of those objects."""

    objects: tuple[str, ...] = ()
    regions: tuple[str, ...] = Field(min_length=1)
    faces: tuple[RestFace, ...] = Field(("upright",), min_length=1)
    min_dist: float = Field(0.10, ge=0.0)
    max_yaw: float = Field(3.1416, gt=0.0, le=3.1416, description="radians")


class Disturbance(Spec):
    """A perturbation the env applies to objects DURING an episode (someone bumps the table, a
    domino gets nudged after it was placed). Everything about it is drawn from the env seed at
    reset, so the same seed disturbs the same object at the same step by the same amount.

    When it fires: `at_step` steps (drawn, rounded) after reset, or after the first event whose
    name starts with `after_event` (e.g. "release:" = right after the robot lets go of something).
    It is cancelled if an event starting with `before_event` has happened by then (e.g.
    "toppled:" = once the chain is falling, stop disturbing it). It fires at all with `prob`.

    What it does to each of `count` objects, chosen from `objects` (empty = every movable one;
    `target="event_object"` = the object named in the triggering event, after its first colon):
    - kind "pose": teleport the object, at rest, by an xy offset of magnitude ~ U(xy) in a uniform
      direction, and a yaw about world z of magnitude ~ U(yaw) with a random sign;
    - kind "knock": give it an angular velocity of magnitude ~ U(ang_vel) about a horizontal axis,
      so its top moves toward `knock_dir` (world xy) or, if None, a uniform random direction;
    - kind "relocate": someone picks it up and puts it down, at rest, on one of `faces` at a
      yaw ~ U(-max_yaw, max_yaw): in one of `regions` (a free spot at least 0.09 m from everything), or, with
      no regions, where it is moved by an xy offset of magnitude ~ U(xy).
    The env never disturbs an object the gripper is touching; it waits for it to be let go."""

    kind: Literal["pose", "knock", "relocate"] = "pose"
    objects: tuple[str, ...] = ()
    count: int = Field(1, ge=1)
    prob: float = Field(1.0, ge=0.0, le=1.0)
    after_event: str | None = None
    at_step: Range = Range.fixed(0.0)  # control steps after reset / after `after_event`, rounded
    before_event: str | None = None
    target: Literal["random", "event_object"] = "random"
    xy: Range = Range.fixed(0.0)  # metres (pose)
    yaw: Range = Range.fixed(0.0)  # radians (pose)
    ang_vel: Range = Range.fixed(0.0)  # rad/s (knock)
    knock_dir: tuple[float, float] | None = None
    regions: tuple[str, ...] = ()  # relocate
    faces: tuple[RestFace, ...] = ("flat",)  # relocate
    max_yaw: float = Field(3.1416, gt=0.0, le=3.1416)  # relocate: radians

    @model_validator(mode="after")
    def _consistent(self) -> Disturbance:
        if self.at_step.lo < 0:
            raise ValueError("disturbance at_step must be >= 0")
        if min(self.xy.lo, self.yaw.lo, self.ang_vel.lo) < 0:
            raise ValueError("disturbance magnitudes (xy, yaw, ang_vel) must be >= 0")
        if self.kind == "pose" and self.xy.hi == 0 and self.yaw.hi == 0:
            raise ValueError("a 'pose' disturbance needs a non-zero xy or yaw range")
        if self.kind == "knock" and self.ang_vel.hi == 0:
            raise ValueError("a 'knock' disturbance needs a non-zero ang_vel range")
        if not self.faces:
            raise ValueError("disturbance faces must not be empty")
        if self.target == "event_object" and not self.after_event:
            raise ValueError("target 'event_object' needs after_event")
        if self.knock_dir is not None and self.knock_dir[0] ** 2 + self.knock_dir[1] ** 2 < 1e-12:
            raise ValueError("knock_dir must be a non-zero xy vector")
        return self


class TableSpec(Spec):
    """The table the robot is bolted to. `width` is across the robot (world y, centred on the
    robot base), `depth` along it (world x, from `back` behind the base). With `shelves`, the
    table stands between two shelf units: solid side panels `shelf_height` tall just outside the
    width, which the arm can collide with. Every region of the scene must then lie inside the
    width, and skills keep the hand inside it."""

    width: float = Field(1.2, gt=0.2, description="metres, world y")
    depth: float = Field(1.2, gt=0.2, description="metres, world x")
    back: float = Field(0.25, ge=0.0, description="metres of table behind the robot base")
    shelves: bool = False
    shelf_height: float = Field(0.6, gt=0.0, description="metres above the table top")
    shelf_thickness: float = Field(0.02, gt=0.0)


class Fixture(Spec):
    """A static box in the cell the robot can collide with (shelf boards, frame posts, a camera
    mount): world-frame centre and half extents, axis-aligned. `ledge`: a shelf board whose edge
    skills may use (e.g. to lay a domino on with one end overhanging, so it can be pinched there)."""

    name: str
    center: Vec3
    half_extents: Vec3
    ledge: bool = False
    rgba: tuple[float, float, float, float] = (0.92, 0.92, 0.9, 1.0)


class SceneSpec(Spec):
    robot: RobotSpec
    table: TableSpec = TableSpec()
    fixtures: tuple[Fixture, ...] = ()
    objects: tuple[ObjectSpec, ...] = ()
    regions: tuple[Region, ...] = ()
    randomization: RandomizationSpec = RandomizationSpec()
    # Cameras the env renders into `Observation.images` every step (e.g. "wrist", "front").
    obs_cameras: tuple[str, ...] = ()
    # Mid-episode perturbations (see Disturbance). Empty = none, and every existing task is unchanged.
    disturbances: tuple[Disturbance, ...] = ()
    # Random initial poses (see Scatter). None = objects start where the spec puts them.
    scatter: Scatter | None = None

    @model_validator(mode="after")
    def _unique_names(self) -> SceneSpec:
        names = [o.name for o in self.objects] + [r.name for r in self.regions] + [f.name for f in self.fixtures]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate object/region names: {sorted(dupes)}")
        movable = {o.name for o in self.objects if not o.fixed}
        regions = {r.name for r in self.regions}
        boxes = {o.name for o in self.objects if o.shape == "box"}
        for i, d in enumerate(self.disturbances):
            bad = [n for n in d.objects if n not in movable]
            if bad:
                raise ValueError(f"disturbance {i} names objects that are missing or fixed: {bad}")
            if d.kind == "relocate":
                if [r for r in d.regions if r not in regions]:
                    raise ValueError(f"disturbance {i} names unknown regions: {list(d.regions)}")
                if any(f != "upright" for f in d.faces) and [n for n in (d.objects or movable) if n not in boxes]:
                    raise ValueError(f"disturbance {i}: faces other than upright need box objects")
        if self.table.shelves:
            y0, hw = self.robot.base_pose.pos[1], self.table.width / 2
            out = [r.name for r in self.regions if abs(r.center[1] - y0) + r.half_extents[1] > hw]
            if out:
                raise ValueError(f"regions reach past the shelves (table width {self.table.width} m): {out}")
        for r in self.regions:  # nothing is to be put where a low fixture would be over it
            for f in self.fixtures:
                if f.center[2] - f.half_extents[2] > 0.15:
                    continue
                if all(abs(r.center[i] - f.center[i]) < r.half_extents[i] + f.half_extents[i] for i in (0, 1)):
                    raise ValueError(f"region '{r.name}' lies under fixture '{f.name}'")
        if self.scatter is not None:
            sc = self.scatter
            bad = [n for n in sc.objects if n not in movable]
            if bad:
                raise ValueError(f"scatter names objects that are missing or fixed: {bad}")
            if [r for r in sc.regions if r not in regions]:
                raise ValueError(f"scatter names unknown regions: {list(sc.regions)}")
            if any(f != "upright" for f in sc.faces) and [n for n in (sc.objects or movable) if n not in boxes]:
                raise ValueError("scatter faces other than upright need box objects")
        return self

    def object(self, name: str) -> ObjectSpec:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(f"no object '{name}'; have {[o.name for o in self.objects]}")

    def region(self, name: str) -> Region:
        for r in self.regions:
            if r.name == name:
                return r
        raise KeyError(f"no region '{name}'; have {[r.name for r in self.regions]}")


class ObjectInRegion(Spec):
    kind: Literal["object_in_region"] = "object_in_region"
    object: str
    region: str


class ObjectLifted(Spec):
    kind: Literal["object_lifted"] = "object_lifted"
    object: str
    min_height: float = Field(0.05, gt=0, description="metres above its start height")


class ObjectOnObject(Spec):
    """`object` rests on top of `support`, judged on TRUE positions by the env:
    - horizontal: the object's centre lies within `support`'s footprint, shrunk by `xy_margin`
      (box: its x/y half-sizes in the support's frame; cylinder/sphere: its radius);
    - vertical: centre-to-centre height is the two resting half-heights summed, within `z_tol`.
    Upright stacks only: orientation is not part of the condition."""

    kind: Literal["object_on_object"] = "object_on_object"
    object: str
    support: str
    xy_margin: float = Field(0.0, ge=0, description="metres the object's centre must sit inside the support's edge")
    z_tol: float = Field(0.01, gt=0, description="metres of slack on the stacked height")

    @model_validator(mode="after")
    def _distinct(self) -> ObjectOnObject:
        if self.object == self.support:
            raise ValueError("object_on_object needs two different objects")
        return self


class ColorInRegion(Spec):
    """Every movable object whose colour this episode is `color` lies inside `region` (on TRUE state).

    The colour is the one the env gave the object this episode, which differs from the spec when
    `shuffle_colors` is on. A region may span several slots; each object only has to be in one."""

    kind: Literal["color_in_region"] = "color_in_region"
    color: ColorName
    region: str


class ObjectToppled(Spec):
    """`object` has fallen over, judged on TRUE pose by the env: its own z axis (a box's height
    axis; spawn it upright) is tilted at least `min_tilt` from world up. Optionally:
    - `direction`: it fell that way. The horizontal projection of its z axis (where its top went)
      is within `dir_tol` of this world xy direction;
    - `region`: its centre lies inside this region (e.g. the lane the chain stands in);
    - `restood`: it must have fallen (tilt >= `min_tilt`) and then stood upright again (tilt
      <= 0.15 rad) earlier in the episode: for "rebuild the fallen chain and topple it again",
      where the start already looks toppled. The env tracks it on true poses.
    A domino leaning on the next one counts: a chain rarely ends up flat."""

    kind: Literal["object_toppled"] = "object_toppled"
    object: str
    min_tilt: float = Field(0.6, gt=0, lt=3.1416, description="radians from upright")
    direction: tuple[float, float] | None = None
    dir_tol: float = Field(0.8, gt=0, le=3.1416, description="radians")
    region: str | None = None
    restood: bool = False

    @model_validator(mode="after")
    def _direction(self) -> ObjectToppled:
        if self.direction is not None and self.direction[0] ** 2 + self.direction[1] ** 2 < 1e-12:
            raise ValueError("object_toppled direction must be a non-zero xy vector")
        return self


class ObjectHung(Spec):
    """`object` (a mug) hangs by its handle from `support` (a fixed cylinder: a hook or peg), judged
    on TRUE state by the env, and has done so for `hold_s` seconds without a break:
    - threaded: the support's axis segment passes through the inside of the handle loop;
    - free: the object touches nothing but fixed objects (no robot, table, floor or movable
      object), so it is neither still held nor resting on something else;
    - calm: the object's centre moves slower than `max_speed` (a mug swinging gently still hangs;
      one sliding off the hook does not)."""

    kind: Literal["object_hung"] = "object_hung"
    object: str
    support: str
    hold_s: float = Field(1.0, ge=0, description="seconds the condition must hold without a break")
    max_speed: float = Field(0.15, gt=0, description="m/s")

    @model_validator(mode="after")
    def _distinct(self) -> ObjectHung:
        if self.object == self.support:
            raise ValueError("object_hung needs two different objects")
        return self


LeafSuccessSpec = Annotated[
    ObjectInRegion | ObjectLifted | ObjectOnObject | ColorInRegion | ObjectToppled | ObjectHung,
    Field(discriminator="kind"),
]


class AllOf(Spec):
    """Every condition holds at the same step. Leaves only (no nesting), so a task stays a flat
    list of checks: e.g. sorting = several object_in_region, stacking = object_on_object chain."""

    kind: Literal["all_of"] = "all_of"
    conditions: tuple[LeafSuccessSpec, ...] = Field(min_length=1)


SuccessSpec = Annotated[
    ObjectInRegion | ObjectLifted | ObjectOnObject | ColorInRegion | ObjectToppled | ObjectHung | AllOf,
    Field(discriminator="kind"),
]
_AnySuccess = ObjectInRegion | ObjectLifted | ObjectOnObject | ColorInRegion | ObjectToppled | ObjectHung | AllOf


def success_conditions(success: _AnySuccess) -> tuple:
    """The leaf conditions of a success spec, in order (one for a leaf, all of them for AllOf)."""
    return tuple(success.conditions) if isinstance(success, AllOf) else (success,)


def success_objects(success: _AnySuccess) -> tuple[str, ...]:
    """Every object a success spec mentions BY NAME (supports included), first mention first, no
    repeats. `color_in_region` names no object: which objects it covers depends on the episode."""
    names: list[str] = []
    for c in success_conditions(success):
        for n in (getattr(c, "object", None), getattr(c, "support", None)):
            if n is not None and n not in names:
                names.append(n)
    return tuple(names)


class TaskSpec(Spec):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    instruction: str = Field(min_length=1, description="natural-language task, as a person would say it")
    scene: SceneSpec
    success: SuccessSpec
    max_steps: int = Field(500, gt=0)
    control_hz: float = Field(20.0, gt=0)

    @model_validator(mode="after")
    def _success_refs_exist(self) -> TaskSpec:
        try:
            for c in success_conditions(self.success):
                if isinstance(c, ColorInRegion):
                    self.scene.region(c.region)
                    if not any(o.color == c.color and not o.fixed for o in self.scene.objects):
                        raise KeyError(f"no movable object is {c.color}")
                    continue
                self.scene.object(c.object)
                if isinstance(c, ObjectInRegion):
                    self.scene.region(c.region)
                if isinstance(c, ObjectOnObject):
                    self.scene.object(c.support)
                if isinstance(c, ObjectToppled) and c.region is not None:
                    self.scene.region(c.region)
                if isinstance(c, ObjectHung):
                    sup = self.scene.object(c.support)
                    if not (sup.fixed and sup.shape == "cylinder"):
                        raise ValueError(f"object_hung: support '{c.support}' must be a fixed cylinder")
                    if self.scene.object(c.object).shape != "mug":
                        raise ValueError(f"object_hung: '{c.object}' must be a mug")
        except KeyError as e:
            raise ValueError(f"success refers to a missing name: {e.args[0]}") from None
        return self
