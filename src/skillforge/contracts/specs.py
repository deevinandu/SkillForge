"""Serializable specs: what a task is, what the scene holds, how it is randomized.

These are the frozen Phase 0 contracts. Change them only by a PR that touches
`contracts/` alone and updates every track that reads them.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Vec3 = tuple[float, float, float]
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


class ObjectSpec(Spec):
    name: str
    shape: Literal["box", "cylinder", "sphere", "mesh"]
    size: Vec3 = Field(description="box: half-extents; cylinder: (radius, half-height, 0); sphere: (radius, 0, 0); mesh: scale")
    mesh: str | None = None
    mass: float = Field(0.1, gt=0, description="kg")
    friction: float = Field(1.0, gt=0)
    pose: Pose = Pose()
    fixed: bool = False

    @model_validator(mode="after")
    def _mesh_needs_path(self) -> ObjectSpec:
        if (self.shape == "mesh") != (self.mesh is not None):
            raise ValueError(f"object '{self.name}': mesh path is required for shape 'mesh' and only for it")
        return self


class Region(Spec):
    """Axis-aligned box in world frame, used for goals and placement."""

    name: str
    center: Vec3
    half_extents: Vec3


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


class SceneSpec(Spec):
    robot: RobotSpec
    objects: tuple[ObjectSpec, ...] = ()
    regions: tuple[Region, ...] = ()
    randomization: RandomizationSpec = RandomizationSpec()

    @model_validator(mode="after")
    def _unique_names(self) -> SceneSpec:
        names = [o.name for o in self.objects] + [r.name for r in self.regions]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate object/region names: {sorted(dupes)}")
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


LeafSuccessSpec = Annotated[ObjectInRegion | ObjectLifted | ObjectOnObject, Field(discriminator="kind")]


class AllOf(Spec):
    """Every condition holds at the same step. Leaves only (no nesting), so a task stays a flat
    list of checks: e.g. sorting = several object_in_region, stacking = object_on_object chain."""

    kind: Literal["all_of"] = "all_of"
    conditions: tuple[LeafSuccessSpec, ...] = Field(min_length=1)


SuccessSpec = Annotated[ObjectInRegion | ObjectLifted | ObjectOnObject | AllOf, Field(discriminator="kind")]


def success_conditions(success: ObjectInRegion | ObjectLifted | ObjectOnObject | AllOf) -> tuple:
    """The leaf conditions of a success spec, in order (one for a leaf, all of them for AllOf)."""
    return tuple(success.conditions) if isinstance(success, AllOf) else (success,)


def success_objects(success: ObjectInRegion | ObjectLifted | ObjectOnObject | AllOf) -> tuple[str, ...]:
    """Every object a success spec mentions (supports included), first mention first, no repeats."""
    names: list[str] = []
    for c in success_conditions(success):
        for n in (c.object, getattr(c, "support", None)):
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
                self.scene.object(c.object)
                if isinstance(c, ObjectInRegion):
                    self.scene.region(c.region)
                if isinstance(c, ObjectOnObject):
                    self.scene.object(c.support)
        except KeyError as e:
            raise ValueError(f"success refers to a missing name: {e.args[0]}") from None
        return self
