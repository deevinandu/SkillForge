"""T2: skills and primitives. Every skill implements contracts.Skill.

`PickPlace` is v0.2 (primitives + grasp planner + retry/slip recovery). The v0.1 baseline stays
available as `PickPlaceV01` for comparison and regression. `PickFromClutter` (PickPlace with a
hand-clearance grasp choice) and `PushToRegion` (two-finger push for objects too wide to grasp), `TubeIntoRack` (stand a tube
upright in a rack slot), `FillRack` (several tubes into their slots, in order), `SortTubes` (tubes into racks by the cap
colour it sees through the wrist camera), `ToppleChain` (set up a domino chain, repair what gets
disturbed, knock it over), `HangMug` (hang a mug by its handle on a peg) are the seed skills for
the clutter and push tasks.
"""

from .dominoes import ToppleChain
from .grasp import Grasp, grasp_to_world, plan_grasps, select_grasp
from .hang_mug import HangMug
from .loader import load_skill
from .pick_clutter import PickFromClutter
from .pick_place import PickPlace
from .pick_place_v01 import PickPlaceV01
from .place_in_order import PlaceInOrder
from .push import PushToRegion
from .rack_fill import FillRack
from .sort_tubes import SortTubes
from .tube_rack import TubeIntoRack

__all__ = ["FillRack", "Grasp", "HangMug", "PickFromClutter", "PickPlace", "PickPlaceV01", "PlaceInOrder", "PushToRegion", "SortTubes", "ToppleChain", "TubeIntoRack", "grasp_to_world", "load_skill", "plan_grasps", "select_grasp"]
