from collections import defaultdict
from enum import IntEnum
from itertools import chain, combinations, product
from typing import Dict, List, Optional, Set, Tuple, Type, Union

import awkward as ak
import networkx as nx
import numpy as np

from utils.stableid import Units

Point = Tuple[int, int]


class EntIdx(IntEnum):
    unit_type = 0
    alliance = 1
    health = 2
    shield = 3
    energy = 4
    cargo_space_taken = 5
    build_progress = 6
    health_ratio = 7
    shield_ratio = 8
    energy_ratio = 9
    display_type = 10
    owner = 11
    x = 12
    y = 13
    facing = 14
    radius = 15
    cloak = 16
    is_selected = 17
    is_blip = 18
    is_powered = 19
    mineral_contents = 20
    vespene_contents = 21
    cargo_space_max = 22
    assigned_harvesters = 23
    ideal_harvesters = 24
    weapon_cooldown = 25
    order_length = 26
    order_id_0 = 27
    order_id_1 = 28
    tag = 29
    hallucination = 30
    buff_id_0 = 31
    buff_id_1 = 32
    addon_unit_type = 33
    active = 34
    is_on_screen = 35
    order_progress_0 = 36
    order_progress_1 = 37
    order_id_2 = 38
    order_id_3 = 39
    is_in_cargo = 40
    buff_duration_remain = 41
    buff_duration_max = 42
    attack_upgrade_level = 43
    armor_upgrade_level = 44
    shield_upgrade_level = 45


def sq_dist(p1: Point, p2: Point) -> int:
    return np.sum(np.square(np.subtract(p2, p1)))


def distance(p1: Point, p2: Point) -> float:
    return np.linalg.norm(np.subtract(p2, p1))


def get_locations(data: np.ndarray) -> Dict[Units, Dict[int, Point]]:
    locations: Dict[Units, Dict[int, Point]] = defaultdict(dict)
    num_entities = data.shape[1]

    for e_idx in range(num_entities):
        entity = data[:, e_idx]
        tag = int(entity[EntIdx.tag])
        x = int(entity[EntIdx.x])
        y = int(entity[EntIdx.y])
        unit_type = Units(entity[EntIdx.unit_type])

        locations[unit_type][tag] = (x, y)

    return locations


class TrajectoryWalker:
    def __init__(self, summarizer_types: List[Type]):
        """Create a container that houses summary objects for generating."""
        self.summarizers: List[TrajectorySummarizer] = [
            summarizer() for summarizer in summarizer_types
        ]

    def __call__(self, step: int, data: Union[np.ndarray, ak.Array]) -> None:
        """
        Update internal summary state models.

        :param step: The step number.
        :param data: A 2d array containing entity `feature_units` data.
        """
        for summarizer in self.summarizers:
            summarizer(step, data)

    def export(self) -> List[str]:
        """Export tags from all internal summary models."""
        outs: List[str] = []
        for summarizer in self.summarizers:
            outs.extend(summarizer.export())
        return outs


class TrajectorySummarizer:
    """Base class for trajectory summarizers"""

    def __init__(self):
        self._exports: List[str] = []
        self.last_step: int = 0

    def __call__(self, step: int, data: Union[np.ndarray, ak.Array]) -> None:
        """
        Evaluate the logical state of the given game step.

        :param step: The step number.
        :param data: A 2d array containing entity `feature_units` data.
        """
        if isinstance(data, ak.Array):
            data = ak.to_numpy(data)
        self._process_step(data)
        self.last_step = step

    def _process_step(data: np.ndarray) -> None:
        """
        Evaluate the logical state of the given game step.

        :param data: A 2d array containing entity `feature_units` data.
        """
        ...

    def export(self):
        """
        Export the generated tags as a list of strings

        :returns: String tags that describe the summary.
        """
        return self._exports.copy()


class EntityMoves(TrajectorySummarizer):
    """
    trajectory summarizer to generate tags describing entity movement
    e.g. "entity A moves from [x0, y0] to [x1, y1]
    """

    def __init__(self):
        super().__init__()
        self.cur_locs: Dict[int, Point] = {}
        self.start_locs: Dict[int, Point] = {}
        self.start_steps: Dict[int, int] = {}
        self.unit_types: Dict[int, Units] = {}

    def _process_step(self, data: np.ndarray) -> None:
        """
        Evaluate the logical state of the given game step.

        :param data: A 2d array containing entity `feature_units` data.
        """
        num_entities = data.shape[1]

        for e_idx in range(num_entities):
            entity = data[:, e_idx]
            tag = entity[EntIdx.tag]
            x = entity[EntIdx.x]
            y = entity[EntIdx.y]
            loc = x, y

            if tag not in self.unit_types:
                self.unit_types[tag] = Units(entity[EntIdx.unit_type])

            try:
                # Check for the first step.
                if tag not in self.cur_locs:
                    continue

                # Now that we know it is not the first step,
                # have we moved?
                if self.cur_locs[tag] == loc:
                    # We did not move.
                    if tag in self.start_locs:
                        # We stopped moving.
                        self._add_event(tag)

                # else: We have moved.
                elif tag not in self.start_locs:
                    # Set the start from the previous position.
                    self.start_locs[tag] = self.cur_locs[tag]
                    self.start_steps[tag] = self.last_step
            finally:
                self.cur_locs[tag] = loc

    def export(self):
        # export any "unfinished" movements
        tags = list(self.start_locs)
        for tag in tags:
            self._add_event(tag)
        return super().export()

    def _add_event(self, tag: int) -> None:
        unit_type = self.unit_types[tag]
        start = self.start_locs[tag]
        start_step = self.start_steps[tag]
        end_step = self.last_step
        cur = self.cur_locs[tag]

        date_label = f"{start_step} -- {end_step}"
        event = f"{unit_type.name} {tag} moves from {start} to {cur}"

        self._exports.append(f"{date_label}: {event}")

        del self.start_locs[tag]
        del self.start_steps[tag]


class EntityReached(TrajectorySummarizer):
    REACHABLE_ENTITIES = {
        Units.Beacon_Terran: "beacon",
        Units.NaturalMinerals: "shard",
    }

    def __init__(self):
        super().__init__()
        self._entity_locs: Dict[Units, Dict[int, Point]] = defaultdict(dict)

    def _process_step(self, data: np.ndarray) -> None:
        """
        Evaluate the logical state of the given game step.

        :param data: A 2d array containing entity `feature_units` data.
        """
        cur_entity_locs: Dict[Units, Dict[int, Point]] = defaultdict(dict)
        cur_marine_locs: Dict[int, Point] = {}

        num_entities = data.shape[1]

        # Record locations of marines and reachable entities
        # at this time step.
        for e_idx in range(num_entities):
            entity = data[:, e_idx]
            tag = int(entity[EntIdx.tag])
            x = int(entity[EntIdx.x])
            y = int(entity[EntIdx.y])
            loc = x, y
            unit_type = Units(entity[EntIdx.unit_type])

            if unit_type in self.REACHABLE_ENTITIES:
                cur_entity_locs[unit_type][tag] = loc
            elif unit_type == Units.Marine:
                cur_marine_locs[tag] = loc

        # Check if any reachable entities have moved or disappeared.
        for unit_type, name in self.REACHABLE_ENTITIES.items():
            entity_locs = self._entity_locs[unit_type]
            cur_locs = cur_entity_locs[unit_type]

            name_cap = name.capitalize()
            entities_to_remove: List[int] = []

            for tag, loc in self._entity_locs[unit_type].items():
                if tag in cur_locs:
                    cur_loc = cur_locs[tag]
                    if loc == cur_loc:
                        continue

                    # else: Location has changed.
                    self._add_event(f"{name_cap} {tag} moves from {loc} to {cur_loc}")
                    entity_locs[tag] = cur_loc

                else:
                    # A previously observed entity no longer exists.
                    # We assume it has been collected.
                    self._add_event(f"{name_cap} {tag} at {loc} is collected")
                    entities_to_remove.append(tag)

                nearest_marine = self._nearest_marine(loc, cur_marine_locs)
                self._add_event(f"Marine {nearest_marine} collects {name} {tag}")

            # Remove collected entities.
            for tag in entities_to_remove:
                del entity_locs[tag]

            for tag in cur_locs.keys() - entity_locs.keys():
                # This is a new entity.
                loc = cur_locs[tag]
                self._add_event(f"{name_cap} {tag} appears at {loc}")
                entity_locs[tag] = loc

    def _nearest_marine(self, entity_loc: int, marine_locs: Dict[int, Point]) -> int:
        # We assume the marine closest to the entity's
        # previous location was the one that collected it.
        nearest_marine: int
        min_distance = np.inf
        for marine_tag, marine_loc in marine_locs.items():
            dist = distance(entity_loc, marine_loc)
            if dist < min_distance:
                min_distance = dist
                nearest_marine = marine_tag
        return nearest_marine

    def _add_event(self, event: str) -> None:
        date_label = f"{self.last_step} -- {self.last_step}"
        self._exports.append(f"{date_label}: {event}")


class EntityMovesRelative(TrajectorySummarizer):
    REACHABLE_ENTITIES = {
        Units.Beacon_Terran: "beacon",
        Units.NaturalMinerals: "shard",
    }

    def __init__(self):
        super().__init__()
        self.distances: nx.Graph = nx.Graph()
        self.start: nx.Graph = nx.Graph()
        self.unit_types: Dict[int, Units] = {}

    def _process_step(self, data: np.ndarray) -> None:
        """
        Evaluate the logical state of the given game step.

        :param data: A 2d array containing entity `feature_units` data.
        """
        locs: Dict[Units, Dict[int, Point]] = get_locations(data)

        for unit_type in locs:
            for tag in locs[unit_type]:
                if tag not in self.unit_types:
                    self.unit_types[tag] = unit_type

        # Construct a graph that records the distances between reachable
        # entities and other entities.
        reachable_entities = set(self.REACHABLE_ENTITIES.keys())
        reachable_locs = chain.from_iterable(
            locs[utype].items() for utype in reachable_entities
        )
        entity_locs = chain.from_iterable(
            locs[utype].items() for utype in locs.keys() - reachable_entities
        )
        for (u1, p1), (u2, p2) in product(entity_locs, reachable_locs):
            curr_dist = sq_dist(p1, p2)

            # Check for the first step.
            if not self.distances.has_edge(u1, u2):
                self.distances.add_edge(u1, u2, dist=curr_dist)
                continue

            try:
                # Now that we know it is not the first step, has the
                # distance between u1 and u2 changed?
                prev_dist = self.distances[u1][u2]["dist"]

                if prev_dist == curr_dist:
                    # Has the distance been changing?
                    if self.start.has_edge(u1, u2):
                        # The distance is no longer changing.
                        self._add_event(u1, u2)
                    continue

                # else: The distance between the entities has changed.
                if not self.start.has_edge(u1, u2):
                    # The distance between the entities has just started
                    # to change; record the initial distance.
                    self.start.add_edge(u1, u2, dist=prev_dist, step=self.last_step)
                    continue

                start_dist = self.start[u1][u2]["dist"]

                if start_dist < prev_dist and prev_dist > curr_dist:
                    # The entities were initially moving apart, but now
                    # are moving closer.
                    self._add_event(u1, u2)
                    self.start.add_edge(u1, u2, dist=prev_dist, step=self.last_step)

                elif start_dist > prev_dist and prev_dist < curr_dist:
                    # The entities were initially moving closer, but now
                    # are moving further apart.
                    self._add_event(u1, u2)
                    self.start.add_edge(u1, u2, dist=prev_dist, step=self.last_step)
            finally:
                self.distances[u1][u2]["dist"] = curr_dist

    def _add_event(self, entity_tag: int, reachable_tag: int):
        start_dist = self.start[entity_tag][reachable_tag]["dist"]
        end_dist = self.distances[entity_tag][reachable_tag]["dist"]

        start_step = self.start[entity_tag][reachable_tag]["step"]
        end_step = self.last_step

        if start_dist > end_dist:
            direction = "closer to"
        elif start_dist < end_dist:
            direction = "farther from"
        else:
            raise RuntimeError("Distance should have changed!")

        date_label = f"{start_step} -- {end_step}"
        event = " ".join(
            map(
                str,
                [
                    self.unit_types[entity_tag].name,
                    entity_tag,
                    "moves",
                    direction,
                    self.REACHABLE_ENTITIES[self.unit_types[reachable_tag]],
                    reachable_tag,
                ],
            )
        )

        self._exports.append(f"{date_label}: {event}")

        self.start.remove_edge(entity_tag, reachable_tag)

    def export(self):
        # Export any "unfinished" movements.
        edges = list(self.start.edges)
        for u1, u2 in edges:
            if self.unit_types[u1] in self.REACHABLE_ENTITIES:
                u1, u2 = u2, u1  # Swap so that reachable_tag is u2
            self._add_event(u1, u2)
        return super().export()


class Groups(TrajectorySummarizer):
    GROUP_ENTITIES = [Units.Marine]

    def __init__(self, threshold: int = 10) -> None:
        super().__init__()

        self._groups: Dict[Units, Dict[int, Set[int]]] = defaultdict(dict)
        self._group_counter: int = 0
        self.threshold: int = threshold

        self._group_cur_locs: Dict[Units, Dict[int, Point]] = defaultdict(dict)
        self._group_start_locs: Dict[Units, Dict[int, Point]] = defaultdict(dict)
        self._start_steps: Dict[Units, Dict[int, int]] = defaultdict(dict)

    def _process_step(self, data: np.ndarray) -> None:
        """
        Evaluate the logical state of the given game step.

        :param data: A 2d array containing entity `feature_units` data.
        """
        locations: Dict[Units, Dict[int, Point]] = get_locations(data)

        # Update each collection of groups using the corresponding
        # distance graph. Then check for group-level movement.
        for unit_type in Groups.GROUP_ENTITIES:
            distance_graph = self._to_graph(locations[unit_type])
            self._update_groups(unit_type, distance_graph)
            self._track_movement(unit_type, locations[unit_type])

    @staticmethod
    def _to_graph(locs: Dict[int, Point]) -> Optional[nx.Graph]:
        g = nx.Graph()
        for (t1, p1), (t2, p2) in combinations(locs.items(), 2):
            g.add_edge(t1, t2, dist=distance(p1, p2))
        n, m = g.number_of_nodes(), g.number_of_edges()
        assert 2 * m == n * (n - 1), "Graph is not complete"
        return g

    def _update_groups(self, unit_type: Units, graph: nx.Graph) -> None:
        groups: Dict[int, Set[int]] = self._groups[unit_type]

        # First, check for group subtractions.
        dissolved: List[int] = []
        for tag, g in groups.items():
            # Check if any member of the group has left.
            has_left: Set[int] = set()
            for u in g:
                if all(graph[u][v]["dist"] > self.threshold for v in g - {u}):
                    has_left.add(u)
            substring = ", ".join(map(str, has_left))
            if len(has_left) == 1:
                self._add_event(f"Entity {substring} leaves group {tag}")
            elif len(has_left) > 1:
                self._add_event(f"Entities {substring} leave group {tag}")
            g -= has_left

            # Now, check if the group is empty.
            if not g:
                # Add event if group was previously moving.
                if tag in self._group_start_locs[unit_type]:
                    self._add_event(self._move_msg(unit_type, tag))
                    del self._group_start_locs[unit_type][tag]

                self._add_event(f"Group {tag} is dissolved.")
                dissolved.append(tag)
                del self._group_cur_locs[unit_type][tag]

        for tag in dissolved:
            del groups[tag]

        # Now, check for group formations and additions.
        unit2group: Dict[int, int] = {}
        for tag, g in groups.items():
            for u in g:
                unit2group[u] = tag

        new_groups: Dict[int, Set[int]] = {}
        for u, v, attr in graph.edges(data=True):
            if attr["dist"] > self.threshold:
                continue

            u_group = self._gid(groups, u)
            if u_group is not None:
                v_group = self._gid(groups, v)
                if v_group is not None:
                    if u_group == v_group:
                        # u and v are already in the same group.
                        continue

                    # Since u and v are in different groups, we will
                    # merge the groups.
                    self._add_event(f"Group {v_group} merges with group {u_group}")
                    groups[u_group] |= groups[v_group]
                    del groups[v_group]
                    continue

                v_group = self._gid(new_groups, v)
                if v_group is not None:
                    # v is in a new group and u is in an existing group,
                    # so we will merge the former group into the latter.
                    substring = ", ".join(map(str, new_groups[v_group]))
                    self._add_event(f"Entities {substring} join group {u_group}")
                    groups[u_group] |= new_groups[v_group]
                    del new_groups[v_group]
                    continue

                # u is in an existing group and v is not in any group.
                self._add_event(f"Entity {v} joins group {u_group}")
                groups[u_group].add(v)
                continue

            u_group = self._gid(new_groups, u)
            if u_group is not None:
                v_group = self._gid(groups, v)
                if v_group is not None:
                    # u is in a new group and v is in an existing group,
                    # so we will merge the former group into the latter.
                    substring = ", ".join(map(str, new_groups[u_group]))
                    self._add_event(f"Entities {substring} join group {v_group}")
                    groups[v_group] |= new_groups[u_group]
                    del new_groups[u_group]
                    continue

                v_group = self._gid(new_groups, v)
                if v_group is not None:
                    # u and v are both in new groups, so we will
                    # merge the latter's group into the former's.
                    new_groups[u_group] |= new_groups[v_group]
                    del new_groups[v_group]
                    continue

                # u is in a new group and v is not in any group.
                new_groups[u_group].add(v)
                continue

            # u is not in any group.
            v_group = self._gid(groups, v)
            if v_group is not None:
                # v is in an existing group, so u joins that group.
                self._add_event(f"Entity {u} joins {v_group}")
                groups[v_group].add(u)
                continue

            v_group = self._gid(new_groups, v)
            if v_group is not None:
                # v is in a new group, so u joins that group.
                new_groups[v_group].add(u)
                continue

            # Neither u nor v are in any group, so create a new one.
            new_groups[self._new_gid()] = {u, v}

        for tag, g in new_groups.items():
            substring = ", ".join(map(str, g))
            self._add_event(f"Entities {substring} form group {tag}")
            groups[tag] = g

    @staticmethod
    def _gid(groups: Dict[int, Set[int]], u: int) -> Optional[int]:
        for tag, g in groups.items():
            if u in g:
                return tag
        return None

    def _new_gid(self) -> int:
        self._group_counter += 1
        return self._group_counter

    def _track_movement(self, unit_type, locs: Dict[int, Point]) -> None:
        # Copy fields to local variables for convenience.
        groups: Dict[int, Set[int]] = self._groups[unit_type]
        cur_locs: Dict[int, Point] = self._group_cur_locs[unit_type]
        start_locs: Dict[int, Point] = self._group_start_locs[unit_type]
        start_steps: Dict[int, int] = self._start_steps[unit_type]

        for g_tag, group in groups.items():
            # Compute current group location.
            x, y = np.mean([locs[tag] for tag in group], axis=0)
            group_loc = round(x), round(y)

            try:
                # Check for the first step.
                if g_tag not in cur_locs:
                    continue

                # Now that we know it is not the first step,
                # have we moved?
                if cur_locs[g_tag] == group_loc:
                    # We did not move.
                    if g_tag in start_locs:
                        # We stopped moving.
                        self._add_move_event(unit_type, g_tag)

                # else: We have moved.
                elif g_tag not in start_locs:
                    # Set the start from the previous position.
                    start_locs[g_tag] = cur_locs[g_tag]
                    start_steps[g_tag] = self.last_step
            finally:
                cur_locs[g_tag] = group_loc

    def export(self) -> List[str]:
        # Export any "unfinished" movements.
        unit_types = list(self._group_start_locs)
        for unit_type in unit_types:
            tags = list(self._group_start_locs[unit_type])
            for tag in tags:
                self._add_move_event(unit_type, tag)
        return super().export()

    def _move_msg(self, unit_type: Units, tag: int):
        start = self._group_start_locs[unit_type][tag]
        cur = self._group_cur_locs[unit_type][tag]
        return f"Group {tag} moves from {start} to {cur}"

    def _add_move_event(self, unit_type: Units, tag: int) -> None:
        start_loc = self._group_start_locs[unit_type][tag]
        end_loc = self._group_cur_locs[unit_type][tag]

        start_step = self._start_steps[unit_type][tag]
        end_step = self.last_step

        date_label = f"{start_step} -- {end_step}"
        event = f"Group {tag} moves from {start_loc} to {end_loc}"

        self._exports.append(f"{date_label}: {event}")

        del self._group_start_locs[unit_type][tag]
        del self._start_steps[unit_type][tag]

    def _add_event(self, event: str) -> None:
        date_label = f"{self.last_step} -- {self.last_step}"
        self._exports.append(f"{date_label}: {event}")
