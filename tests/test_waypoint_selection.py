import numpy as np
import pytest

from equiv_primitive.policies.utils.misc import (
    choose_ids,
    choose_ids_rdp,
    closest_object_center_id,
)


def test_closest_approach_anchor_is_preserved_in_normal_selection():
    candidate_ids = list(range(12))
    essential_ids = [2, 9]
    required_id = 6

    np.random.seed(0)
    selected_ids = choose_ids(
        8, candidate_ids, essential_ids, required_ids=[required_id]
    )

    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == len(set(selected_ids)) == 8
    assert {0, 2, required_id, 9, 11}.issubset(selected_ids)


def test_overlapping_closest_approach_anchor_does_not_duplicate_ids():
    np.random.seed(0)
    selected_ids = choose_ids(
        8, list(range(12)), [2, 9], required_ids=[2]
    )

    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == len(set(selected_ids)) == 8
    assert {0, 2, 9, 11}.issubset(selected_ids)


def test_missing_visible_target_point_cloud_raises():
    with pytest.raises(ValueError, match="no valid visible point cloud"):
        closest_object_center_id(
            eef_positions=np.zeros((3, 3)),
            object_point_clouds=np.zeros((3, 0, 3)),
            candidate_ids=[0, 1, 2],
            visibility=np.ones(3, dtype=bool),
        )


def test_closest_object_center_uses_only_visible_candidate_frames():
    eef_positions = np.array([[3.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.5, 0.0, 0.0]])
    object_point_clouds = np.array(
        [
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )

    selected_id = closest_object_center_id(
        eef_positions,
        object_point_clouds,
        candidate_ids=[0, 1, 2],
        visibility=np.array([True, False, True]),
    )

    assert selected_id == 2


def test_rdp_selection_preserves_closest_approach_anchor():
    candidate_ids = list(range(12))
    trajectory = np.stack(
        [np.arange(12, dtype=np.float64), np.zeros(12), np.zeros(12)], axis=1
    )

    np.random.seed(0)
    selected_ids = choose_ids_rdp(
        trajectory, 8, candidate_ids, [2, 9], required_ids=[6]
    )

    assert selected_ids == sorted(selected_ids)
    assert len(selected_ids) == len(set(selected_ids)) == 8
    assert {0, 2, 6, 9, 11}.issubset(selected_ids)
