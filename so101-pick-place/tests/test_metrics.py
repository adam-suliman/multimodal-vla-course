import numpy as np
import pytest
from so101_lab.metrics import SuccessTracker


def sample(position=(0, 0, 0.015), **kwargs):
    return (
        dict(
            position=np.array(position),
            rotation=np.eye(3),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            finger_contacts=(False, False),
            gripper_open=True,
            dt=1 / 120,
        )
        | kwargs
    )


def tracker():
    return SuccessTracker(0.015, (0, 0), 0.0)


def lift(t, height=0.025):
    t.update(
        **sample(
            (0, 0, 0.015 + height), finger_contacts=(True, True), gripper_open=False
        )
    )


def hold(t, s, frames=120):
    for _ in range(frames):
        t.update(**s)
    return t.success


def test_lift_release_and_full_second():
    t = tracker()
    lift(t)
    assert not hold(t, sample(), 119)
    assert hold(t, sample(), 1)
    assert t.bilateral_contact_seen


@pytest.mark.parametrize("height", [0, 0.019, 0.019999])
def test_insufficient_lift(height):
    t = tracker()
    lift(t, height)
    assert not hold(t, sample(), 240)


def test_held_above_target():
    t = tracker()
    lift(t)
    assert not hold(
        t, sample((0, 0, 0.045), finger_contacts=(True, True), gripper_open=False), 240
    )
    assert not hold(t, sample((0, 0, 0.045)), 240)


@pytest.mark.parametrize("xy", [(0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04)])
def test_partial_containment(xy):
    t = tracker()
    lift(t)
    assert not hold(t, sample((*xy, 0.015)), 240)


def test_rotated_corner_outside():
    t = tracker()
    lift(t)
    angle = np.pi / 4
    rot = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    assert not hold(t, sample((0.03, 0, 0.015), rotation=rot), 240)


def test_transient_crossing_resets_dwell():
    t = tracker()
    lift(t)
    assert not hold(t, sample(), 110)
    t.update(**sample((0.1, 0, 0.015)))
    assert not hold(t, sample(), 110)
    assert hold(t, sample(), 10)


@pytest.mark.parametrize(
    "extra",
    [
        {"linear_velocity": np.array([0.02, 0, 0])},
        {"angular_velocity": np.array([0.2, 0, 0])},
        {"finger_contacts": (True, False)},
        {"gripper_open": False},
    ],
)
def test_motion_or_remaining_contact(extra):
    t = tracker()
    lift(t)
    assert not hold(t, sample(**extra), 240)


def test_nonfinite_state_fails_closed():
    with pytest.raises(ValueError):
        tracker().update(**sample((np.nan, 0, 0)))
