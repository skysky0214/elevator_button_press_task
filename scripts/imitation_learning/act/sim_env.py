"""Stub sim_env — ACT's original used dm_control/MuJoCo for ALOHA eval.
We use Isaac Sim for rollout, so only BOX_POSE stub is needed for imports.
"""

BOX_POSE = [None]


def make_sim_env(task_name):
    raise NotImplementedError("sim_env disabled; rollout in Isaac Sim via separate script")


def sample_box_pose():
    raise NotImplementedError


def sample_insertion_pose():
    raise NotImplementedError
