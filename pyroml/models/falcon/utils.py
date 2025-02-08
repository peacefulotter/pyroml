import os
import math
import torch
import numpy as np
import gurobipy as gp

from gurobipy import GRB


def entropy(p: torch.Tensor):
    return torch.special.entr(p).sum()


def cosine_annealing(
    initial_value: float, min_value: float, cur_step: int, total_steps: int
):
    assert min_value <= initial_value
    assert cur_step <= total_steps
    return min_value + (initial_value - min_value) * 0.5 * (
        1 + math.cos(math.pi * cur_step / total_steps)
    )


def compute_assignment_from_cost(cost: np.ndarray, reg_coef=1.0, time_limit=60):
    with gp.Env(empty=True) as env:
        wlsaccessID = os.getenv("GRB_WLSACCESSID", None)
        licenseID = os.getenv("GRB_LICENSEID", None)
        wlsSecrets = os.getenv("GRB_WLSSECRET", None)

        env.setParam("OutputFlag", 0)
        env.setParam("WLSACCESSID", wlsaccessID) if wlsaccessID is not None else None
        env.setParam("LICENSEID", int(licenseID)) if licenseID is not None else None
        env.setParam("WLSSECRET", wlsSecrets) if wlsSecrets is not None else None
        env.start()

        model = gp.Model(env=env)
        model.params.TimeLimit = time_limit
        # model.setParam("MIPGap", mipgap)

        fine_classes, coarse_classes = cost.shape
        assert fine_classes > coarse_classes

        assignments = model.addMVar((fine_classes, coarse_classes), vtype=GRB.BINARY)

        cls_objective = (-cost * assignments).sum()
        set_sizes = assignments.sum(axis=0)
        reg_objective = (set_sizes * set_sizes).sum() / coarse_classes - (
            (fine_classes / coarse_classes) ** 2
        )

        objective = cls_objective + reg_coef * reg_objective
        model.setObjective(objective, GRB.MINIMIZE)

        model.addConstr(assignments.sum(axis=1) == np.full(fine_classes, 1))
        model.addConstr(assignments.sum(axis=0) >= np.full(coarse_classes, 1))

        # Optimize the model
        model.optimize()

    return torch.from_numpy(assignments.x).float()
