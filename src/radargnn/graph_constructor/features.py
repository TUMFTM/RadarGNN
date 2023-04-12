import numpy as np
from typing import Tuple


# Calculate E(n) equivariant metrics of a point pair and its normal vector
def get_En_equivariant_point_pair_metrics(p1: np.ndarray, p2: np.ndarray, v1: np.ndarray, v2: np.ndarray, mode: str) -> Tuple[float]:
    """ Calculates point-pair features between two radar points.

    Args:
        p1: Spatial coordinates of point 1.
        p2: Spatial coordinates of point 2.
        v1: Velocity vector of point 1.
        v2: Velocity vector of point 2.
        mode: Directed or undirected edges.

    Returns:
        d: Euclidean distance between the points.
        theta_v1_v2: Angle between velocity vectors.
        theta_d_v_min: Angle between velocity vector of point 1 and the points connection vector.
        theta_d_v_max: Angle between velocity vector of point 2 and the points connection vector.
    """

    # handle zero velocity vector
    if sum(v1 == np.zeros_like(v1)) == v1.shape[0] and \
       sum(v2 == np.zeros_like(v2)) == v2.shape[0]:
        v1_norm = np.zeros_like(v1)
        v2_norm = np.zeros_like(v2)

    elif sum(v1 == np.zeros_like(v1)) == v1.shape[0]:
        v1_norm = np.zeros_like(v1)
        v2_norm = v2 / np.linalg.norm(v2, ord=2)

    elif sum(v2 == np.zeros_like(v2)) == v2.shape[0]:
        v1_norm = v1 / np.linalg.norm(v1, ord=2)
        v2_norm = np.zeros_like(v2)

    else:
        # normalized velocity
        v1_norm = v1 / np.linalg.norm(v1, ord=2)
        v2_norm = v2 / np.linalg.norm(v2, ord=2)

    # get l2 norm of distance
    d = np.linalg.norm(p1 - p2, ord=2)

    # angle between velocity vectors
    dot_prod = np.dot(v1_norm.T, v2_norm)

    # correction if rounding leads to dot product bigger than 1 -> To achieve maximal robustness and not obtain any "nan" (maybe outsource as serperate function)
    if abs(dot_prod) > 1:
        if (abs(dot_prod) - 1) < 1E-3:
            if dot_prod > 0:
                dot_prod = np.array([[1]])
            elif dot_prod < 0:
                dot_prod = np.array([[-1]])
        else:
            raise Exception("Error in dot product calculation")

    theta_v1_v2 = np.arccos(dot_prod)[0][0] * 180 / np.pi

    if mode == "directed":
        # normalized distance vector
        if np.linalg.norm(p2 - p1, ord=2) == 0:
            d_vec_norm = np.zeros_like(p2)
        else:
            d_vec_norm = (p2 - p1) / np.linalg.norm(p2 - p1, ord=2)

        dot_prod = np.dot(v1_norm.T, d_vec_norm)

        # correction if rounding leads to dot product bigger than 1 -> To achieve maximal robustness and not obtain any "nan" (maybe outsource as serperate function)
        if abs(dot_prod) > 1:
            if (abs(dot_prod) - 1) < 1E-3:
                if dot_prod > 0:
                    dot_prod = np.array([[1]])
                elif dot_prod < 0:
                    dot_prod = np.array([[-1]])
            else:
                raise Exception("Error in dot product calculation")

        theta_d_v1 = np.arccos(dot_prod)[0][0] * 180 / np.pi

        dot_prod = np.dot(v2_norm.T, d_vec_norm)

        # correction if rounding leads to dot product bigger than 1 -> To achieve maximal robustness and not obtain any "nan" (maybe outsource as serperate function)
        if abs(dot_prod) > 1:
            if (abs(dot_prod) - 1) < 1E-3:
                if dot_prod > 0:
                    dot_prod = np.array([[1]])
                elif dot_prod < 0:
                    dot_prod = np.array([[-1]])
            else:
                raise Exception("Error in dot product calculation")

        theta_d_v2 = np.arccos(dot_prod)[0][0] * 180 / np.pi

        theta_d_v_min = theta_d_v1
        theta_d_v_max = theta_d_v2

    elif mode == "undirected":
        # normalized distance vector
        if np.linalg.norm(p1 - p2, ord=2) == 0:
            d1_vec_norm = np.zeros_like(p2)
        else:
            d1_vec_norm = (p1 - p2) / np.linalg.norm(p1 - p2, ord=2)

        theta_d1_v1 = np.arccos(np.dot(v1_norm.T, d1_vec_norm))[0][0] * 180 / np.pi
        theta_d1_v2 = np.arccos(np.dot(v2_norm.T, d1_vec_norm))[0][0] * 180 / np.pi

        if np.linalg.norm(p2 - p1, ord=2) == 0:
            d2_vec_norm = np.zeros_like(p2)
        else:
            d2_vec_norm = (p2 - p1) / np.linalg.norm(p2 - p1, ord=2)

        theta_d2_v1 = np.arccos(np.dot(v1_norm.T, d2_vec_norm))[0][0] * 180 / np.pi
        theta_d2_v2 = np.arccos(np.dot(v2_norm.T, d2_vec_norm))[0][0] * 180 / np.pi

        theta_d_v1 = min(theta_d1_v1, theta_d2_v1)
        theta_d_v2 = min(theta_d1_v2, theta_d2_v2)

        theta_d_v_min = min(theta_d_v1, theta_d_v2)
        theta_d_v_max = max(theta_d_v1, theta_d_v2)

    return d, theta_v1_v2, theta_d_v_min, theta_d_v_max
