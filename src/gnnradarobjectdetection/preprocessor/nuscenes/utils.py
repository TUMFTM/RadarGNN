import numpy as np

from nuscenes import nuscenes


def extended_points_in_box(box: nuscenes.Box, points: np.ndarray, wlh_factor: float = 1.0,
                           wlh_offset: float = 0.0, use_z: bool = True) -> np.ndarray:
    """ Returns a mask indicating whether or not each point is inside the bounding box.

    Inspired by NuScenes points_in_box.

    Arguments:
        box: Bounding box in nuScenes format.
        points: Radar points (3, N).
        wlh_factor: Factor to inflate or deflate the box (1.1 makes it 10% larger in all dimensions).
        wlh_offset: Offset to inflate or deflate the box (1.0 makes it 1 m larger in all dimensions, on both sides).
        use_z: Whether the z coordinate is taken into account.

    Returns:
        mask: Mask indicating whether or not each point is inside the bounding box (N,).
    """
    corners = box.corners(wlh_factor=wlh_factor)

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v) / np.linalg.norm(i)
    jv = np.dot(j, v) / np.linalg.norm(j)
    kv = np.dot(k, v) / np.linalg.norm(k)

    mask_x = np.logical_and(0 - wlh_offset <= iv, iv <= np.linalg.norm(i) + wlh_offset)
    mask_y = np.logical_and(0 - wlh_offset <= jv, jv <= np.linalg.norm(j) + wlh_offset)

    if use_z:
        mask_z = np.logical_and(0 - wlh_offset <= kv, kv <= np.linalg.norm(k) + wlh_offset)
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    else:
        mask = np.logical_and(mask_x, mask_y)

    return mask
