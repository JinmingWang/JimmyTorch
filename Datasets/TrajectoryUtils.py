from .DatasetUtils import *
import matplotlib.pyplot as plt
import numpy as np

Traj = FT32[Tensor, "L 2"]
BatchTraj = FT32[Tensor, "B L 2"]


def getLng(trajs: BatchTraj | Traj) -> Any:
    """
    Get the longitude of the trajectory.
    :param trajs: Trajectory or batch of trajectories.
    :return: Longitude of the trajectory.
    """
    return trajs[..., 0]


def getLat(trajs: BatchTraj | Traj) -> Any:
    """
    Get the latitude of the trajectory.
    :param trajs: Trajectory or batch of trajectories.
    :return: Latitude of the trajectory.
    """
    return trajs[..., 1]


def computeDistance(trajs: BatchTraj | Traj) -> Any:
    """
    Compute the distance of the trajectory.
    :param trajs: Trajectory or batch of trajectories.
    :return: Distance of trajectories.
    """
    return torch.sqrt(torch.sum(torch.square(trajs[..., 1:] - trajs[..., :-1]), dim=-1))


def cropPadTraj(traj: Traj, target_len: int, pad_value: float = 0.0) -> Traj:
    """
    Pad a trajectory to the maximum length.

    :param traj: Trajectory to be padded.
    :param target_len: Maximum length of the trajectory.
    :param pad_value: Value to pad with. Default is 0.0.
    :return: Padded trajectory.
    """
    if traj.shape[0] >= target_len:
        return traj[:target_len]
    elif traj.shape[0] < target_len:
        pad_size = target_len - traj.shape[0]
        return torch.nn.functional.pad(traj, (0, 0, 0, pad_size), value=pad_value)
    else:
        return traj


def flipTrajWestEast(trajs: BatchTraj | Traj) -> BatchTraj | Traj:
    """
    Flip a trajectory horizontally (West-East).
    To achieve this, we simply negate the longitude values.

    :param trajs: Trajectory or batch of trajectories.
    :return: Flipped trajectory.
    """
    trajs[..., 0] = -trajs[..., 0]
    return trajs


def flipTrajNorthSouth(trajs: BatchTraj | Traj) -> BatchTraj | Traj:
    """
    Flip a trajectory vertically (North-South).
    To achieve this, we simply negate the latitude values.

    :param trajs: Trajectory or batch of trajectories.
    :return: Flipped trajectory.
    """
    trajs[..., 1] = -trajs[..., 1]
    return trajs


def centerTraj(trajs: BatchTraj | Traj) -> BatchTraj | Traj:
    """
    Center a trajectory around the origin (0, 0).
    This is done by subtracting the mean of the trajectory from each point.

    :param trajs: Trajectory or batch of trajectories.
    :return: Centered trajectory.
    """
    mean = torch.mean(trajs, dim=-2, keepdim=True)
    trajs -= mean
    return trajs


def zScoreTraj(trajs: BatchTraj | Traj) -> BatchTraj | Traj:
    """
    Standardize a trajectory to have zero mean and unit variance.
    This is done by subtracting the mean and dividing by the standard deviation.

    :param trajs: Trajectory or batch of trajectories.
    :return: Standardized trajectory.
    """
    mean = torch.mean(trajs, dim=-2, keepdim=True)
    std = torch.std(trajs, dim=-2, keepdim=True)
    trajs = (trajs - mean) / std
    return trajs


def minMaxTraj(trajs: BatchTraj | Traj) -> BatchTraj | Traj:
    """
    Normalize a trajectory to the range [0, 1].
    This is done by subtracting the minimum and dividing by the range.

    :param trajs: Trajectory or batch of trajectories.
    :return: Normalized trajectory.
    """
    min_val = torch.min(trajs, dim=-2, keepdim=True)[0]
    max_val = torch.max(trajs, dim=-2, keepdim=True)[0]
    trajs = (trajs - min_val) / (max_val - min_val)
    return trajs


def rotateTraj(trajs: BatchTraj | Traj, angles: Tensor) -> BatchTraj | Traj:
    """
    Rotate a trajectory by a given angle.
    ! Only apply rotation after centering the trajectory.
    This is done using a rotation matrix.

    :param trajs: Trajectory or batch of trajectories.
    :param angles: Angles in degrees to rotate the trajectory. Expected shape: () or (B,).
    :return: Rotated trajectory.
    """
    if trajs.ndim == 2:
        trajs = trajs.unsqueeze(0)
    if angles.ndim == 0:
        angles = angles.unsqueeze(0)
    if angles.shape[0] != trajs.shape[0]:
        angles = angles.expand(trajs.shape[0])

    angles_rad = torch.deg2rad(angles)
    cos_angles = torch.cos(angles_rad)  # (B,)
    sin_angles = torch.sin(angles_rad)  # (B,)
    # rot_mat: (B, 2, 2)
    rot_mat = torch.stack([cos_angles, -sin_angles, sin_angles, cos_angles], dim=-1).reshape(-1, 2, 2)
    # trajs: (B, L, 2), rot_mat: (B, 2, 2)
    return torch.einsum("bij,blj->bil", rot_mat, trajs)


def interpTraj(trajs: Traj | BatchTraj, num_points: int, mode: str = "linear") -> Traj | BatchTraj:
    """
    Interpolate trajectories to a given number of points.

    :param trajs: Trajectory to be interpolated.
    :param num_points: Number of points to interpolate to.
    :return: Interpolated trajectory.
    """

    if trajs.ndim == 2:
        batch_trajs = trajs.unsqueeze(0)
    else:
        batch_trajs = trajs

    interp_trajs = torch.nn.functional.interpolate(batch_trajs.transpose(1, 2), num_points, mode=mode).transpose(1, 2)

    if trajs.ndim == 2:
        interp_trajs = interp_trajs.squeeze(0)
    return interp_trajs


def plotTraj(ax: plt.Axes,
             trajs: Traj | BatchTraj,
             traj_lengths: Optional[Tensor] = None,
             color: str = "blue",
             linewidth: int=1,
             markersize: int=1) -> None:
    """
    Plot a trajectory on the given axis.

    :param ax: Axis to plot on.
    :param trajs: Trajectory or batch of trajectories.
    :param color: Color of the trajectory. Default is "blue".
    :param label: Label for the trajectory. Default is None.
    """
    # Add batch dimension if needed
    if trajs.ndim == 2:
        trajs = trajs.unsqueeze(0)
    trajs = trajs.cpu().numpy()
    if traj_lengths is not None:
        traj_lengths = traj_lengths.cpu().numpy()

    # Plot each trajectory in the batch
    for ti, traj in enumerate(trajs):
        if traj_lengths is not None:
            traj = traj[:traj_lengths[ti]]
        ax.plot(getLng(traj), getLat(traj), color=color, linewidth=linewidth,
                marker='.', markersize=markersize)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def geometricDistance(pred_points: Tensor,
                      gt_points: Tensor,
                      reduction: Callable = "mean"  # "mean", "sum", "none"
                      ) -> Tensor:
    """
    Compute the geometric distance between predicted and ground truth points.

    :param pred_points: Predicted points of shape (..., 2)
    :param gt_points: Ground truth points of shape (..., 2)
    :param reduction: Reduction method to apply to the distance. Default is "none".
                      "mean" or "sum" can be used to reduce the distance.
    :return: Geometric distance
    """

    # Each point is represented as (longitude, latitude)
    # We need to compute the distance between each pair of points in meters
    # Using Haversine formula to compute the distance
    R = 6371000  # Radius of the Earth in meters
    pred_rad = torch.deg2rad(pred_points)
    gt_rad = torch.deg2rad(gt_points)

    dlat = getLat(pred_rad) - getLat(gt_rad)
    dlon = getLng(pred_rad) - getLng(gt_rad)

    a = torch.sin(dlat / 2) ** 2 + torch.cos(getLat(gt_rad)) * torch.cos(getLat(pred_rad)) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c  # Distance in meters, shape: (...,)

    if reduction == "mean":
        return torch.mean(distance)
    elif reduction == "sum":
        return torch.sum(distance)
    return distance


def computeJSD(dataset1: BatchTraj, dataset2: BatchTraj, num_bins: int = 50) -> float:
    """
    Compute Jensen-Shannon Divergence (JSD) between two trajectory datasets.

    :param dataset1: First trajectory dataset of shape (N1, L, 2)
    :param dataset2: Second trajectory dataset of shape (N2, L, 2)
    :param num_bins: Number of bins for histogram computation. Default is 50.
    :return: Jensen-Shannon Divergence value
    """
    # Flatten trajectories to get all points
    points1 = dataset1.reshape(-1, 2).cpu().numpy()  # (N1*L, 2)
    points2 = dataset2.reshape(-1, 2).cpu().numpy()  # (N2*L, 2)

    # Find common bounds for both datasets
    all_points = np.concatenate([points1, points2], axis=0)
    lng_min, lng_max = all_points[:, 0].min(), all_points[:, 0].max()
    lat_min, lat_max = all_points[:, 1].min(), all_points[:, 1].max()

    # Create 2D histograms
    hist1, _, _ = np.histogram2d(points1[:, 0], points1[:, 1],
                                bins=num_bins,
                                range=[[lng_min, lng_max], [lat_min, lat_max]])
    hist2, _, _ = np.histogram2d(points2[:, 0], points2[:, 1],
                                bins=num_bins,
                                range=[[lng_min, lng_max], [lat_min, lat_max]])

    # Normalize to get probability distributions
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()

    # Flatten histograms
    p = hist1.flatten()
    q = hist2.flatten()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = p + eps
    q = q + eps

    # Renormalize
    p = p / p.sum()
    q = q / q.sum()

    # Compute average distribution
    m = 0.5 * (p + q)

    # Compute KL divergences
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))

    # Jensen-Shannon Divergence
    jsd = 0.5 * kl_pm + 0.5 * kl_qm

    return float(jsd)

