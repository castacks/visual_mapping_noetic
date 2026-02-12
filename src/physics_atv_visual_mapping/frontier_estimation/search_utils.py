import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from tqdm import tqdm
from PIL import Image
from natsort import natsorted

import skfmm
from scipy.spatial.transform import Rotation as R
import scipy
import heapq
import time

import numpy as np
from scipy.spatial.distance import mahalanobis

def sample_by_heading(V, start, headings, r):
    """
    Sample cost-to-reach radius r for a set of headings.

    Args:
        V (H, W): value function from fastmarch_radius
        start (2,): (row, col)
        headings (N,): angles in radians (0 = +x, pi/2 = +y)
        r (float): radius in pixels

    Returns:
        costs (N,): cost to reach radius r along each heading
        points (N, 2): sampled points
    """

    headings = np.asarray(headings)

    # Convert headings to grid coordinates
    points = np.stack([
        start[0] + r * np.sin(headings),
        start[1] + r * np.cos(headings)
    ], axis=1).astype(int)

    H, W = V.shape
    points[:, 0] = np.clip(points[:, 0], 0, H - 1)
    points[:, 1] = np.clip(points[:, 1], 0, W - 1)

    V_start = V[start[0], start[1]]
    # costs = V_start - V[points[:, 0], points[:, 1]]
    # costs = V[points[:, 0], points[:, 1]] - V_start
    costs = V[points[:, 0], points[:, 1]]

    return costs, points


def fastmarch(costmap, goal):
    # costmap[costmap < .55] = .01
    phi = np.ones_like(costmap)
    phi[:] = 1
    phi[goal[0],goal[1]] = -1   # boundary condition

    V = skfmm.travel_time(phi, speed=1.0 / (costmap + 1e-6))
    V[goal[0],goal[1]] = 0
    return V

def fastmarch_local(costmap, start, goal):
    H, W = costmap.shape
    pad = int(np.linalg.norm(start - goal)) + 1000

    x0 = max(0, start[0] - pad)
    x1 = min(H, start[0] + pad)
    y0 = max(0, start[1] - pad)
    y1 = min(W, start[1] + pad)

    sub = costmap[x0:x1, y0:y1]
    start_sub = start - np.array([x0, y0])
    goal_sub = goal - np.array([x0,y0])

    V_sub = fastmarch(sub, goal_sub)

    # Embed back into full map
    V = np.full_like(costmap, 1e5)
    V[x0:x1, y0:y1] = V_sub
    return V

def fastmarch_local_start(costmap, start, R):
    H, W = costmap.shape
    
    pad = 2*R

    x0 = max(0, start[0] - R - pad)
    x1 = min(H, start[0] + R + pad)
    y0 = max(0, start[1] - R - pad)
    y1 = min(W, start[1] + R + pad)

    sub = costmap[x0:x1, y0:y1]
    start_sub = start - np.array([x0, y0])

    V_sub = fastmarch(sub, start_sub)

    # Embed back into full map
    V = np.full_like(costmap, 1e5)
    V[x0:x1, y0:y1] = V_sub
    return V

def fastmarch_radius(costmap, start, R, num_goals=None, lethal = .5):
    """
    Fast marching value function for reaching radius R from start in any direction.

    Args:
        costmap (H, W): traversal cost map (higher = slower)
        start (2,): (row, col)
        R (float): radius in pixels
        num_goals (int, optional): number of boundary points on the circle

    Returns:
        V (H, W): value function (min cost to reach radius R)
    """

    H, W = costmap.shape

    if num_goals is None:
        # Dense enough to form a closed contour
        num_goals = max(16, int(2 * np.pi * R / 2))

    # Create circular boundary
    angles = np.linspace(0, 2 * np.pi, num_goals, endpoint=False)
    circle = np.stack([
        start[0] + R * np.sin(angles),
        start[1] + R * np.cos(angles)
    ], axis=1).astype(int)

    # Clamp to map bounds
    circle[:, 0] = np.clip(circle[:, 0], 0, H - 1)
    circle[:, 1] = np.clip(circle[:, 1], 0, W - 1)

    costs = costmap[circle[:,0], circle[:,1]]
    keep = costs < lethal
    circle = circle[keep]

    # Initialize level set
    phi = np.ones_like(costmap, dtype=float)
    phi[circle[:, 0], circle[:, 1]] = -1

    # Run fast marching inward
    V = skfmm.travel_time(phi, speed=1.0 / (costmap + 1e-6))

    # Ensure boundary condition is exactly zero
    V[circle[:, 0], circle[:, 1]] = 0

    return V

def fastmarch_radius_local(costmap, start, R, num_goals, lethal):
    H, W = costmap.shape
    pad = 2*R

    x0 = max(0, start[0] - R - pad)
    x1 = min(H, start[0] + R + pad)
    y0 = max(0, start[1] - R - pad)
    y1 = min(W, start[1] + R + pad)

    sub = costmap[x0:x1, y0:y1]
    start_sub = start - np.array([x0, y0])

    V_sub = fastmarch_radius(sub, start_sub, R, num_goals, lethal)

    # Embed back into full map
    V = np.full_like(costmap, np.inf)
    V[x0:x1, y0:y1] = V_sub
    return V


def extract_smooth_path(V, start, goal, step_size=1.0, max_steps=1000, grads = None):
    path = [start]
    y, x = start
    H, W = V.shape

    if grads is None:
        gy, gx = np.gradient(V)  # gy = dV/dy, gx = dV/dx
    else:
        gy, gx = grads

    success = False
    for step_num in range(max_steps):
        # bilinear interpolation of gradient
        y0, x0 = np.floor([y, x]).astype(int)
        y1, x1 = np.ceil([y, x]).astype(int)
        y1 = min(y1, H-1)
        x1 = min(x1, W-1)
        wy = y - y0
        wx = x - x0

        grad_y = (
            (1-wx)*(1-wy)*gy[y0,x0] + wx*(1-wy)*gy[y0,x1] +
            (1-wx)*wy*gy[y1,x0] + wx*wy*gy[y1,x1]
        )
        grad_x = (
            (1-wx)*(1-wy)*gx[y0,x0] + wx*(1-wy)*gx[y0,x1] +
            (1-wx)*wy*gx[y1,x0] + wx*wy*gx[y1,x1]
        )

        grad = np.array([grad_y, grad_x])
        norm = np.linalg.norm(grad)
        if norm < 1e-6:
            # print("Gradient too small, stopping")
            break

        dy, dx = -grad / norm * step_size
        y += dy
        x += dx
        path.append((y, x))

        # check if close enough to goal
        if np.hypot(y - goal[0], x - goal[1]) < step_size:
            path.append(goal)
            # print("REACHED")
            success = True
            break
    # print(f"Steps: {step_num+1}")
    return path, success

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra



def add_wall_behind(costmap, start, yaw, length=5, width=25, cost = 10000):
    """
    Adds a virtual obstacle behind the robot.
    """
    H, W = costmap.shape
    y0, x0 = start

    for d in range(1, length):
        for w in range(-width, width + 1):
            by = y0 - d * np.sin(yaw) + w * np.cos(yaw)
            bx = x0 - d * np.cos(yaw) - w * np.sin(yaw)
            iy, ix = int(round(by)), int(round(bx))
            if 0 <= iy < H and 0 <= ix < W:
                costmap[iy, ix] = cost  # obstacle

def add_limit_turn(costmap, start, yaw,
                                    radius=12.0,
                                    extent=30,
                                    cost=10000):
    """
    Enforces minimum turning radius with NO gaps.
    Blocks all states requiring curvature > 1/radius.
    """
    H, W = costmap.shape
    y0, x0 = start

    cy = np.sin(yaw)
    cx = np.cos(yaw)

    tradius = 3 + radius
    # Min-turn circle centers
    left_cy  = y0 + tradius * cx
    left_cx  = x0 - tradius * cy
    right_cy = y0 - tradius * cx
    right_cx = x0 + tradius * cy

    for d in range(-extent, extent + 1):
        for w in range(-extent, extent + 1):

            if d == 0 and w == 0:
                continue

            py = y0 + d * cy + w * cx
            px = x0 + d * cx - w * cy
            iy, ix = int(round(py)), int(round(px))

            if not (0 <= iy < H and 0 <= ix < W):
                continue

            # Distance to min-turn circles
            dl = np.hypot(py - left_cy,  px - left_cx)
            dr = np.hypot(py - right_cy, px - right_cx)

            # Inside either circle => illegal curvature
            if dl < radius or dr < radius:
                costmap[iy, ix] = cost


def get_plan(costmap, start, goal, yaw = 0, get_val = False, clear_idx = None,max_steps=1000):
    # V = fastmarch(costmap, goal)
    V = fastmarch_local(costmap,start, goal)
    path = extract_smooth_path(V, start, goal, max_steps)

    return np.array(path), V

def get_plan_together(costmap, start, goals, radius):
    # V = fastmarch(costmap, goal)

    V = fastmarch_local_start(costmap, start, radius+50)
    grads = np.gradient(V)
    now = time.perf_counter()


    
    all_paths = []
    for goal in goals:
        optpath, success = extract_smooth_path(V, goal, start, grads=grads)
        optpath = np.array(optpath)
        tpath = optpath.astype(int)
        pathcost = costmap[tpath[:,0], tpath[:,1]].mean()

        if success:
            all_paths.append(optpath)


    return V, all_paths

def main():
    DATA_DIR_BASE = '/home/tartandriver/workspace/datasets/lrdp'

    # K = np.array([455.7750, 0., 497.1180, 0., 456.3191, 251.8580, 0., 0., 1.]).reshape(3,3)
    K = np.array([4.776049499511718750e+02,
        0.000000000000000000e+00,
        4.995000000000000000e+02,
        0.000000000000000000e+00,
        4.776049499511718750e+02,
        2.520000000000000000e+02,
        0.000000000000000000e+00,
        0.000000000000000000e+00,
        1.000000000000000000e+00]).reshape(3,3)

    # plot_cost_map_and_heading(occ, start, thresh, headings, costs)
    # trainset6 = FrontierDataset(DATA_DIR_BASE + '/04_garage_to_turnpike_afternoon', augment=AUGMENT)
    # trainset = FrontierDataset(DATA_DIR_BASE + '/20251009_3_red_course', apply_footprint = False)

    trainset7 = FrontierDataset(DATA_DIR_BASE + '/2025-10-16-18-00-49_snow_bag', apply_footprint = False)
    trainset = FrontierDataset(DATA_DIR_BASE + '/2023-11-14-15-02-21_figure_8', apply_footprint = False)
    trainset2= FrontierDataset(DATA_DIR_BASE + '/turnpike_2023-09-12-12-53-32', apply_footprint = False)
    trainset3 = FrontierDataset(DATA_DIR_BASE + '/20251009_3_red_course', apply_footprint = False)
    trainset4 = FrontierDataset(DATA_DIR_BASE + '/20251009_4_fig8_to_horseshoe', apply_footprint = False)
    trainset5 = FrontierDataset(DATA_DIR_BASE + '/20251009_5_turnpike', apply_footprint = False)
   
    # datasets = [trainset, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7]
    datasets = [trainset2, trainset, trainset3, trainset4, trainset5, trainset7]
    # datasets = [trainset3, trainset4, trainset5, trainset7]

    # datasets = [trainset7]

    horizon = 600
    goal_dist = 500
    for ds in datasets:
        ds.horizon = horizon
    trainset_all = ConcatDataset(datasets)

    eval_loader = torch.utils.data.DataLoader(
        trainset_all, batch_size=1, shuffle=True, drop_last=True
    )

    RADIUS = 200  # desired radius
    NUM_GOALS = 120  # number of goals around the circle
    SUBRADIUS = 50

    count = 0

    lethal_thresh = .55
    lethal_add = 100.
    costmap = trainset.costmap.copy()
    # costmap[costmap < .5] = 0
    costmap[costmap >= lethal_thresh] += lethal_add
    costmap += 1.

    # costmap = trainset.costmap.copy()
    # lethal_add = 1.5
    # costmap += 1.
    # costmap *= 2
    # lethal_add *= 2

    # costmap *= 0
    # costmap += 2.

    for i, data in enumerate(eval_loader):
        count += 1
        # if i > 10:
            # break

        start = data['origin_idx'][0].cpu().numpy().astype(int)
        demo_idx = data['og_demo_idx'][0].cpu().numpy()
        
        yaw = data['origin'][0][2]
        yaw += np.pi/2
        rot = R.from_euler('z', yaw)
        rot_inv = rot.inv()

        # Create goals around the start at radius RADIUS
        angles = np.linspace(0, 2*np.pi, NUM_GOALS, endpoint=False)
        goals = np.array([start + RADIUS * np.array([np.sin(a), np.cos(a)]) for a in angles]).astype(int)

        costs = costmap[goals[:,0], goals[:,1]]
        keep = costs < lethal_add
        goals = goals[keep]

        all_paths = []
        all_costs = []
        all_goals = []

        now = time.perf_counter()
        # V = fastmarch_radius(costmap, start,RADIUS, num_goals = NUM_GOALS,lethal=1000)
        # V = fastmarch_radius_local(costmap, start,RADIUS, num_goals = NUM_GOALS,lethal=lethal_add)
        # plt.imshow(V, vmin= 0, vmax = RADIUS*2)
        # plt.show()
        # V_start = fastmarch(costmap, start)

        # fig, ax =  plt.subplots(1,2)
        # ax[0].imshow(V, vmin= 0, vmax = RADIUS*2)
        # ax[1].imshow(V_start)
        # plt.show()
        # import pdb;pdb.set_trace()

        # V = V + V_start
        # plt.imshow(V)
        # plt.show()
        
        # V -= V[start[0], start[1]]
        # V /= RADIUS

        # print(time.perf_counter() - now)
        # vals = []
        # for i in range(30):
        #     nstart = start + np.random.randint(-50,50,2)
        #     vals.append(V[nstart[0],nstart[1]])
        #     path = extract_smooth_path(V, nstart, (0,0), max_steps=300)
        #     all_paths.append(np.array(path))
        #     all_goals.append(nstart)

        # for goal in goals:
        #     # now=  time.perf_counter()
        #     optpath, V = get_plan(
        #         trainset.costmap, start, goal, yaw=yaw, get_val=True, clear_idx=demo_idx.astype(int)
        #     )
        #     tpath = optpath.astype(int)
        #     pathcost = costmap[tpath[:,0], tpath[:,1]].sum()
        #     # print(time.perf_counter() - now)
        #     # if pathcost < 1000:
        #     all_paths.append(optpath)
        #     all_goals.append(goal)
        #     all_costs.append(pathcost)
        #     # all_paths.append(goal)


        V, all_paths = get_plan_together(trainset.costmap, start, goals, RADIUS)
        all_costs = []
        for path in all_paths:
            tpath = path.astype(int)
            print(tpath.shape)
            cost = costmap[tpath[:,0], tpath[:,1]].sum()
            all_costs.append(cost)
        print(time.perf_counter() - now)
        # print(all_costs)
        all_costs = np.array(all_costs)

        # Plot all paths together
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        
        ax[0].imshow(V, vmin= 0, vmax = RADIUS*2)
        ax[0].scatter(start[1], start[0], c='red', label='Start')
        # for goal, path, cost in zip(all_goals, all_paths, all_costs):
        #     # ax[0].scatter(goal[1], goal[0], c='green')
        #     ax[0].scatter(goal[1], goal[0], c=cost)
            # ax[0].plot(path[:,1], path[:,0])
        ax[0].set_title('Value Function with All Paths')

        ax[1].imshow(costmap)
        ax[1].scatter(start[1], start[0], c='red', label='Start')
        for j, path in enumerate(all_paths):
            # ax[1].scatter(goal[1], goal[0], c='green')
            cost = all_costs[j]
            if cost > 3000:
                c = '-r'
            else:
                c = '-g'
            ax[1].plot(path[:,1], path[:,0], c)


        plt.show()


if __name__ == "__main__":
    main()




