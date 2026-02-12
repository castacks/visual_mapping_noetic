import numpy as np
import matplotlib.pyplot as plt
import cv2

# def overlay_heatmap_on_image(image, heatmap, max_val = 1., threshold=0.01, alpha=0.6, cmap='jet'):

#     H, W = image.shape[:2]
#     heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

#     # Normalize heatmap to [0, 1]
#     # hm_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
#     hm_norm = np.clip(heatmap/max_val, 0, 1.)
#     # Apply matplotlib colormap (returns RGBA)
#     # cmap_fn = cm.get_cmap(cmap)
#     hm_color = CMAP_JET(hm_norm)[..., :3]  # drop alpha channel

#     # Create binary mask where heatmap exceeds threshold
#     mask = (hm_norm > threshold)[..., None].astype(float)

#     # Convert image to float in [0,1] if needed
#     if image.dtype == np.uint8:
#         img_float = image.astype(np.float32) / 255.0
#     else:
#         img_float = image.copy()

#     # Blend where mask is active
#     overlay = img_float * (1 - alpha * mask) + hm_color * (alpha * mask)

#     # Convert back to uint8
#     overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
#     return overlay

def overlay_heatmap_on_image(
    image,
    heatmap,
    max_val=3.0,
    threshold=0.01,
    alpha=0.6,
):
    H, W = heatmap.shape[:2]

    # Resize heatmap
    # heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_NEAREST)
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
    # Normalize to [0, 255]
    hm = np.clip(heatmap / max_val, 0, 1)
    hm_u8 = (hm * 255).astype(np.uint8)

    # Apply OpenCV colormap (BGR!)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)

    # Threshold mask
    mask = hm > threshold

    # Ensure image uint8
    if image.dtype != np.uint8:
        img_u8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        img_u8 = image

    # Alpha blend only where mask is true
    overlay = img_u8.copy()
    overlay[mask] = (
        (1 - alpha) * img_u8[mask] +
        alpha * hm_color[mask]
    ).astype(np.uint8)

    return overlay

def project_points_rectified(
    xyz_robot,                # (N,3) points in robot frame
    image,           # optional (H, W) to mask to image bounds
    color = (0,0,255),
    thickness=2,
    alpha = .4
):

    extrinsics = torch.Tensor([[ 0.0033, -0.9997,  0.0236,  0.1726],
        [-0.2247, -0.0237, -0.9741, -0.1523],
        [ 0.9744, -0.0021, -0.2247,  0.0571],
        [ 0.0000,  0.0000,  0.0000,  1.0000]])
    
    intrinsics = torch.Tensor([[455.7750,   0.0000, 497.1180,   0.0000],
        [  0.0000, 456.3191, 251.8580,   0.0000],
        [  0.0000,   0.0000,   1.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   1.0000]])
    
    P = get_projection_matrix(intrinsics, extrinsics).to('cpu')

    img = torch.from_numpy(image)
    xyz_robot = torch.from_numpy(xyz_robot)
    footprint_pcl_px_in_frame, ind_in_frame = get_pixel_projection(xyz_robot, P.unsqueeze(0),img.unsqueeze(0))
    footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[0]
    ind_in_frame = ind_in_frame[0]

    footprint_pcl_px_in_frame = footprint_pcl_px_in_frame[ind_in_frame]
    # print(footprint_pcl_px_in_frame.shape)
    # traj_mask_px = torch.unique(footprint_pcl_px_in_frame.long(), dim=0)
    # print(traj_mask_px.shape)
    overlay = image.copy()
    traj_mask_px = footprint_pcl_px_in_frame.long().numpy()
    plot_color = tuple(int(x) for x in color)
    for i in range(len(traj_mask_px)-1):
        pt1 = tuple(traj_mask_px[i])
        pt2 = tuple(traj_mask_px[i+1])
        cv2.line(overlay, pt1, pt2, color=plot_color, thickness=thickness)

    image = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0.)
    return image


def plot_cost_map_and_heading_image(cost_map, start, threshold, headings, costs, img, max_val = 5., sname=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    # Normalize heading costs
    # normed = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-9)

    forward_mask = (np.cos(headings) >= 0)  # same as abs(heading) <= π/2
    forward_costs = costs[forward_mask]

    if np.any(np.isfinite(forward_costs)):
        min_c = np.nanmin(forward_costs)
        max_c = np.nanmax(forward_costs)
    else:
        # fallback if all NaNs or inf
        min_c, max_c = np.nanmin(costs), np.nanmax(costs)

    # Normalize all headings using forward range
    # normed = (costs - min_c) / (max_c - min_c + 1e-9)
    normed = costs/max_val
    normed = np.clip(normed, 0, 1)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    # Cost map
    ax0.imshow(cost_map, cmap='viridis')
    ax0.contour(cost_map > threshold, levels=[0.5], colors='red', linewidths=1)
    ax0.plot(start[1], start[0], 'wo', markersize=8, markeredgecolor='k')
    ax0.set_title("Cost Map with Lethal Regions (red)")
    viz_range = 250
    ax0.set_xlim(start[1] - viz_range, start[1] + viz_range)
    ax0.set_ylim(start[0] + viz_range, start[0] - viz_range)
    # arrow_length = 55.0  # meters, or scale by score if you want
    # points = []
    # for h, s in zip(headings, costs):
    #     dx = np.cos(h) * arrow_length
    #     dy = np.sin(h) * arrow_length
    #     dz = 0.0  # assume points are at ground level
    #     points.append([dx, dy, dz])
    # points = np.array(points)  # shape (K,3)
    # points[:, 2] -= 1.7  # camera height
    # # print(points)
    

    # origin = np.array([[0,0,-1.7]])  # camera center in local frame
    # drawn = img.copy()
    # for p, score in zip(points, normed):
    #     path_color = CMAP(score)
    #     drawn = project_points_rectified(np.vstack([origin, p]).astype(np.float32), drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    num_pts = 80  # how smooth the projected line should look
    origin = np.array([0, 0, -1.7])  # camera origin in local frame
    drawn = img.copy()
    for h, s in zip(headings, normed):
        # Length of arrow can be fixed or proportional to score
        arrow_length = 10.0 * (s / np.max(normed))  # meters
        end = np.array([np.cos(h)*arrow_length, np.sin(h)*arrow_length, -1.7])
        
        # Interpolate between origin and end
        t = np.linspace(0, 1, num_pts)
        line_pts = origin + np.outer(t, end - origin)  # (num_pts, 3)
        line_pts = line_pts.astype(np.float32)

        path_color = CMAP(s)
        # Project and draw
        drawn = project_points_rectified(line_pts, drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    # Image
    ax1.imshow(drawn)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Polar heatmap
    ax2.bar(headings, np.ones_like(costs), width=2*np.pi/len(costs),
            color=plt.cm.inferno(normed), edgecolor='none')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(1)
    ax2.set_title("Heading Cost Heatmap (higher = better)")

    plt.tight_layout()
    if sname is not None:
        plt.savefig(sname + ".png", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close("all")
    else:
        plt.show()

def plot_heatmap_and_heading_image(heatmap, threshold, headings, costs, img, sname=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec

    heatmap_viz = heatmap.copy()
    heatmap_viz[heatmap< -6.0] = 0
    overlay = overlay_heatmap_on_image(img, heatmap_viz)
    # plt.imshow(overlay)
    # plt.show()

    # Normalize heading costs
    # normed = (costs - np.nanmin(costs)) / (np.nanmax(costs) - np.nanmin(costs) + 1e-9)

    forward_mask = (np.cos(headings) >= 0)  # same as abs(heading) <= π/2
    forward_costs = costs[forward_mask]

    if np.any(np.isfinite(forward_costs)):
        min_c = np.nanmin(forward_costs)
        max_c = np.nanmax(forward_costs)
    else:
        # fallback if all NaNs or inf
        min_c, max_c = np.nanmin(costs), np.nanmax(costs)

    # Normalize all headings using forward range
    normed = (costs - min_c) / (max_c - min_c + 1e-9)
    # normed = np.clip(costs, 0, 1)

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')

    # Cost map
    ax0.imshow(overlay, cmap='viridis')
    ax0.set_title("Affordance Heatmap")
    ax0.axis('off')
    
    num_pts = 80  # how smooth the projected line should look
    origin = np.array([0, 0, -1.7])  # camera origin in local frame
    drawn = img.copy()
    for h, s in zip(headings, normed):
        # Length of arrow can be fixed or proportional to score
        arrow_length = 10.0 * (s / np.max(normed))  # meters
        end = np.array([np.cos(h)*arrow_length, np.sin(h)*arrow_length, -1.7])
        
        # Interpolate between origin and end
        t = np.linspace(0, 1, num_pts)
        line_pts = origin + np.outer(t, end - origin)  # (num_pts, 3)
        line_pts = line_pts.astype(np.float32)

        path_color = CMAP(s)
        # Project and draw
        drawn = project_points_rectified(line_pts, drawn, np.array(path_color[:3])*255, thickness=4, alpha=1.)

    # Image
    ax1.imshow(drawn)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Polar heatmap
    ax2.bar(headings, np.ones_like(costs), width=2*np.pi/len(costs),
            color=plt.cm.inferno(normed), edgecolor='none')
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(1)
    ax2.set_title("Heading Cost Heatmap (higher = better)")

    plt.tight_layout()
    if sname is not None:
        # plt.savefig(sname + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(sname + ".jpg", dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close("all")
    else:
        plt.show()