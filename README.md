<!-- Module Description -->
# physics_atv_visual_mapping

Preliminary repo for BEV mapping of visual features. For the time being, this will be in Python for prototyping and stuff.

***

<!-- Technical Approach -->
## Technical Approach

High level approach will be the following:

    1. Run image through some visual encoder (e.g. SAM, Dino, semantics, passthrough) to get FPV image-space embeddings

    2. Project latest pointcloud into image space and tag points with their corresponding visual embedding

    3. Map the points (and their embeddings) into BEV space

    4. Aggregate BEV maps over time with odometry

<!-- Requirements -->
## Requirements

1. Camera intrinsics
2. Camera to lidar extrinsics
3. (good) odometry

<!-- Outputs -->
## Outputs

1. Embedding image
2. ```Gridmap``` of instantaneous features
3. ```Gridmap``` of aggregated features