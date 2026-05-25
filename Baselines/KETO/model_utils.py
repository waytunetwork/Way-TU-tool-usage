import random
import numpy as np 

def adaptive_farthest_point_sampling(pcl, k):
    n, d = pcl.shape
    assert d == 4, "Input point cloud must have 4 columns (XYZ + Labels)"

    pc = pcl[:, :3]
    labels = pcl[:, 3]

    # Count unique labels and their point counts
    unique_labels, class_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_labels)

    # Calculate the target number of points per class
    points_per_class = k // num_classes
    remaining_points = k % num_classes  # Handle rounding issues

    sampled_points = []
    for label in unique_labels:
        class_mask = labels == label
        class_points = pc[class_mask]
        class_labels = labels[class_mask]

        # Assign remaining points to some classes to make total = k
        extra_point = 1 if remaining_points > 0 else 0
        num_samples = min(points_per_class + extra_point, len(class_points))  # Don't oversample
        remaining_points -= extra_point  # Reduce remaining points

        if len(class_points) > 0:
            sampled = farthest_point_sampling(np.column_stack((class_points, class_labels)), num_samples)
            sampled_points.append(sampled)

    # Merge results
    downsampled_pc_with_labels = np.vstack(sampled_points) if sampled_points else np.array([])

    return downsampled_pc_with_labels

def farthest_point_sampling(pcl, k):
    n,d = pcl.shape

    if d == 4: 
        pc = pcl[:, :3]
        labels = pcl[:, 3]
    else: 
        pc = pcl
    
    idx = random.randint(0, n - 1)
    downsampled_pc = [pc[idx]]
    
    if d == 4: 
        downsampled_labels = [labels[idx]]

    distances = np.ones(n) * np.inf
    
    for i in range(k-1): 
        last_added = downsampled_pc[-1]
        dist_to_last = np.linalg.norm(pc - last_added, axis =1)
        distances = np.minimum(distances, dist_to_last)
        farthest_idx = np.argmax(distances)

        downsampled_pc.append(pc[farthest_idx])
        if d ==4 :
            downsampled_labels.append(labels[farthest_idx])
    
    if d ==4 :
        downsampled_pc_with_labels = np.column_stack((downsampled_pc, downsampled_labels))
        return np.array(downsampled_pc_with_labels)
    else : 
        return np.array(downsampled_pc)