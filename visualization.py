from matplotlib import pyplot as plt
import numpy as np

def initialize_viz(num_plots=1):
    """
    Creates the objects needed to visualize the data
    """
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    axs = []
    if num_plots == 1:
        ax = fig.add_subplot(111, projection='3d')
        axs.append(ax)
    elif num_plots == 2:
        ax = fig.add_subplot(121, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(122, projection='3d')
        axs.append(ax)
    elif num_plots == 3:
        ax = fig.add_subplot(131, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(132, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(133, projection='3d')
        axs.append(ax)
    elif num_plots == 4:
        ax = fig.add_subplot(221, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(222, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(223, projection='3d')
        axs.append(ax)
        ax = fig.add_subplot(224, projection='3d')
        axs.append(ax)
    else:
        raise ValueError(f"Invalid number of plots: {num_plots}")

    return fig, axs

def plot_coordinate_frame(viz_obj, T, axis_length=1.0, use_labels=True, is_camera=False, plot_num=0):
    """
    Plots a coordinate frame in a 3D plot.

    Parameters:
    ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D subplot to plot on.
    T (numpy.ndarray): 4x4 transformation matrix.
    axis_length (float): Length of the frame axes.
    """
    ax = viz_obj[1][plot_num]
    origin = T[:3,3]
    rotation_matrix = T[:3,:3]

    # Unit vectors for X, Y, Z
    unit_vectors = np.identity(3)

    # Transform unit vectors using the rotation matrix
    transformed_vectors = rotation_matrix.dot(unit_vectors) * axis_length

    # Plotting each axis
    colors = ['r', 'g', 'b'] if not is_camera else ['r', 'g', 'k']
    for i, (color, label) in enumerate(zip(colors, "XYZ")):
        # Compute the end point of the axis
        end_point = origin + transformed_vectors[:, i]
        ax.plot([origin[0], end_point[0]], [origin[1], end_point[1]], [origin[2], end_point[2]], color=color, label=f'{label}-axis' if use_labels else None)
    
    if use_labels:
        ax.legend()

def plot_corners(viz_obj, corners_mm, plot_num=0):
    """
    Plots the corners of a tag in a 3D plot. Puts lines between the corners to show the edges of the tag.
    """
    ax = viz_obj[1][plot_num]
    # Plot the corners
    ax.scatter(corners_mm[:,0], corners_mm[:,1], corners_mm[:,2], color='k')
    # Plot the edges
    for i in range(4):
        ax.plot([corners_mm[i,0], corners_mm[(i+1)%4,0]], [corners_mm[i,1], corners_mm[(i+1)%4,1]], [corners_mm[i,2], corners_mm[(i+1)%4,2]], color='k')

def set_aspect_ratio(viz_obj, Ts, plot_num=0):
    """
    Sets the aspect ratio of the 3D plot to be equal in all dimensions.
    """
    origins = np.array([T[:3,3] for T in Ts])
    xs, ys, zs = origins[:,0], origins[:,1], origins[:,2]
    ax = viz_obj[1][plot_num]
    # ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    # Find the max range over all axes
    max_range = np.array([np.ptp(xs), np.ptp(ys), np.ptp(zs)]).max()
    # Get the mid points over all axes
    mid_x = np.mean(xs)
    mid_y = np.mean(ys)
    mid_z = np.mean(zs)
    # Set the limits to be symmetric
    ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])

    return max_range