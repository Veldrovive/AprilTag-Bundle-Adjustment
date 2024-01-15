"""
Loads a set of images containing AprilTags and estimates the pose of each tag and camera in the world frame.
Uses an incremental strategy to optimize the camera and tag poses for each image before moving on to the next.
"""

import cv2
import numpy as np
from pathlib import Path
from pupil_apriltags import Detector
import time

from pose_estimation import estimate_camera_pose, estimate_tag_pose, get_tag_corners_mm, bundle_adjust, find_jacobian, multi_tag_estimate_camera_pose, find_corner_px_projection
from visualization import initialize_viz, plot_coordinate_frame, plot_corners, set_aspect_ratio

detector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)

Pose = np.ndarray  # 4x4 homogeneous transformation matrix
PxPos = np.ndarray  # 2x1 pixel position

def load_calibration_data(cal_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the camera calibration data from the given file.

    Parameters:
    cal_file (Path): Path to the calibration data file.

    Returns:
    tuple[np.ndarray, np.ndarray]: Tuple containing the camera matrix and distortion coefficients.
    """
    cal_data = np.load(cal_file.absolute().as_posix())
    camera_matrix = cal_data['mtx']
    dist_coeffs = cal_data['dist']
    return camera_matrix, dist_coeffs

def get_corner_config(tag_sizes: list[float], tag_config: str, use_center: bool = True) -> list[list[int]]:
    """
    Gets an ndarray of the corner indices for each tag in the given tag configuration which matches the pixel ordering of the AprilTag library.
    """
    tag_corners = []
    if tag_config == "single":
        tag_size = tag_sizes[0]
        tag_corners.extend([
            [-tag_size/2, -tag_size/2, 0],
            [tag_size/2, -tag_size/2, 0],
            [tag_size/2, tag_size/2, 0],
            [-tag_size/2, tag_size/2, 0],
        ])
        if use_center:
            tag_corners.append([0, 0, 0])
    elif tag_config == "double":
        tag_1_size = tag_sizes[0]
        # Add the first tag
        tag_corners.extend([
            [-tag_1_size/2, -tag_1_size/2, 0],
            [tag_1_size/2, -tag_1_size/2, 0],
            [tag_1_size/2, tag_1_size/2, 0],
            [-tag_1_size/2, tag_1_size/2, 0],
        ])
        if use_center:
            tag_corners.append([0, 0, 0])
        # Add the second tag
        tag_2_size = tag_sizes[1]
        tag_2_center_x = tag_1_size / 2 - tag_2_size/2
        tag_spacing = tag_1_size / 8
        tag_2_center_y = tag_1_size / 2 + tag_spacing + tag_2_size/2
        tag_corners.extend([
            [tag_2_center_x - tag_2_size/2, tag_2_center_y - tag_2_size/2, 0],
            [tag_2_center_x + tag_2_size/2, tag_2_center_y - tag_2_size/2, 0],
            [tag_2_center_x + tag_2_size/2, tag_2_center_y + tag_2_size/2, 0],
            [tag_2_center_x - tag_2_size/2, tag_2_center_y + tag_2_size/2, 0],
        ])
        if use_center:
            tag_corners.append([tag_2_center_x, tag_2_center_y, 0])
    else:
        raise Exception(f"Unknown tag config: {tag_config}")

    return np.array(tag_corners)

def estimate_poses(img_files: list[Path], camera_matrix: np.ndarray, dist_coeffs: np.ndarray, tag_sizes: list[float], tag_config: str, out_dir: Path, base_frame_tag_id=1, use_scipy=True, show_raw=False) -> dict[int, Pose]:
    tag_corner_3d_positions = get_corner_config(tag_sizes, tag_config, use_center=True)

    tag_px_positions: dict[int, list[PxPos | None]] = {}  # Maps from tag ID to position. If the tag does not appear in the image, the position is None.
    tag_visibility_map: list[set[int]] = []  # Maps from image index to a set of tag IDs that are visible in that image

    all_tag_ids = set()
    for img_file in img_files:
        print(f"Loading image {img_file}...")
        # Load the image
        image = cv2.imread(img_file.absolute().as_posix())
        # Needs to be in grayscale for the apriltag detector
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect AprilTags in the image
        tags = detector.detect(gray)
        # Sort by tag.tag_id
        tags.sort(key=lambda tag: tag.tag_id)
        if tag_config == "single":
            # Add the tag IDs to the set of all tag IDs
            all_tag_ids.update([tag.tag_id for tag in tags])
            # Add the tags to the list of tags for this image
            tag_visibility_map.append(set([tag.tag_id for tag in tags]))
            # Add the tag positions to the list of tag positions for this image
            un_seen_tag_ids = all_tag_ids - set([tag.tag_id for tag in tags])
            for tag in tags:
                tag_id = tag.tag_id
                tag_corners_px = tag.corners
                # Features are the corners and the center
                tag_features_px = np.vstack((tag_corners_px, tag.center))
                tag_px_positions.setdefault(tag_id, [None] * (len(tag_visibility_map) - 1)).append(tag_features_px)
            for tag_id in un_seen_tag_ids:
                tag_px_positions.setdefault(tag_id, []).append(None)

            # Draw dots at each corner of each detected AprilTag and save the image
            for tag in tags:
                for corner in tag.corners:
                    cv2.circle(image, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
                # Add the center of the tag to the image
                cv2.circle(image, tuple(tag.center.astype(int)), 10, (0, 0, 255), -1)
            cv2.imwrite((out_dir / f"{img_file.stem}_with_dots.jpg").absolute().as_posix(), image)
        if tag_config == "double":
            # Then we will be collecting tags into groups of two
            # Tag IDs for a single group will be 2n and 2n+1
            tag_groups = {}
            # For each even numbered tag id we find, we will check if 2n+1 is also visible. If not, we just move on.
            for tag_idx, tag in enumerate(tags):
                tag_id = tag.tag_id
                if tag_id % 2 == 0:
                    # Then this is an even numbered tag
                    # Check if the next tag is also visible
                    if tag_idx + 1 < len(tags) and tags[tag_idx+1].tag_id == tag_id + 1:
                        # Then we have a group. We rename this group to be tag_id / 2
                        tag_groups[tag_id // 2] = [tag, tags[tag_idx+1]]
            # Add the tag IDs to the set of all tag IDs
            all_tag_ids.update(tag_groups.keys())
            # Add the tags to the list of tags for this image
            tag_visibility_map.append(set(tag_groups.keys()))
            # Add the tag positions to the list of tag positions for this image
            un_seen_tag_ids = all_tag_ids - set(tag_groups.keys())
            for tag_id in tag_groups:
                center_tag = tag_groups[tag_id][0]
                upper_tag = tag_groups[tag_id][1]
                # Features are the corners and the center. Center tag is specified first.
                tag_features_px = np.vstack((center_tag.corners, center_tag.center, upper_tag.corners, upper_tag.center))
                tag_px_positions.setdefault(tag_id, [None] * (len(tag_visibility_map) - 1)).append(tag_features_px)
            for tag_id in un_seen_tag_ids:
                tag_px_positions.setdefault(tag_id, []).append(None)

            # Draw dots at each corner of each detected AprilTag and save the image
            for tag in tags:
                for corner in tag.corners:
                    cv2.circle(image, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
                # Add the center of the tag to the image
                cv2.circle(image, tuple(tag.center.astype(int)), 10, (0, 0, 255), -1)
            cv2.imwrite((out_dir / f"{img_file.stem}_with_dots.jpg").absolute().as_posix(), image)

    initial_tag_poses: dict[int, list[Pose | None]] = {
        tag_id: [None] * len(img_files) for tag_id in all_tag_ids
    }  # Maps from tag ID to a list of poses for each image. If the tag does not appear in the image, the pose is None.

    refined_tag_poses: dict[int, Pose] = {
        base_frame_tag_id: np.eye(4)  # The base frame is the world frame
    }  # Maps from tag ID to the refined pose of the tag in the world frame
    refined_camera_poses: list[Pose] = [None] * len(img_files)  # List of refined camera poses for each image

    remaining_images = set(range(len(img_files)))  # Set of indices of images that have not been processed yet
    processed_image_mask = [False] * len(img_files)  # List of booleans indicating whether an image has been processed yet

    def find_next_additional_image(first_image: bool) -> int:
        """
        Uses a heuristic to choose the next image to process.
        If this is the first image, then we choose the image with the most visible tags that contains tag ID 1 which is our base frame.
        Otherwise, we choose the image with the most overlap with the tags that have already been processed.
        This heuristic should ensure that we have the best possible initial estimate for the poses of the tags and camera before moving onto a new tag.
        """
        if first_image:
            # Find the image with the most visible tags that contains the base frame tag ID
            max_num_visible_tags = 0
            max_num_visible_tags_img = None
            for img_idx, visible_tag_ids in enumerate(tag_visibility_map):
                if base_frame_tag_id in visible_tag_ids and len(visible_tag_ids) > max_num_visible_tags:
                    max_num_visible_tags = len(visible_tag_ids)
                    max_num_visible_tags_img = img_idx
            assert max_num_visible_tags_img is not None, f"No image contains tag ID {base_frame_tag_id}"
            return max_num_visible_tags_img
        else:
            # Find the image with the most overlap with the tags that have already been processed
            max_num_overlap = 0
            max_num_overlap_img = None
            for img_idx, visible_tag_ids in enumerate(tag_visibility_map):
                if processed_image_mask[img_idx]:
                    continue
                overlap = len(visible_tag_ids & set(refined_tag_poses.keys()))
                if overlap > max_num_overlap:
                    max_num_overlap = overlap
                    max_num_overlap_img = img_idx
            assert max_num_overlap_img is not None, "No image has any overlap with the tags that have already been processed"
            return max_num_overlap_img

    def visualize_curr_state(refined=True, raw=False):
        """
        Visualizes the current refined tag and camera poses.
        """
        num_plots = 0
        if refined:
            num_plots += 1
        if raw:
            num_plots += 1
        assert num_plots > 0, "Must choose at least one plot to display."
        viz_obj = initialize_viz(num_plots)
        # Get the tag poses
        plot_number = 0
        if refined:
            all_tag_poses = list(refined_tag_poses.values())
            # Get the camera poses
            all_camera_poses = [T for T in refined_camera_poses if T is not None]
            Ts = all_camera_poses + all_tag_poses
            max_range = set_aspect_ratio(viz_obj, Ts, plot_num=plot_number)
            use_labels = [False] * len(all_camera_poses) + [False] * (len(all_tag_poses)-1) + [True]
            is_camera = [True] * len(all_camera_poses) + [False] * len(all_tag_poses)
            for T, use_label, is_camera in zip(Ts, use_labels, is_camera):
                plot_coordinate_frame(viz_obj, T, axis_length=max_range/10, use_labels=use_label, is_camera=is_camera, plot_num=plot_number)
                if not is_camera:
                    corners = get_tag_corners_mm(T, tag_corner_3d_positions)
                    plot_corners(viz_obj, corners, plot_num=plot_number)
            plot_number += 1
        if raw:
            # Then we use the initial tag poses
            all_tag_poses = [T for tag_id, tag_poses in initial_tag_poses.items() for T in tag_poses if T is not None]
            # Get the camera poses
            all_camera_poses = [T for T in refined_camera_poses if T is not None]
            Ts = all_camera_poses + all_tag_poses
            max_range = set_aspect_ratio(viz_obj, Ts, plot_num=plot_number)
            use_labels = [False] * len(all_camera_poses) + [False] * (len(all_tag_poses)-1) + [True]
            is_camera = [True] * len(all_camera_poses) + [False] * len(all_tag_poses)
            for T, use_label, is_camera in zip(Ts, use_labels, is_camera):
                plot_coordinate_frame(viz_obj, T, axis_length=max_range/10, use_labels=use_label, is_camera=is_camera, plot_num=plot_number)
                if not is_camera:
                    corners = get_tag_corners_mm(T, tag_corner_3d_positions)
                    plot_corners(viz_obj, corners, plot_num=plot_number)
            plot_number += 1
        # Display the plot
        print("Displaying plot...")
        viz_obj[0].show()
        input("Press Enter to continue...")

    bundle_start_time = time.time()
    while len(remaining_images) > 0:
        # Choose the next image to process
        next_img_idx = find_next_additional_image(sum(processed_image_mask) == 0)
        # and remove it from the set of remaining images
        remaining_images.remove(next_img_idx)
        print(f"Processing image {next_img_idx}: {img_files[next_img_idx]}...")
        print(f"\tNumber of remaining images: {len(remaining_images)}.")
        print(f"\tSeconds elapsed: {time.time() - bundle_start_time}.")
        print(f"\tContains tags {tag_visibility_map[next_img_idx]}.")
        
        # Step 1: Estimate the camera pose
        # We do this by finding all the tags that are visible in this image and have a known pose
        # Then we solve a PnP problem to estimate the camera pose
        visible_and_known_tag_ids = tag_visibility_map[next_img_idx] & set(refined_tag_poses.keys())
        # Build the dicts used in the multi_tag_estimate_camera_pose function
        curr_img_tag_poses = {tag_id: refined_tag_poses[tag_id] for tag_id in visible_and_known_tag_ids}
        curr_img_tag_px_positions = {tag_id: tag_px_positions[tag_id][next_img_idx] for tag_id in visible_and_known_tag_ids}
        # Estimate the camera pose
        T_1_Ci = multi_tag_estimate_camera_pose(curr_img_tag_poses, curr_img_tag_px_positions, tag_corner_3d_positions, camera_matrix, dist_coeffs)
        # Add the camera pose to the list of refined camera poses
        refined_camera_poses[next_img_idx] = T_1_Ci
        
        # Step 2: Estimate the pose of all tags in the image
        visible_tags = tag_visibility_map[next_img_idx]
        for tag_id in visible_tags:
            tag_corners_px = tag_px_positions[tag_id][next_img_idx]
            T_1_Ai = estimate_tag_pose(tag_corners_px, tag_corner_3d_positions, camera_matrix, dist_coeffs, T_1_Ci)
            initial_tag_poses[tag_id][next_img_idx] = T_1_Ai
            # If this is the first time we have seen this tag, then add it to the list of refined tag poses
            if tag_id not in refined_tag_poses:
                refined_tag_poses[tag_id] = T_1_Ai
        
        # Step 3: Refine the camera and tag poses jointly using bundle adjustment
        processed_image_mask[next_img_idx] = True
        # Construct the data that is required for the bundle adjustment
        # The bundle adjustment function expects us to have all the tags and camera poses, but we only have the ones that have been processed so far
        # Therefore, we need to construct a new tag_px_positions and refined_camera_poses that only have the currently processed images
        # We don't actually need to worry about the tags because it is constructed to use whatever tag ids are in refined_tag_poses
        # We will remove the tags that have not been processed yet from our new tag_px_positions though to be thorough
        curr_tag_px_positions = {}
        curr_camera_estimates = []
        curr_tag_poses = refined_tag_poses
        curr_camera_indices = [i for i, processed in enumerate(processed_image_mask) if processed]

        for tag_id in curr_tag_poses:
            curr_tag_px_positions[tag_id] = [tag_px_positions[tag_id][i] for i in curr_camera_indices]
        for i in curr_camera_indices:
            curr_camera_estimates.append(refined_camera_poses[i])
        curr_refined_tag_poses, curr_refined_camera_poses = bundle_adjust(
            tag_px_positions_0=curr_tag_px_positions,
            tag_poses_0=curr_tag_poses,
            camera_poses_0=curr_camera_estimates,
            tag_corners_mm_Ai=tag_corner_3d_positions,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs, 
            base_frame_tag_id=base_frame_tag_id,  
            use_scipy=use_scipy
        )
        # Update the refined tag and camera poses
        refined_tag_poses = curr_refined_tag_poses
        for i, camera_pose in zip(curr_camera_indices, curr_refined_camera_poses):
            refined_camera_poses[i] = camera_pose

    # Now we re-do the dots on the images, but this time with the expected corner position derived using find_corner_px_projection
    for img_idx, img_file in enumerate(img_files):
        print(f"visualizing image {img_idx}: {img_file}...")
        # Load the image
        image = cv2.imread(img_file.absolute().as_posix())
        # Get the camera pose
        T_1_Ci = refined_camera_poses[img_idx]
        # For each tag that is visible in this image, draw the dots
        for tag_id, T_1_Ai in refined_tag_poses.items():
            # Also draw the dots for the ground truth tags
            gt_corners_px = tag_px_positions[tag_id][img_idx]
            if gt_corners_px is not None:
                for corner in gt_corners_px:
                    cv2.circle(image, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
            
            # And draw the dots for the estimated tags
            corners_px = find_corner_px_projection(T_1_Ci, T_1_Ai, tag_corner_3d_positions, camera_matrix, dist_coeffs)
            for corner in corners_px:
                # Check if the corner is within the image bounds
                if corner[0] < 0 or corner[0] >= image.shape[1] or corner[1] < 0 or corner[1] >= image.shape[0]:
                    continue
                cv2.circle(image, tuple(corner.astype(int)), 10, (255, 0, 0), -1)
        # Save the image
        cv2.imwrite((out_dir / f"{img_file.stem}_with_dots.jpg").absolute().as_posix(), image)

    # Visualize the final state
    visualize_curr_state(refined=True, raw=show_raw)

    return refined_tag_poses, refined_camera_poses

        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Estimates the poses of AprilTags in a set of images.")
    parser.add_argument("--img-dir", type=str, help="Path to the directory containing the images.", required=True)
    parser.add_argument("--file-type", type=str, help="File type of the images.", choices=["jpg", "png"], default="jpg", required=False)
    parser.add_argument("--cal-file", type=str, help="Path to the camera calibration data file.", required=True)
    parser.add_argument("--tag-size", type=float, help="Size of the tags in mm.", default=None, required=False)
    parser.add_argument("--tag-sizes", type=float, nargs="+", help="Size of the tags in mm.", default=None, required=False)
    parser.add_argument("--out-dir", type=str, help="Path to the directory to save the output files.", required=True)
    parser.add_argument("--tag-config", type=str, help="The type of tag expected", choices=["single", "double"], required=True)
    parser.add_argument("--base-frame-tag-id", type=int, help="The tag ID of the base frame.", default=1, required=False)
    parser.add_argument("--use-gauss-newton", action="store_true", help="Use a manually implemented Gauss-Newton optimization (Don't use this).", required=False)
    parser.add_argument("--show-raw", action="store_true", help="Show the raw tag poses along with the refined ones.", required=False)
    args = parser.parse_args()

    # Get the image file paths
    img_dir = Path(args.img_dir)
    assert img_dir.is_dir(), f"{args.img_dir} is not a directory."
    img_files = list(img_dir.glob(f'*.{args.file_type}'))
    assert len(img_files) > 0, f"No images found in {args.img_dir}."

    # Load the calibration data
    cal_file = Path(args.cal_file)
    assert cal_file.is_file(), f"{args.cal_file} is not a file."
    camera_matrix, dist_coeffs = load_calibration_data(cal_file)

    # Create the output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate the tag sizes
    # If the config is single, then there should only be one tag size which may either be specified as a single value or a list of values
    tag_sizes = None
    if args.tag_config == "single":
        # Then either tag-size must be specified or a single tag-size must be specified in tag-sizes
        if args.tag_size is not None:
            tag_sizes = [args.tag_size]
        elif args.tag_sizes is not None:
            assert len(args.tag_sizes) == 1, f"Expected a single tag size, but got {len(args.tag_sizes)}."
            tag_sizes = args.tag_sizes
        else:
            raise Exception("Must specify either --tag-size or --tag-sizes.")
    elif args.tag_config == "double":
        # Then tag-sizes must be specifed and be a list of two values
        assert args.tag_sizes is not None, "Must specify --tag-sizes."
        assert len(args.tag_sizes) == 2, f"Expected two tag sizes, but got {len(args.tag_sizes)}."
        tag_sizes = args.tag_sizes
    else:
        raise Exception(f"Unknown tag config: {args.tag_config}")


    # Estimate the poses of the tags and camera
    tag_poses, camera_poses = estimate_poses(img_files, camera_matrix, dist_coeffs, tag_sizes, args.tag_config, out_dir, base_frame_tag_id=args.base_frame_tag_id, use_scipy=not args.use_gauss_newton, show_raw=args.show_raw)

    # Save the tag and camera poses
    out_file = out_dir / "poses.npz"
    np.savez(out_file.absolute().as_posix(), tag_poses=tag_poses, camera_poses=camera_poses)

    print("Done.")