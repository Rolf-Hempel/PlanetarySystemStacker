import glob
from time import time

from align_frames import AlignFrames
from alignment_points import AlignmentPoints
from configuration import Configuration
from frames import Frames
from rank_frames import RankFrames
from quality_areas import QualityAreas

if __name__ == "__main__":
    names = glob.glob('Images/2012*.tif')
    # names = glob.glob('Images/Moon_Tile-031*ap85_8b.tif')
    # names = glob.glob('Images/Example-3*.jpg')
    print(names)
    configuration = Configuration()
    try:
        frames = Frames(names, type='image')
        print("Number of images read: " + str(frames.number))
        print("Image shape: " + str(frames.shape))
    except Exception as e:
        print("Error: " + e.message)
        exit()

    rank_frames = RankFrames(frames, configuration)
    start = time()
    rank_frames.frame_score()
    end = time()
    print('Elapsed time in ranking images: {}'.format(end - start))
    print("Index of maximum: " + str(rank_frames.frame_ranks_max_index))
    print("Frame scores: " + str(rank_frames.frame_ranks))
    print("Frame scores (sorted): " + str([rank_frames.frame_ranks[i] for i in rank_frames.quality_sorted_indices]))
    print("Sorted index list: " + str(rank_frames.quality_sorted_indices))

    align_frames = AlignFrames(frames, rank_frames, configuration)
    start = time()
    (x_low_opt, x_high_opt, y_low_opt, y_high_opt) = align_frames.select_alignment_rect(
        configuration.alignment_rectangle_scale_factor)
    end = time()
    print('Elapsed time in computing optimal alignment rectangle: {}'.format(end - start))
    print("optimal alignment rectangle, x_low: " + str(x_low_opt) + ", x_high: " + str(x_high_opt) + ", y_low: " + str(
        y_low_opt) + ", y_high: " + str(y_high_opt))
    reference_frame_with_alignment_points = align_frames.frames_mono[align_frames.frame_ranks_max_index].copy()
    reference_frame_with_alignment_points[y_low_opt, x_low_opt:x_high_opt] = reference_frame_with_alignment_points[
                                                                             y_high_opt - 1, x_low_opt:x_high_opt] = 255
    reference_frame_with_alignment_points[y_low_opt:y_high_opt, x_low_opt] = reference_frame_with_alignment_points[
                                                                             y_low_opt:y_high_opt, x_high_opt - 1] = 255
    # plt.imshow(reference_frame_with_alignment_points, cmap='Greys_r')
    # plt.show()

    start = time()
    align_frames.align_frames()
    end = time()
    print('Elapsed time in aligning all frames: {}'.format(end - start))
    print("Frame shifts: " + str(align_frames.frame_shifts))
    print("Intersection: " + str(align_frames.intersection_shape))

    start = time()
    alignment_points = AlignmentPoints(configuration, frames, rank_frames, align_frames)
    end = time()
    print('Elapsed time in computing average frame: {}'.format(end - start))
    print("Average frame computed from the best " + str(alignment_points.average_frame_number) + " frames.")
    # plt.imshow(align_frames.mean_frame, cmap='Greys_r')
    # plt.show()

    step_size = configuration.alignment_box_step_size
    box_size = configuration.alignment_box_size
    start = time()
    alignment_points.create_alignment_boxes(step_size, box_size)
    end = time()
    print('Elapsed time in alignment box creation: {}'.format(end - start))
    print("Number of alignment boxes created: " + str(len(alignment_points.alignment_boxes)))

    structure_threshold = configuration.alignment_point_structure_threshold
    brightness_threshold = configuration.alignment_point_brightness_threshold
    contrast_threshold = configuration.alignment_point_contrast_threshold
    print("Selection of alignment points, structure threshold: " + str(
        structure_threshold) + ", brightness threshold: " + str(brightness_threshold) + ", contrast threshold: " + str(
        contrast_threshold))
    start = time()
    alignment_points.select_alignment_points(structure_threshold, brightness_threshold, contrast_threshold)
    end = time()
    print('Elapsed time in alignment point selection: {}'.format(end - start))
    print("Number of alignment points selected: " + str(len(alignment_points.alignment_points)))

    for frame_index in range(frames.number):
        frame_with_shifts = reference_frame_with_alignment_points.copy()
        start = time()
        point_shifts, errors, diffphases = alignment_points.compute_alignment_point_shifts(frame_index)
        end = time()
        print("Elapsed time in computing point shifts for frame number " + str(frame_index) + ": " + str(end - start))

    start = time()
    quality_areas = QualityAreas(configuration, frames, align_frames, alignment_points)
    quality_areas.select_best_frames()
    quality_areas.truncate_best_frames()
    end = time()
    print('Elapsed time in quality area creation and frame ranking: {}'.format(end - start))
    print("Number of frames to be stacked for each quality area: " + str(quality_areas.stack_size))