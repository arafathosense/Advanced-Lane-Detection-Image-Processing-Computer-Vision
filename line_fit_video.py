import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from combined_thresh import combined_thresh
from perspective_transform import perspective_transform
from Line import Line
from line_fit import line_fit, tune_fit, final_viz, calc_curve, calc_vehicle_offset
from moviepy.editor import VideoFileClip


# Load Camera Calibration
with open('calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)

mtx = save_dict['mtx']
dist = save_dict['dist']

# Global variables
window_size = 5
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False
left_curve, right_curve = 0., 0.
left_lane_inds, right_lane_inds = None, None


def annotate_image(img_in):
    """Annotate a single frame with lane markings."""
    global mtx, dist, left_line, right_line, detected
    global left_curve, right_curve, left_lane_inds, right_lane_inds

    # Undistort
    undist = cv2.undistort(img_in, mtx, dist, None, mtx)

    # Threshold + perspective transform
    img, abs_bin, mag_bin, dir_bin, hls_bin = combined_thresh(undist)
    binary_warped, _, _, m_inv = perspective_transform(img)

    # First detection (slow)
    if not detected:
        ret = line_fit(binary_warped)

        if ret is None:
            return undist  # fallback

        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        nonzerox = ret['nonzerox']
        nonzeroy = ret['nonzeroy']
        left_lane_inds = ret['left_lane_inds']
        right_lane_inds = ret['right_lane_inds']

        # Smooth
        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)

        # Curvature
        left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)
        detected = True

    else:  # Fast fit
        prev_left = left_line.get_fit()
        prev_right = right_line.get_fit()

        ret = tune_fit(binary_warped, prev_left, prev_right)

        if ret is None:
            detected = False
            return undist

        left_fit = ret['left_fit']
        right_fit = ret['right_fit']
        nonzerox = ret['nonzerox']
        nonzeroy = ret['nonzeroy']
        left_lane_inds = ret['left_lane_inds']
        right_lane_inds = ret['right_lane_inds']

        left_fit = left_line.add_fit(left_fit)
        right_fit = right_line.add_fit(right_fit)

        left_curve, right_curve = calc_curve(left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

    # Vehicle offset
    vehicle_offset = calc_vehicle_offset(undist, left_fit, right_fit)

    # Final visualization
    result = final_viz(undist, left_fit, right_fit, m_inv,
                       left_curve, right_curve, vehicle_offset)

    return result


def annotate_video(input_file, output_file):
    video = VideoFileClip(input_file)
    annotated_video = video.fl_image(annotate_image)
    annotated_video.write_videofile(output_file, audio=False)


if __name__ == '__main__':
    # Output video
    annotate_video('project_video.mp4', 'output_video.mp4')

    # Single image test
    img = mpimg.imread('test_images/test2.jpg')
    result = annotate_image(img)
    plt.imshow(result)
    plt.show()
