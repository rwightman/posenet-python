""" Program counts squats """
import time
import copy
import numpy as np
import posenet
import webcam_demo



def main():
    """ Wait 3 sec for person to get stable position, find initial coordinates,
        until poseNet is stopped via 'q', check if knees positions have changed
    """
    time.sleep(3)
    outputs = webcam_demo.main()

    for _ in range(30):
        count = 1
        pose_scores_start, _, kp_coords_start = next(outputs)
        for pose, kp_coord_start in enumerate(kp_coords_start):
            if pose_scores_start[pose] != 0.:
                try:
                    kp_coords_start_av = kp_coords_start_av + kp_coord_start
                except NameError:
                    kp_coords_start_av = copy.deepcopy(kp_coord_start)
                else:
                    count += 1
        if count != 0:
            kp_coords_start_av = np.divide(kp_coords_start_av, count)

    compare_val = abs(kp_coords_start_av[0, 0] - kp_coords_start_av[-1, 0])*0.03
    count = 0
    down = False

    while True:
        try:
            pose_scores, keypoint_scores, kp_coords = next(outputs)
            for pose in range(len(pose_scores)):
                if pose_scores[pose] == 0.:
                    break
                l_knee_in = posenet.PART_NAMES.index("leftKnee")
                r_knee_in = posenet.PART_NAMES.index("rightKnee")
                if keypoint_scores[pose, l_knee_in] > 0.5:
                    diff = abs(kp_coords[pose, l_knee_in, :] - kp_coords_start_av[l_knee_in, :])
                elif keypoint_scores[pose, r_knee_in] > 0.5:
                    diff = abs(kp_coords[pose, r_knee_in, :] - kp_coords_start_av[r_knee_in, :])
                else:
                    break

                if diff[0] > compare_val and not down:
                    down = True
                elif diff[0] < compare_val and down:
                    down = False
                    count += 1

        except StopIteration:
            break

    print("Squats amount: ", count)


if __name__ == "__main__":
    main()
