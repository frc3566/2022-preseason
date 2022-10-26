# original source: https://github.com/Kazuhito00/AprilTag-Detection-Python-Sample/blob/main/sample.py
#!/usr/bin/env python
import copy
import time
import argparse

import cv2 as cv
from pupil_apriltags import Detector


def main():
    # Camera setup
    cap = cv.VideoCapture(0)

    # Detector setup
    at_detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    elapsed_time = 0

    while True:
        start_time = time.time()

        # camera capture
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)

        # detection
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        tags = at_detector.detect(
            image,
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )

        # draw tags
        debug_image = draw_tags(debug_image, tags, elapsed_time)

        elapsed_time = time.time() - start_time

        # quit
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # show feed
        cv.imshow('AprilTag Detect Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()


def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # 中心
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)

        # 各辺
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        # タグファミリー、タグID
        # cv.putText(image,
        #            str(tag_family) + ':' + str(tag_id),
        #            (corner_01[0], corner_01[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
        #            0.6, (0, 255, 0), 1, cv.LINE_AA)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    # 処理時間
    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


if __name__ == '__main__':
    main()