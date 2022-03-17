import cv2  # while not self.stream.emtpy():
#     frame = self.stream.non_blocking_read()
# if there are transforms to be done, might as well
# do them on producer thread before handing back to
# consumer thread. ie. Usually the producer is so far
# ahead of consumer that we have time to spare.
#
# Python is not parallel but the transform operations
# are usually OpenCV native so release the GIL.
#
# Really just trying to avoid spinning up additional
# native threads and overheads of additional
# producer/consumer queues since this one was generally
# idle grabbing frames.
import numpy as np
from .Measures import obtain_color_range


def mask_colors(img, color_range: list) -> np.array:
    mask = np.zeros_like(img[:, :, 1])
    for color in color_range:
        if len(color[0]) < 3:
            mask += cv2.inRange(img, color[0][0], color[0][1])
            mask += cv2.inRange(img, color[1][0], color[1][1])
        else:
            mask += cv2.inRange(img, color[0], color[1])
    return mask


def draw_grid(img: np.array,
              line_color: tuple = (0, 255, 0), line_thickness: float = 1, type_=cv2.LINE_AA,
              line_dst: float = 50) -> np.array:
    """
    Draw square grid lines on image
    :param np.array img: image
    :param tuple line_color: line color Default = (0, 255, 0)
    :param float line_thickness: line thickness Default=1
    :param type_: line type Default = cv2.LINE_AA
    :param grid_size: distance between lines
    :return: images with added gridlines
    """
    h, w = img.shape[:2]
    x = int((w % line_dst) / 2)
    y = int((h % line_dst) / 2)
    while x < w:
        cv2.line(img, (x, 0), (x, h), color=line_color, lineType=type_, thickness=line_thickness)
        x += line_dst

    while y < h:
        cv2.line(img, (0, y), (w, y), color=line_color, lineType=type_, thickness=line_thickness)
        y += line_dst
    return img
