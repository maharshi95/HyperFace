import dlib


def find_iou(bb1, bb2):
    top_left_x1, top_left_y1, bottom_right_x1, bottom_right_y1 = bb1

    top_left_x2, top_left_y2, bottom_right_x2, bottom_right_y2 = bb2

    intersect_top_left_x = max(top_left_x1, top_left_x2)
    intersect_top_left_y = max(top_left_y1, top_left_y2)
    intersect_bottom_right_x = min(bottom_right_x1, bottom_right_x2)
    intersect_bottom_right_y = min(bottom_right_y1, bottom_right_y2)

    intersect_area = (intersect_bottom_right_x - intersect_top_left_x + 1) * (intersect_bottom_right_y - intersect_top_left_y + 1)

    total_area = (
            (bottom_right_x1 - top_left_x1 + 1) * (bottom_right_y1 - top_left_y1 + 1)
            + (bottom_right_x2 - top_left_x2 + 1) * (bottom_right_y2 - top_left_y2 + 1)
            - intersect_area
    )

    iou = float(intersect_area) / float(total_area + 0.0)

    return iou


def selective_search(img, w, h, ground_truth):
    rects = []
    dlib.find_candidate_object_locations(img, rects, min_size=500)
    filter_positive_rects = []
    filter_negative_rects = []

    for rect in rects:
        iou = find_iou(ground_truth, (rect.left(), rect.top(), rect.right(), rect.bottom()))

        if iou > 0.5:
            filter_positive_rects.append([rect.top() / h, rect.left() / w, rect.bottom() / h, rect.right() / w])
        elif iou < 0.35:
            filter_negative_rects.append([rect.top() / h, rect.left() / w, rect.bottom() / h, rect.right() / w])

    return filter_positive_rects, filter_negative_rects
