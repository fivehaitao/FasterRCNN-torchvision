import random

import cv2


def _random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def draw_picture(src_image, inference_out, label_names, threshold):
    src_img = src_image
    boxes = inference_out[0]['boxes']
    labels = inference_out[0]['labels']
    scores = inference_out[0]['scores']

    for idx in range(boxes.shape[0]):
        if scores[idx] >= threshold:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = label_names.get(str(labels[idx].item()))
            # cv2.rectangle(img,(x1,y1),(x2,y2),colors[labels[idx].item()],thickness=2)
            cv2.rectangle(src_img, (x1, y1), (x2, y2), _random_color(), thickness=2)
            cv2.putText(src_img, text=name, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

    return src_img
