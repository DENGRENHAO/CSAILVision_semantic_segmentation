import argparse
import base64
import json
import os
import os.path as osp

import imgviz
import PIL.Image

from labelme.logger import logger
from labelme import utils
import numpy as np

def json_to_label(json_file, colormap, out_dir='./output'):
    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name not in label_name_to_value:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    uniques, counts = np.unique(lbl, return_counts=True)
    all_labels = ['_background_', 'tree', 'grass', 'plant', 'flower', 'sky', 'building']
    real_label = lbl
    idx = []
    for label in label_names:
        idx.append(all_labels.index(label))
    for i in range(real_label.shape[0]):
        for j in range(real_label.shape[1]):
            real_label[i][j] = idx[real_label[i][j]]+10
    real_label -= 10
    lbl_viz = imgviz.label2rgb(
        real_label, imgviz.asgray(img), label_names=all_labels, font_size=20, colormap=colormap, loc="rb"
    )
    filename = osp.basename(json_file)[:-5]
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, filename+"_ground_truth_label.png"))
    return real_label
