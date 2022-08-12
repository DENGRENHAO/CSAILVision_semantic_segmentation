'''
This code is to run simple version of inference model of SOTA semantic segmentation with pytorch
'''

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import csv
from PIL import Image
if not hasattr(Image, 'Resampling'):  # Pillow<9.0
    Image.Resampling = Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
import urllib.request
import argparse
import pandas as pd
# MIT libraries
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, unique
from mit_semseg.lib.utils import as_numpy
from mit_semseg.lib.nn import async_copy_to

import imgviz
import PIL.Image
from json_to_label import json_to_label
from sklearn.metrics import confusion_matrix

def parse_argument():
    parser = argparse.ArgumentParser(description='Get GVI, Sky, Building scores by Semantic Segmentation')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input images directory')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output results directory')
    parser.add_argument('-j', '--json', type=str, required=False,
                        help='Input labeled json file directory')
    parser.add_argument('-e', '--evaluate', type=bool, default=False,
                        help='If evaluate model\'s scores')

    return parser.parse_args()

def build_encoder(arch_encoder,fc_dim,weights):
	# Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=arch_encoder,
        fc_dim=fc_dim,
        weights=weights)

    return net_encoder

def build_decoder(arch_decoder,fc_dim,num_class,weights):
    # Network Builders
    net_decoder = ModelBuilder.build_decoder(
        arch=arch_decoder,
        fc_dim=fc_dim,
        num_class=num_class,
        weights=weights,
        use_softmax=True)

    return net_decoder

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.Resampling.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def img_transform(img):
    # 0-255 to 0-1
    img = np.float32(np.array(img)) / 255.
    img = img.transpose((2, 0, 1))
    # mean and std
    normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img.copy()))
    return img

def datapreprocess(image_path,imgSizes,imgMaxSize,padding_constant):
    img = Image.open(image_path).convert('RGB')
    ori_width, ori_height = img.size
    # Check if image is larger than 2048 x 2048
    # if too large than resize
    if ori_width > 2048 or ori_height > 2048:
        img = imresize(img, (2048, 2048), interp='bilinear')
        ori_width, ori_height = img.size

    img_resized_list = []

    for this_short_size in imgSizes:
        # calculate target height and width
        scale = min(this_short_size / float(min(ori_height, ori_width)),
                    imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)

        # to avoid rounding in network
        target_width = round2nearest_multiple(target_width, padding_constant)
        target_height = round2nearest_multiple(target_height, padding_constant)

        # resize images
        img_resized = imresize(img, (target_width, target_height), interp='bilinear')

        # image transform, to torch float tensor 3xHxW
        img_resized = img_transform(img_resized)
        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)

    data = dict()
    data['img_ori'] = np.array(img)
    data['img_data'] = [x.contiguous() for x in img_resized_list]
    
    segSize = (data['img_ori'].shape[0],
               data['img_ori'].shape[1])
    img_resized_list = data['img_data']

    return img_resized_list, segSize

def compute_scores(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    pred_labels = np.unique(y_pred)
    true_labels = np.unique(y_true)
    labels = np.union1d(pred_labels, true_labels)
    current = confusion_matrix(y_true, y_pred, labels=labels)
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    pixel_accuracy = np.sum(intersection) / len(y_pred)
    return np.mean(IoU), pixel_accuracy

row_lists = []

def visualize_result(image_path, output_path, image_name, pred, names, colors, W_global, H_global, json_path, evaluate=False):
    img_ori = cv2.imread(os.path.join(image_path, image_name))
    img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    if img_ori.shape[0] > 2048 or img_ori.shape[1] > 2048:
        img_ori = cv2.resize(img_ori,(2048,2048))

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    gvi = 0
    sky = 0
    building = 0
    row_dict = {
        "AudioFile_name": image_name,
        "GVI": 0.00,
        "sky": 0.00,
        "building": 0.00,
        "grass": 0.00,
        "tree": 0.00,
        "flower": 0.00,
        "plant": 0.00,
    }
    print("Prediction info is:")
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))
        if name=='grass' or name=='tree' or name=='flower' or name=='plant':
            gvi += ratio
            row_dict[name] = ratio
        elif name=='sky':
            sky += ratio
        elif name=='building' or name=='house':
            building += ratio
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # resize to global if necessary:
    if img_ori.shape[0] != H_global and img_ori.shape[1] != W_global:
        img_ori = cv2.resize(img_ori,(W_global,H_global))
        pred_color = cv2.resize(pred_color,(W_global,H_global))

    # show original images and semantics together
    im_vis = np.concatenate((img_ori, pred_color), axis=1)
    plt.imsave(os.path.join(output_path+image_name[:-4]+'.png'),im_vis)

    if evaluate:
        label_names = ['_background_', 'tree', 'grass', 'plant', 'flower', 'sky', 'building']
        labelmap = pred
        for i in range(labelmap.shape[0]):
            for j in range(labelmap.shape[1]):
                if labelmap[i][j] != 4 and labelmap[i][j] != 9 and labelmap[i][j] != 17 and labelmap[i][j] != 66 and \
                        labelmap[i][j] != 2 and labelmap[i][j] != 1:
                    labelmap[i][j] = 0
                elif labelmap[i][j] == 1:
                    labelmap[i][j] = 6
                elif labelmap[i][j] == 4:
                    labelmap[i][j] = 1
                elif labelmap[i][j] == 2:
                    labelmap[i][j] = 5
                elif labelmap[i][j] == 66:
                    labelmap[i][j] = 4
                elif labelmap[i][j] == 17:
                    labelmap[i][j] = 3
                elif labelmap[i][j] == 9:
                    labelmap[i][j] = 2
        colormap = np.array([[0, 0, 0], colors[4], colors[9], colors[17], colors[66], colors[2], colors[7]])
        lbl_viz = imgviz.label2rgb(
            labelmap, imgviz.asgray(img_ori), label_names=label_names, font_size=20, colormap=colormap, loc="rb"
        )
        PIL.Image.fromarray(lbl_viz).save(os.path.join(output_path, image_name[:-4] + "_inference_label.png"))
        real_label = json_to_label(os.path.join(json_path, image_name[:-4] + ".json"), colormap, output_path)
        miou, pixel_accuracy = compute_scores(labelmap, real_label)
        row_dict["pixel_accuracy"] = pixel_accuracy
        row_dict["mIoU"] = miou
        values, counts = np.unique(real_label, return_counts=True)
        real_gvi = 0
        real_sky = 0
        real_building = 0
        for v, c in zip(values, counts):
            if v >= 1 and v <= 4:
                real_gvi += c
            elif v==5:
                real_sky += c
            elif v==6:
                real_building += c
        row_dict["real_GVI"] = real_gvi / real_label.size * 100
        row_dict["real_sky"] = real_sky / real_label.size * 100
        row_dict["real_building"] = real_building / real_label.size * 100
    row_dict["GVI"] = gvi
    row_dict["sky"] = sky
    row_dict["building"] = building
    row_lists.append(row_dict)
    # df = pd.DataFrame({"key": row_dict.keys(), "value": row_dict.values()})
    # img_name = image_name.split(".")
    # path = os.path.join(os.getcwd(), os.path.join(output_path, 'scores'))
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # df.to_csv(os.path.join(path, 'scores_of_' + img_name[0] + '.csv'), index=False)
    return

# arguments for model HRNETV2:
arch_encoder = 'hrnetv2'
arch_decoder = 'c1'
fc_dim = 720
encoder_weights_path = './ade20k-hrnetv2-c1/encoder_epoch_30.pth'
decoder_weights_path = './ade20k-hrnetv2-c1/decoder_epoch_30.pth'
num_class = 150
imgSizes = (300, 375, 450, 525, 600) # multi-scale prediction
padding_constant = 32
imgMaxSize = 1000

weights_folder = './ade20k-hrnetv2-c1/'
if not os.path.exists(weights_folder):
    os.makedirs(weights_folder)
if not os.path.isfile(encoder_weights_path):
    print('Downloading encoder weights......')
    urllib.request.urlretrieve("http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/encoder_epoch_30.pth",
                               weights_folder + "encoder_epoch_30.pth")
if not os.path.isfile(decoder_weights_path):
    print('Downloading decoder weights......')
    urllib.request.urlretrieve("http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-hrnetv2-c1/decoder_epoch_30.pth",
                               weights_folder + "decoder_epoch_30.pth")

image_path = ''
output_path = ''
json_path = ''
evaluate = False
args = parse_argument()
if args.input:
    image_path = args.input
if args.output:
    output_path = args.output
if args.json:
    json_path = args.json
if args.evaluate:
    evaluate = args.evaluate

# read color table:
colors = loadmat('./data/color150.mat')['colors']
names = {}
with open('./data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

# turn on GPU for models:
gpu = -1 # gpu device index
if gpu >= 0:
    torch.cuda.set_device(gpu)

# Build Models:
net_encoder = build_encoder(arch_encoder=arch_encoder, fc_dim=fc_dim, weights=encoder_weights_path)

net_decoder = build_decoder(arch_decoder=arch_decoder, fc_dim=fc_dim, num_class=num_class, weights=decoder_weights_path)

# Negative likelihood loss:
crit = nn.NLLLoss(ignore_index=-1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

# turn on cuda:
if gpu >= 0:
    segmentation_module.cuda()

segmentation_module.eval()

for image_name in os.listdir(image_path):
    
    # record the global image size:
    img = Image.open(os.path.join(image_path, image_name)).convert('RGB')
    W_global, H_global = img.size
    del img

    # read image and process:
    print("Read image {} and preprocessing......".format(image_name))
    img_resized_list, segSize = datapreprocess(os.path.join(image_path, image_name),imgSizes,imgMaxSize,padding_constant)

    # inference:
    print("Inference starts......")
    
    with torch.no_grad():
        scores = torch.zeros(1, num_class, segSize[0], segSize[1])
        if gpu >= 0: # use GPU memory to handle data
            scores = async_copy_to(scores, gpu)

        for img in img_resized_list:
            feed_dict = {}
            feed_dict['img_data'] = img
            if gpu >= 0: # use GPU memory to handle data
                feed_dict = async_copy_to(feed_dict, gpu)
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            scores = scores + pred_tmp / len(imgSizes)
        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

    # visualize and save:
    print("Visualization and save......")
    # return original sized prediction discretized class map
    visualize_result(image_path,output_path,image_name,pred,names,colors,W_global,H_global,json_path,evaluate=evaluate)
path = os.path.join(os.getcwd(), os.path.join(output_path, 'scores'))
if not os.path.exists(path):
    os.makedirs(path)
avg_pixel_accuracy = sum(row_dict["pixel_accuracy"] for row_dict in row_lists) / len(row_lists)
avg_miou = sum(row_dict["mIoU"] for row_dict in row_lists) / len(row_lists)
row_lists[0]["avg_pixel_accuracy"] = avg_pixel_accuracy
row_lists[0]["avg_mIoU"] = avg_miou
df = pd.DataFrame(row_lists)
df.to_csv(os.path.join(path, 'all_scores.csv'), index=False)
