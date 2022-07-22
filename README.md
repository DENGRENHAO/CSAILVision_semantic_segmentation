# Semantic Segmentation
Simple inference implementation with trained HRNet on MIT ADE20K dataset, using PyTorch 1.6.0. Most of the code taken from [1].

## Prerequisite
```
git clone https://github.com/DENGRENHAO/semantic-segmentation.git
```
```
pip install -r requirements.txt
```

## Usage
查看可用的選項：
```
python main.py --help
```
### 範例
```
python semantic_segmentation.py -i ./input_img_folder/ -o ./output_img_folder/
```
- 在 `./input_img_folder/`資料夾中的圖片皆會經過Semantic Segmentation，最後輸出結果至`./output_img_folder/`資料夾中

## Results

### semantic map

![Image of semantic map](https://github.com/liuch37/semantic-segmentation/blob/master/misc/ADE_test_00000272.png)

## Source
[1] Original code: https://github.com/CSAILVision/semantic-segmentation-pytorch.

[2] HRNet: https://arxiv.org/abs/1904.04514.
