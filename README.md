# Semantic Segmentation
Getting GVI, SVF, BFV scores from semantic segmentation. Using pretrained HRNet on MIT ADE20K dataset, most of the code taken from [CSAILVision](https://github.com/CSAILVision/semantic-segmentation-pytorch) and simple inference implementation from [liuch37](https://github.com/liuch37/semantic-segmentation).

## Prerequisite
```
git clone https://github.com/DENGRENHAO/semantic-segmentation.git
```
```
cd .\semantic-segmentation\
```
```
pip install -r requirements.txt
```

## Usage
### 範例
```
python semantic_segmentation.py -i ./input_img_folder/ -o ./output_img_folder/
```
- 在 `./input_img_folder/`資料夾中的圖片皆會經過Semantic Segmentation，最後輸出結果至`./output_img_folder/`資料夾中

## Results

### semantic map

![Image of semantic map](https://github.com/liuch37/semantic-segmentation/blob/master/misc/ADE_test_00000272.png)

