# COCO-mAP

## COCO evaluation
https://github.com/cocodataset/cocoapi

# 주의!!!!!!!!!!!!!!
* 이 코드는 coco annotation의 'iscrowd' flag를 무시, 입력되는 ground-truth와 detection이 모두 유효하다고 가정함.
* 이 코드는 모든 bounding box의 area 값을 박스의 가로와 세로의 곱으로 설정. (coco의 ground-truth annotation의 area는 세그멘테이션 넓이로 설정됨.)
* coco example 비교: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb 
    * 위 두가지 주의점을 반영하지 않은채로 mAP 측정
        * pycocotools
            * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.505 ...
        * 내 코드
            * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502 ...
        * 차이 발생!
    * 위 두가지 주의점을 반영하기위해 coco ground-truth의 area를 박스 넓이로, is_crowd를 모두 0으로 설정하여 mAP 측정
        * pycocotools
            * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
        * 내 코드
            * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.502
        * 동일한 결과.
* **그러므로 이 코드의 결과값은 공식적인 COCO mAP 측정값과 비교 불가.**

## 목적
* COCO evaluation 수행하려면 COCO dataset format으로 ground-truth / detection 파일 생성이 필요하여 불편하므로 inference 과정에서 ground-truth / detection 결과를 넘겨주어 mAP 계산이 가능하도록 함.

## 사용
이미지별로 detection, ground-truth numpy ndarray를 add method를 통해 추가 후 accumulate(), summarize()

* detection: numpy array / shape: (m, 6) / x(left), y(top), w, h, score, category
* ground-truth: numpy array / shape: (n, 5) / x(left), y(top), w, h, category
### code
```python
from evaluator import Evaluator

...

coco_eval = Evaluator()
for gt, input_ in dataset:
    ...
    dt = model_inference(input_)
    coco_eval.add(detection, gt)
    
coco_eval.accumulate()
coco_eval.summarize()
```
### output
```text
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.680
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.305
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.921
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.480
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.723
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.946
```
