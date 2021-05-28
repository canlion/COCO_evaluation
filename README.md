# COCO-mAP

## COCO evaluation
https://github.com/cocodataset/cocoapi

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
