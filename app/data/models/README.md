# Model Files for Object Detection

This directory needs to contain the following files for the object detection tool to work:

1. `yolov3.weights` - The YOLOv3 weights file
2. `yolov3.cfg` - The YOLOv3 configuration file  
3. `coco.names` - The COCO class names file

You can download these files from:
- YOLOv3 weights: https://pjreddie.com/media/files/yolov3.weights
- YOLOv3 cfg and coco.names: https://github.com/pjreddie/darknet/tree/master/cfg

Or use your own custom models by specifying their paths in the object_detection function call. 