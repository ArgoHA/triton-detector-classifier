# triton-detector-classifier

### Main idea
This repo shows how to deploy a pipeline with two neural networks utilizing Nvidia GPUs and [Triton Server](https://github.com/triton-inference-server).
### Architecture
Fitsr model is ia detector (trained on 1 class), it's a pretty fast one, but it's tuned to get higher recall than precision. To get less false positives second model was used - classifier, which is trained on more classes (5) and can catch false positives. Classifier works on crops of detector so it turns on only when detector catches something.
Two stages architecture gives us speed and accuracy at the same time.

### Performance
Designed for [Jetson](https://developer.nvidia.com/embedded-computing) nano, but works with other [cuda capable GPUs](https://developer.nvidia.com/cuda-gpus#compute). First neuralnet is [Yolov5](https://github.com/ultralytics/yolov5) on PyTorch converted to TensorRT, second one is [EfficientNetb0](https://arxiv.org/abs/1905.11946) deployed as [TensorFlow model](https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet).
I used rtsp stream 1920x1080 and got 6 fps. If classifier is triggered - processing of that frame is 2 times slower, so if at every frame detector finds one object - I get 3 fps.

### Preperations
- Install nvidia libs for deep learning (nvidia-drivers, cuda toolkit, cudnn)
- Install PyTorch
- Install Triton server

### Training
- Trained Yolov5s with custom dataset and used export.py to get model.engine (which is the same as model.plan)
- Trained EfficientNetb0 with slights changes of architecture using Tensorflow (see train_classifier.py)

### Deployment
For detector:
- put model.plan in model_repository/yolov5/1/
- correct config.pbtxt if needed (if ypu have another input/output for your model). Number of classes should be changed in output dims (number of classes + 5)

For classifier:
- put tensorflow_savedmodel files to model_repository/efnet/1/model.savedmodel/
- correct config.pbtxt if needed

### Start triton
I use systemctl to make a service from triton backend, so it is always alive when machine is powered. Use command like this to start triton server:
```
/installation_triton/bin/tritonserver --model-repository=/model_repository/ --backend-directory=/installation_triton/backends
```

### Start pipeline
- For usage on you webcam use:
```
python3 main.py --src webcam
```

- For running prerecorded video run:
```
python3 main.py --src test --test_vid_path 'path_to_video_here.mp4'
```

- For running rtsp streams in parallel (define links in main.py):
```
python3 main.py
```
