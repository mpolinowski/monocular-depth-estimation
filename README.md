# Depth Perception Projects

Run PyTorch projects inside a [pytorch-jupyter](https://github.com/mpolinowski/pytorch-jupyter) container:

```bash
docker run --ipc=host --gpus all -ti --rm \
    -v $(pwd):/opt/app -p 8888:8888 \
    --name pytorch-jupyter \
    pytorch-jupyter:latest
```

1. [MiDaS v3 Image Depth Maps](https://github.com/mpolinowski/monocular-depth-estimation/tree/master/01_MiDaS3_DPT_PyTorch_Hub_Images)
2. [MiDaS v3 RTSP Stream](https://github.com/mpolinowski/monocular-depth-estimation/tree/master/02_MiDaS3_DPT_PyTorch_Hub_RTSP)
3. [NianticLabs MonoDepth2](https://github.com/mpolinowski/monocular-depth-estimation/tree/master/03_NianticLabs_Monodepth2)
