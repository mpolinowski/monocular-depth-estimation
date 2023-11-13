# Depth Perception Projects

Run PyTorch projects inside a [pytorch-jupyter](https://github.com/mpolinowski/pytorch-jupyter) container:

```bash
docker run --ipc=host --gpus all -ti --rm \
    -v $(pwd):/opt/app -p 8888:8888 \
    --name pytorch-jupyter \
    pytorch-jupyter:latest
```
