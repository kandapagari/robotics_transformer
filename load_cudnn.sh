#! /bin/bash

CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
