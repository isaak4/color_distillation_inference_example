# color_distillation_inference_example

## Overview

This is a small example of using the [ColorCNN](https://github.com/hou-yz/color_distillation) architecture to reduce the colors of an image.
I wanted to checkout the results when the number of colors is set to 16.
As described in the [paper](https://arxiv.org/abs/2003.07848) this architecture has advantages when used in very low dimensional color space (e.g. 2 or 4 colors) and does not work well for more colors, where other clustering methods give superior results.

## Dependencies

Use the ```spec-file.txt``` to create a conda environment by executing:

``` conda create --name myenv --file spec-file.txt```

## How to use

``` 
conda activate myenv
python color_cnn_inference.py
```
