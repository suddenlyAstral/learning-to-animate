# Learning change through time from synthetic data

<p align="center">
  <img width="100%" src="https://mbaradad.github.io/shaders21k/images/teaser.png">
</p>

This project takes code and inspiration from shaders21k:

[[Project page](https://mbaradad.github.io/shaders21k)] 
[[Paper](https://arxiv.org/abs/2211.16412)]

# Requirements 
```pip install requirements.txt```

The training code also logs to wandb which requires a wandb account.

To render with the shaders with OpenGL and GPU, NVIDIA cards supporting CUDA should be able to render by default. 

# Download shader codes, data and models
For the shader codes used in the paper, we provide a downloaded version from the original sources as they were publicily available during October 2021.
```
./scripts/download/download_shader_codes.sh
```
The license for the codes are the same as the original shaders, and can be accessed using the identifiers included in the previous under shader_codes/shader_info.

# Data generation

The main rendering functionality for shaders is under ```image_generation/shaders/renderer_moderngl.py```. 

If you want to generate data for a single shader, you can use the utility ```image_generation/shaders/generate_data_single_shader.py```, for example as:

```
python image_generation/shaders/generate_data_single_shader.py --shader-file shader_codes/shadertoy/W/Wd2XWy.fragment --n-samples 105000 --resolution 256 --output-path shader_images/Wd2XWy
```

If you want to generate data for a multiple shaders, you can use the utility ```image_generation/shaders/generate_by_shader_dir.py```, for example as:

```
python image_generation/shaders/generate_by_shader_dir.py --shader-dir shader_codes/shadertoy/W --n-samples 105000 --resolution 256 --output-path shader_images/Wd2XWy
```



# Training
The training code unique to this project is (mostly) in cvae_training and image_generation/dataloaing.py
You can train the model like so:

```python cvae_training/main.py --twigl --lr 0.0001 --latent_dim 64 --triplets 1.0```
