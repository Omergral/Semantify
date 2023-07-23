<h1 align="center">Semantify:<br>Simplifying the Control of 3D Morphable Models using CLIP<br>ICCV 2023</h1>

<h2 align="center"><p>
  <a href="https://Omergral.github.io/Semantify" align="center">🖥️ Project Page</a> | 
  <a href="https://arxiv.org/abs/2211.17256" align="center">📄 Paper</a>
</p></h2>

<p align="center">
    <img src="https://github.com/Omergral/Semantify/blob/master/static/images/semantify_teaser-bg.png" alt="example" width=80%>
    <br>
    <em>Semantify offers a method to create and edit a 3D parametric model using semantically meaningful descriptors. Semantify is based on a self-supervised method that utilizes the semantic power of CLIP language-vision model to build a mapping between semantic descriptors to 3DMM model coefficients. This can be used in an interactive application defining a slider for each descriptor (a), or to fit a model to an image in a zero shot manner by feeding the image into CLIP and obtaining a vector of semantic scores that can be mapped to shape parameters (b).</em>
</p>

## Installation
1. Clone the repo:
```
git clone https://github.com/Omergral/Semantify.git
cd Semantify 
```
2. Create new conda environment:
```
conda env create -f requirements.yml
conda activate semantify
pip install -e .
```

## Download Models
TO BE FILLED

## Run Demos
### Sliders Application
```
python semantify/applications/sliders_demo_py3d.py --model_type <MODELTYPE> --specific <SPECIFIC>
```
<ins>Arguments Description:</ins>
* ```model_type``` - "smplx" | "smpl" | "flame" | "smal"
* ```specific``` - "male" | "female" | "neutral" | "expression" | "shape"
* ```image_path (str)``` - If you want to run the demo on an image, provide the path to the image
* ```mapper_path (str)``` - If you don't want to use Semantify's mappers, set the path to desired ckpt file
* ```use_raw_blendshapes``` - Use the raw parametric blendshapes of the model

### Image-to-Shape
```
python semantify/applications/image2shape.py --images_paths <IMAGES_PATHS> --model_type <TYPE> --specific <SPECIFIC> --output_path <PATH>
```

<ins>Arguments Description</ins>
* ```images_paths (List[str])``` - Paths to the images you wish to run the demo on
* ```model_type``` - "smplx" | "smpl" | "flame" | "smal"
* ```specific``` - "male" | "female" | "neutral" | "expression" | "shape"
* ```mapper_path (str)``` - If you don't want to use Semantify's mappers, set the path to desired ckpt file

## Dataset Creation
1. Create Data
   ```
   python semantify/data_management/creation/create_data.py --output_path <PATH> --model_type <TYPE> --specific <SPECIFIC> --use_tex --multiview --num_of_imgs <NUM>
   ```
   This script will create as many images as you like, by randomly sample the parametric space of the given 3DMM. The output for a single sample will be a ```.png``` file and ```.json``` file that contains the sampled shape coefficients.<br><br>
   <ins>Arguments Description:</ins>
   * ```output_path (str)``` - Path to a folder to store the data
   * ```use_tex``` - Use models textures (if any) - HIGHLY RECOMMENDED
   * ```multiview``` - Render the model from both frontal and side views instead of frontal only
   * ```num_of_imgs (int)``` - How many images to create

2. Generate CLIP's Ratings
   ```
   python semantify/data_management/creation/clip_encoder.py --imgs_dir <PATH> --model_type <TYPE> --specific <SPECIFIC> --multiview
   ```
   This script will run over the directory of images provided as input, along with a set of word descriptors and generate CLIP's           ratings for each image against all descriptors. The output for a single sample will be a ```.json``` file that contains the ratings     for each descriptor.<br><br>
   <ins>Arguments Description:</ins><br>
   * ```imgs_dir (str)``` - Path to the directory of created images from phase (1).
   * ```descriptors (List[str])``` - List of descriptors to use. We supply a default set, so this field is optional.<br><br>

## Train from Scratch
First in ```semantify/config/train_mapper.yaml``` fill the following fields:
* ```output_path``` - Where to store the mapper
* ```run_name``` - Name the run
* ```dataset.datadir``` - Path to the data directory
* ```dataset.optimize_feature``` - betas (SMPLX | SMPL) / beta (SMPL) / expression_params (FLAME) / shape_params (FLAME)
* ```dataset.labels_to_get``` - On which descriptors to optimize<br><br>
These are **MUST** arguments to fill in order to train the mapper, in the ```.yaml``` file you will find more configuration options.

Then, to train the mapper run:
```
python semantify/train/train_mapper.py
```

## Citation
If you make use of our work, please cite our paper:
```
@inproceedings{Semantify-23,
    author  = {Omer Gralnik and Guy Gafni and Ariel Shamir},
    title   = {Semantify: Simplifying the Control of 3D Morphable Models using CLIP},
    booktitle = {Proceedings of the International Conference on Computer Vision},
    pages = {Accepted},
    year    = {2023},
}
```
