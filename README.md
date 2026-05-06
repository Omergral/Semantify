<h1 align="center">Semantify:<br>Simplifying the Control of 3D Morphable Models using CLIP<br>ICCV 2023</h1>

<h2 align="center"><p>
  <a href="https://Omergral.github.io/Semantify" align="center">🖥️ Project Page</a> | 
  <a href="https://arxiv.org/abs/2308.07415" align="center">📄 Paper</a>
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
### Semantify Mappers
  * run ```bash ./get_mappers.sh```
    ```
    models_ckpts
      ├── flame
      │   ├── expression
      │   │   ├── flame_expression.ckpt
      │   │   └── flame_expression_metadata.json
      │   └── shape
      │       ├── flame_shape.ckpt
      │       └── flame_shape_metadata.json
      ├── smal
      │   ├── smal.ckpt
      │   └── smal_metadata.json
      ├── smpl
      │   ├── smpl_female.ckpt
      │   ├── smpl_female_metadata.json
      │   ├── smpl_male.ckpt
      │   ├── smpl_male_metadata.json
      │   ├── smpl_neutral.ckpt
      │   └── smpl_neutral_metadata.json
      └── smplx
          ├── smplx_female.ckpt
          ├── smplx_female_metadata.json
          ├── smplx_male.ckpt
          ├── smplx_male_metadata.json
          ├── smplx_neutral.ckpt
          └── smplx_neutral_metadata.json
    ```
    
### 3D Morphable Models
* **FLAME** [Li et al. 2017]
  * Register to [FLAME](https://flame.is.tue.mpg.de/) and download the following:<br>
    - FLAME 2020
    - FLAME texture space
    - Download [static landmarks embedding](https://github.com/TimoBolkart/TF_FLAME/blob/master/data/flame_static_embedding_68.pkl)
  * Then organize the files in ```semantify/assets/flame``` as follows:
      ```
      ├── flame
          ├── female_model.pkl
          ├── flame.py
          ├── flame_static_embedding_68.pkl
          ├── ft.npy
          ├── generic_model.pkl
          ├── male_model.pkl
          ├── mean.npy
          └── vt.npy
      ```
* **SMPL** [Loper et al. 2015]
  * Register to [SMPL](https://smpl.is.tue.mpg.de/) and download the following:<br>
    - Download version 1.1.0 for python 2.7
    - Download UV map in OBJ format
    - Textures could be downloaded from [MeshCapade](https://github.com/Meshcapade/SMPL_texture_samples/tree/main/Textured_Mesh_samples)
  * Then organize the files in ```semantify/assets/smpl``` as follows:
      ```
      ├── smpl
          ├── SMPL_FEMALE.pkl
          ├── SMPL_MALE.pkl
          ├── SMPL_NEUTRAL.pkl
          ├── smpl_texture.png
          └── smpl_uv.obj
      ```

* **SMPL-X** [Pavlakos et al. 2019]
  * Register to [SMPLX](https://smpl-x.is.tue.mpg.de/) and download the following:<br>
    - SMPL-X v1.1 (NPZ+PKL, 830MB)
    - Textures could be downloaded from [qzane](https://github.com/qzane/textured_smplx/blob/main/data/obj1/texture_smplx.png) or using the SMPL texture which will not fit perfectly, but our method works well with it.<br>
  * Then organize the files in ```semantify/assets/smplx``` as follows:
      ```
      ├── smplx
          ├── a_pose.npy
          ├── SMPLX_FEMALE.npz
          ├── SMPLX_MALE.npz
          ├── SMPLX_NEUTRAL.npz
          └── smplx_texture.png
      ```
* **SMAL** [Zuffi et al. 2017]
  * Register to [SMAL](https://smal.is.tue.mpg.de/) and download the following:<br>
    - SMAL model version 1.0.0
  * Then get ```smal_CVPR2017.pkl``` file and place it in ```semantify/assets/smal```
      ```
      ├── smal
          ├── smal_CVPR2017.pkl
          └── smal_layer.py
      ```
### Pose Estimation
  * **SPIN** [Kolotouros et al. 2019]
    * Clone [SPIN](https://github.com/nkolot/SPIN) repo, then run:
      ```
      cd SPIN
      ./fetch_data.sh
      ```
    * Then copy ```SPIN/data/smpl_mean_params.npz``` and ```SPIN/data/model_checkpoint.pt``` to ```semantify/assets/spin```
      ```
      └── spin
          ├── model_checkpoint.pt
          ├── smpl_mean_params.npz
          └── spin_model.py
      ```


## Run Demos
### Sliders Application
```
python semantify/applications/sliders_demo_py3d.py --model_type <MODELTYPE> --specific <SPECIFIC>
```
<ins>Arguments Description:</ins>
* ```model_type (str)``` - "smplx" | "smpl" | "flame" | "smal"
* ```specific (str)``` - "male" | "female" | "neutral" | "expression" | "shape"
* ```image_path (Optional[str])``` - If you want to run the demo on an image, provide the path to the image
* ```mapper_path (Optional[str])``` - If you don't want to use Semantify's mappers, set the path to desired ckpt file
* ```use_raw_blendshapes (Optional)``` - Use the raw parametric blendshapes of the model
* ```out_dir (Optional[str])``` - Path of directory to save outputs in<br>
for more optional arguments please visit ```semantify/applications/sliders_demo_py3d.py```

<p align="center">
    <img src="https://github.com/Omergral/Semantify/blob/master/static/images/semantify_smplx_male_app.gif" alt="example" width=50%>
</p>


### Image-to-Shape
```
python semantify/applications/image2shape.py --images_paths <IMAGES_PATHS> --model_type <TYPE> --specific <SPECIFIC> --output_path <PATH>
```

<ins>Arguments Description</ins>
* ```images_paths (List[str])``` - Paths to the images you wish to run the demo on
* ```model_type (str)``` - "smplx" | "smpl" | "flame" | "smal"
* ```specific (str)``` - "male" | "female" | "neutral" | "expression" | "shape"
* ```mapper_path (Optional[str])``` - If you don't want to use Semantify's mappers, set the path to desired ckpt file

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
@InProceedings{Gralnik_2023_ICCV,
    author    = {Gralnik, Omer and Gafni, Guy and Shamir, Ariel},
    title     = {Semantify: Simplifying the Control of 3D Morphable Models Using CLIP},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {14554-14564}
}
```

## License

The Semantify codebase is released under the [MIT License](LICENSE).

### Third-party code

The following files included in this repository are from third-party sources and retain their original licenses:

| File | Source | License |
|---|---|---|
| `semantify/assets/flame/flame.py` | [soubhiksanyal/FLAME_PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch) | MIT |
| `semantify/assets/smal/smal_layer.py` | [silviazuffi/smalst](https://github.com/silviazuffi/smalst) | MIT |
| `semantify/assets/spin/spin_model.py` | [nkolot/SPIN](https://github.com/nkolot/SPIN) | BSD 3-Clause |

### Third-party model weights

The 3D morphable model weights used by Semantify are **not** included in this repository and are subject to their own separate licenses. You must register and agree to each model's terms before downloading:

- **FLAME** — [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/)
- **SMPL** — [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/)
- **SMPL-X** — [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/)
- **SMAL** — [smal.is.tue.mpg.de](https://smal.is.tue.mpg.de/)
