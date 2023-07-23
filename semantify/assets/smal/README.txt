You have downloaded the model described in the paper:
Silvia Zuffi, Angjoo Kanazawa, David Jacobs, and Michael J. Black, "3D Menagerie: Modeling the 3D Shape and Pose of Animals," CVPR 2017.

CONTENT:

We provide:
   - the model (smal_CVPR2017.pkl)
   - additional model data (smal_CVPR2017_data.pkl)
   - demo code (demo.py)
   - visualization code (my_mesh directory)


REQUIREMENTS:

The model has been developed with Python version 2.7.
It requires the package pickle and the SMPL model to load.
SMPL can be obtained here:
http://smpl.is.tue.mpg.de/

It also requires methods to read and write ply files and for mesh visualization.

In the paper we have used a proprietary library for visualize and render 3D meshes. 
In order to distribute our work, we provide an alternative way for visualization.  
In particular to read/write 3D meshes the demo code uses the python-plyfile library (https://github.com/dranjan/python-plyfile) that you should download and install.
The demo does not support interactive visualization, but it saves the meshes in ply format that can be visualized with meshlab (http://meshlab.sourceforge.net/).


USE:

Download and unzip the software. 
Read and run the demo to see how to use the model.


Please send any questions to:
Silvia Zuffi: silvia.zuffi@tue.mpg.de or silvia@mi.imati.cnr.it






