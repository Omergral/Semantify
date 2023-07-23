'''
    This is a short demo to see how to load and use the SMAL model.
    Please read the README.txt file for requirements.

'''

from smpl_webuser.serialization import load_model
from my_mesh.mesh import myMesh
import pickle as pkl

# Load the smal model 
model_path = 'smal_CVPR2017.pkl'
model = load_model(model_path)

# Save the mean model
# model.r are the model vertexes, and model.f are the mesh faces.
m = myMesh(v=model.r, f=model.f)
m.save_ply('smal_mean_shape.ply')
print 'saved mean shape'

# Load the family clusters data (see paper for details)
# and save the mean per-family shape
# 0-felidae(cats); 1-canidae(dogs); 2-equidae(horses);
# 3-bovidae(cows); 4-hippopotamidae(hippos);
# The clusters are over the shape coefficients (betas);
# setting different betas changes the shape of the model
model_data_path = 'smal_CVPR2017_data.pkl'
data = pkl.load(open(model_data_path))
for i, betas in enumerate(data['cluster_means']):
    model.betas[:] = betas
    model.pose[:] = 0.
    model.trans[:] = 0.
    m = myMesh(v=model.r, f=model.f)
    m.save_ply('family_'+str(i)+'.ply')
print 'saved mean per-family shape'

# Show the toys in T-pose
for i, betas in enumerate(data['toys_betas']):
    model.betas[:] = betas
    model.pose[:] = 0.
    model.trans[:] = 0.
    m = myMesh(v=model.r, f=model.f)
    m.save_ply('toy_'+str(i)+'.ply')
print 'saved toys in t-pose'


