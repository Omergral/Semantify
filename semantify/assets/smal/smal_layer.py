"""
This script is taken from https://github.com/silviazuffi/smalst/tree/master/smal_model
This is an implemented Pytorch version of the SMAL model by [Zuffi et al. 2017]
"""


import numpy as np
import torch
from torch.autograd import Variable
import pickle as pkl
import torch.nn as nn


def batch_skew(vec, batch_size=None, opts=None):
    """
    vec is N x 3, batch_size is int
    returns N x 3 x 3. Skew_sym version of each matrix.
    """
    if batch_size is None:
        batch_size = vec.shape.as_list()[0]
    col_inds = torch.LongTensor([1, 2, 3, 5, 6, 7])
    indices = torch.reshape(torch.reshape(torch.arange(0, batch_size) * 9, [-1, 1]) + col_inds, [-1, 1])
    updates = torch.reshape(
        torch.stack([-vec[:, 2], vec[:, 1], vec[:, 2], -vec[:, 0], -vec[:, 1], vec[:, 0]], dim=1), [-1]
    )
    out_shape = [batch_size * 9]
    res = torch.Tensor(np.zeros(out_shape[0])).to(device=vec.device)
    res[np.array(indices.flatten())] = updates
    res = torch.reshape(res, [batch_size, 3, 3])

    return res


def batch_rodrigues(theta, opts=None):
    """
    Theta is Nx3
    """
    batch_size = theta.shape[0]

    angle = (torch.norm(theta + 1e-8, p=2, dim=1)).unsqueeze(-1)
    r = (torch.div(theta, angle)).unsqueeze(-1)

    angle = angle.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    outer = torch.matmul(r, r.transpose(1, 2))

    eyes = torch.eye(3).unsqueeze(0).repeat([batch_size, 1, 1]).to(device=theta.device)
    H = batch_skew(r, batch_size=batch_size, opts=opts)
    R = cos * eyes + (1 - cos) * outer + sin * H

    return R


def batch_lrotmin(theta):
    """
    Output of this is used to compute joint-to-pose blend shape mapping.
    Equation 9 in SMPL paper.
    Args:
      pose: `Tensor`, N x 72 vector holding the axis-angle rep of K joints.
            This includes the global rotation so K=24
    Returns
      diff_vec : `Tensor`: N x 207 rotation matrix of 23=(K-1) joints with identity subtracted.,
    """
    # Ignore global rotation
    theta = theta[:, 3:]

    Rs = batch_rodrigues(torch.reshape(theta, [-1, 3]))
    lrotmin = torch.reshape(Rs - torch.eye(3), [-1, 207])

    return lrotmin


def batch_global_rigid_transformation(Rs, Js, parent, rotate_base=False, betas_logscale=None, opts=None):
    """
    Computes absolute joint locations given pose.
    rotate_base: if True, rotates the global rotation by 90 deg in x axis.
    if False, this is the original SMPL coordinate.
    Args:
      Rs: N x 24 x 3 x 3 rotation vector of K joints
      Js: N x 24 x 3, joint locations before posing
      parent: 24 holding the parent id for each index
    Returns
      new_J : `Tensor`: N x 24 x 3 location of absolute joints
      A     : `Tensor`: N x 24 4 x 4 relative joint transformations for LBS.
    """
    if rotate_base:
        print("Flipping the SMPL coordinate frame!!!!")
        rot_x = torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        rot_x = torch.reshape(torch.repeat(rot_x, [N, 1]), [N, 3, 3])  # In tf it was tile
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]

    # Now Js is N x 24 x 3 x 1
    Js = Js.unsqueeze(-1)
    N = Rs.shape[0]

    Js_orig = Js.clone()

    scaling_factors = torch.ones(N, parent.shape[0], 3).to(Rs.device)
    if betas_logscale is not None:
        leg_joints = list(range(7, 11)) + list(range(11, 15)) + list(range(17, 21)) + list(range(21, 25))
        tail_joints = list(range(25, 32))
        ear_joints = [33, 34]

        beta_scale_mask = torch.zeros(35, 3, 6).to(betas_logscale.device)
        beta_scale_mask[leg_joints, [2], [0]] = 1.0  # Leg lengthening
        beta_scale_mask[leg_joints, [0], [1]] = 1.0  # Leg fatness
        beta_scale_mask[leg_joints, [1], [1]] = 1.0  # Leg fatness

        beta_scale_mask[tail_joints, [0], [2]] = 1.0  # Tail lengthening
        beta_scale_mask[tail_joints, [1], [3]] = 1.0  # Tail fatness
        beta_scale_mask[tail_joints, [2], [3]] = 1.0  # Tail fatness

        beta_scale_mask[ear_joints, [1], [4]] = 1.0  # Ear y
        beta_scale_mask[ear_joints, [2], [5]] = 1.0  # Ear z

        beta_scale_mask = torch.transpose(beta_scale_mask.reshape(35 * 3, 6), 0, 1)

        betas_scale = torch.exp(betas_logscale @ beta_scale_mask)
        scaling_factors = betas_scale.reshape(-1, 35, 3)

    scale_factors_3x3 = torch.diag_embed(scaling_factors, dim1=-2, dim2=-1)

    def make_A(R, t):
        # Rs is N x 3 x 3, ts is N x 3 x 1
        R_homo = torch.nn.functional.pad(R, (0, 0, 0, 1, 0, 0))
        t_homo = torch.cat([t, torch.ones([N, 1, 1]).to(Rs.device)], 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]
    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]

        s_par_inv = torch.inverse(scale_factors_3x3[:, parent[i]])
        rot = Rs[:, i]
        s = scale_factors_3x3[:, i]

        rot_new = s_par_inv @ rot @ s

        A_here = make_A(rot_new, j_here)
        res_here = torch.matmul(results[parent[i]], A_here)

        results.append(res_here)

    # 10 x 24 x 4 x 4
    results = torch.stack(results, dim=1)

    # scale updates
    new_J = results[:, :, :3, 3]

    # --- Compute relative A: Skinning is based on
    # how much the bone moved (not the final location of the bone)
    # but (final_bone - init_bone)
    # ---
    Js_w0 = torch.cat([Js_orig, torch.zeros([N, 33, 1, 1]).to(Rs.device)], 2)
    init_bone = torch.matmul(results, Js_w0)
    # Append empty 4 x 3:
    init_bone = torch.nn.functional.pad(init_bone, (3, 0, 0, 0, 0, 0, 0, 0))
    A = results - init_bone

    return new_J, A


# There are chumpy variables so convert them to numpy.
def undo_chumpy(x):
    return x if isinstance(x, np.ndarray) else x.r


class SMAL(nn.Module):
    def __init__(self, device, pkl_path, shape_family_id=-1, dtype=torch.float):
        super(SMAL, self).__init__()

        with open(pkl_path, "rb") as f:
            dd = pkl.load(f, encoding="latin1")

        self.f = dd["f"]

        self.faces = torch.from_numpy(self.f.astype(int)).to(device)

        v_template = dd["v_template"]

        # Size of mesh [Number of vertices, 3]
        self.size = [v_template.shape[0], 3]
        self.num_betas = dd["shapedirs"].shape[-1]
        # Shape blend shape basis

        shapedir = np.reshape(undo_chumpy(dd["shapedirs"]), [-1, self.num_betas]).T.copy()
        self.shapedirs = Variable(torch.Tensor(shapedir), requires_grad=False).to(device)

        if shape_family_id != -1:
            with open(pkl_path, "rb") as f:
                u = pkl._Unpickler(f)
                u.encoding = "latin1"
                data = u.load()

            # Select mean shape for quadruped type
            betas = data["cluster_means"][shape_family_id]
            v_template = v_template + np.matmul(betas[None, :], shapedir).reshape(-1, self.size[0], self.size[1])[0]

        v_sym, self.left_inds, self.right_inds, self.center_inds = align_smal_template_to_symmetry_axis(v_template)

        # Mean template vertices
        self.v_template = Variable(torch.Tensor(v_sym), requires_grad=False).to(device)

        # Regressor for joint locations given shape
        self.J_regressor = Variable(torch.Tensor(dd["J_regressor"].T.todense()), requires_grad=False).to(device)

        # Pose blend shape basis
        num_pose_basis = dd["posedirs"].shape[-1]

        posedirs = np.reshape(undo_chumpy(dd["posedirs"]), [-1, num_pose_basis]).T
        self.posedirs = Variable(torch.Tensor(posedirs), requires_grad=False).to(device)

        # indices of parents for each joints
        self.parents = dd["kintree_table"][0].astype(np.int32)

        # LBS weights
        self.weights = Variable(torch.Tensor(undo_chumpy(dd["weights"])), requires_grad=False).to(device)

    def __call__(self, beta, theta, trans=None, del_v=None, betas_logscale=None, get_skin=True, v_template=None):

        if True:
            nBetas = beta.shape[1]
        else:
            nBetas = 0

        # v_template = self.v_template.unsqueeze(0).expand(beta.shape[0], 3889, 3)
        if v_template is None:
            v_template = self.v_template

        # 1. Add shape blend shapes

        if nBetas > 0:
            if del_v is None:
                v_shaped = v_template + torch.reshape(
                    torch.matmul(beta, self.shapedirs[:nBetas, :]), [-1, self.size[0], self.size[1]]
                )
            else:
                v_shaped = (
                    v_template
                    + del_v
                    + torch.reshape(torch.matmul(beta, self.shapedirs[:nBetas, :]), [-1, self.size[0], self.size[1]])
                )
        else:
            if del_v is None:
                v_shaped = v_template.unsqueeze(0)
            else:
                v_shaped = v_template + del_v

        # 2. Infer shape-dependent joint locations.
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        # 3. Add pose blend shapes
        # N x 24 x 3 x 3
        if len(theta.shape) == 4:
            Rs = theta
        else:
            Rs = torch.reshape(batch_rodrigues(torch.reshape(theta, [-1, 3])), [-1, 35, 3, 3])

        # Ignore global rotation.
        # pose_feature = torch.reshape(Rs[:, 1:, :, :] - torch.eye(3).to(beta.device), [-1, 288])
        pose_feature = torch.reshape(torch.zeros(1, 32, 3, 3), [-1, 288]).cuda()

        v_posed = torch.reshape(torch.matmul(pose_feature, self.posedirs), [-1, self.size[0], self.size[1]]) + v_shaped

        # 4. Get the global joint location
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, betas_logscale=betas_logscale)

        # 5. Do skinning:
        num_batch = theta.shape[0]

        weights_t = self.weights.repeat([num_batch, 1])
        W = torch.reshape(weights_t, [num_batch, -1, 33])

        T = torch.reshape(torch.matmul(W, torch.reshape(A, [num_batch, 33, 16])), [num_batch, -1, 4, 4])
        v_posed_homo = torch.cat([v_posed, torch.ones([num_batch, v_posed.shape[1], 1]).to(device=beta.device)], 2)
        v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))

        verts = v_homo[:, :, :3, 0]

        if trans is None:
            trans = torch.zeros((num_batch, 3)).to(device=beta.device)

        verts = verts + trans[:, None, :]

        # Get joints:
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        joints = torch.cat(
            [
                joints,
                verts[:, None, 1863],  # end_of_nose
                verts[:, None, 26],  # chin
                verts[:, None, 2124],  # right ear tip
                verts[:, None, 150],  # left ear tip
                verts[:, None, 3055],  # left eye
                verts[:, None, 1097],  # right eye
            ],
            dim=1,
        )

        if get_skin:
            return verts, joints, Rs, v_shaped
        else:
            return joints


def align_smal_template_to_symmetry_axis(v):
    # These are the indexes of the points that are on the symmetry axis
    I = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        37,
        55,
        119,
        120,
        163,
        209,
        210,
        211,
        213,
        216,
        227,
        326,
        395,
        452,
        578,
        910,
        959,
        964,
        975,
        976,
        977,
        1172,
        1175,
        1176,
        1178,
        1194,
        1243,
        1739,
        1796,
        1797,
        1798,
        1799,
        1800,
        1801,
        1802,
        1803,
        1804,
        1805,
        1806,
        1807,
        1808,
        1809,
        1810,
        1811,
        1812,
        1813,
        1814,
        1815,
        1816,
        1817,
        1818,
        1819,
        1820,
        1821,
        1822,
        1823,
        1824,
        1825,
        1826,
        1827,
        1828,
        1829,
        1830,
        1831,
        1832,
        1833,
        1834,
        1835,
        1836,
        1837,
        1838,
        1839,
        1840,
        1842,
        1843,
        1844,
        1845,
        1846,
        1847,
        1848,
        1849,
        1850,
        1851,
        1852,
        1853,
        1854,
        1855,
        1856,
        1857,
        1858,
        1859,
        1860,
        1861,
        1862,
        1863,
        1870,
        1919,
        1960,
        1961,
        1965,
        1967,
        2003,
    ]

    v = v - np.mean(v)
    y = np.mean(v[I, 1])
    v[:, 1] = v[:, 1] - y
    v[I, 1] = 0

    left = v[:, 1] < 0
    right = v[:, 1] > 0
    center = v[:, 1] == 0
    v[left[2]] = np.array([1, -1, 1]) * v[0]

    left_inds = np.where(left)[0]
    right_inds = np.where(right)[0]
    center_inds = np.where(center)[0]

    try:
        assert len(left_inds) == len(right_inds)
    except:
        import pdb

        pdb.set_trace()

    return v, left_inds, right_inds, center_inds


def get_smal_layer(device: str, pkl_path: str):

    smal_layer = SMAL(device, pkl_path)
    return smal_layer
