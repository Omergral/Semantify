import torch


class SMPLXParams:
    def __init__(
        self,
        betas: torch.Tensor = None,
        expression: torch.Tensor = None,
        body_pose: torch.Tensor = None,
        global_orient: torch.Tensor = None,
        transl: torch.Tensor = None,
        smpl_model: bool = False,
        num_coeffs: int = 10,
    ):
        if betas is not None:
            betas: torch.Tensor = betas
        else:
            betas = torch.zeros(1, num_coeffs)
        if expression is not None:
            expression: torch.Tensor = expression
        else:
            expression: torch.Tensor = torch.zeros(1, 10)
        if body_pose is not None:
            body_pose: torch.Tensor = body_pose
        else:
            body_pose = torch.eye(3).expand(1, 21, 3, 3)
        left_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        right_hand_pose: torch.Tensor = torch.eye(3).expand(1, 15, 3, 3)
        if global_orient is not None:
            global_orient: torch.Tensor = global_orient
        else:
            global_orient: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        if transl is not None:
            transl: torch.Tensor = transl
        else:
            transl: torch.Tensor = torch.zeros(1, 3)
        jaw_pose: torch.Tensor = torch.eye(3).expand(1, 1, 3, 3)
        if smpl_model:
            self.params = {"betas": betas, "body_pose": torch.cat([body_pose, torch.eye(3).expand(1, 2, 3, 3)], dim=1)}
        else:
            self.params = {
                "betas": betas,
                "body_pose": body_pose,
                "left_hand_pose": left_hand_pose,
                "right_hand_pose": right_hand_pose,
                "global_orient": global_orient,
                "transl": transl,
                "jaw_pose": jaw_pose,
                "expression": expression,
            }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}



class FLAMEParams:
    def __init__(
        self,
        shape_params: torch.tensor = None,
        expression_params: torch.tensor = None,
        jaw_pose: torch.Tensor = None,
    ):
        if shape_params is None:
            shape_params = torch.zeros(1, 100, dtype=torch.float32)
        if expression_params is None:
            expression_params = torch.zeros(1, 50, dtype=torch.float32)
        shape_params = shape_params.cuda()
        expression_params = expression_params.cuda()
        if jaw_pose is None:
            pose_params_t = torch.zeros(1, 6, dtype=torch.float32)
        else:
            pose_params_t = torch.cat([torch.zeros(1, 3), torch.tensor([[jaw_pose, 0.0, 0.0]])], 1)
        pose_params = pose_params_t.cuda()
        self.params = {
            "shape_params": shape_params,
            "expression_params": expression_params,
            "pose_params": pose_params,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}


class SMALParams:
    def __init__(self, beta: torch.tensor = None):
        if beta is None:
            beta = torch.zeros(1, 41, dtype=torch.float32)
        beta = beta.cuda()
        theta = torch.eye(3).expand(1, 35, 3, 3).to("cuda")
        self.params = {
            "beta": beta,
            "theta": theta,
        }

    def to(self, device):
        return {param_name: param.to(device) for param_name, param in self.params.items()}
