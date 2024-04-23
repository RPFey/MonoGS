import random
import time

import torch
from torch import Tensor
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
from typing import List, Dict
from queue import Queue
import cv2
import matplotlib.pyplot as plt

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping
from utils.camera_utils import Camera

import PyDBoW2 as bow

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.image.lpips import _NoTrainLpips, _LPIPS

def _normalize_tensor(in_feat: Tensor, eps: float = 1e-8):
    """Normalize input tensor."""
    norm_factor = torch.sqrt(eps + torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / norm_factor


def _resize_tensor(x: Tensor, size: int = 64):
    if x.shape[-1] > size and x.shape[-2] > size:
        return torch.nn.functional.interpolate(x, (size, size), mode="area")
    return torch.nn.functional.interpolate(x, (size, size), mode="bilinear", align_corners=False)

class LPIPExtractor(_LPIPS):
    def extract(self, in0, normalize=False):
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1

        # normalize input
        in0_input = self.scaling_layer(in0)

        # resize input if needed
        if self.resize is not None:
            in0_input = _resize_tensor(in0_input, size=self.resize)

        outs0 = self.net.forward(in0_input)
        feats0 = {}

        for kk in range(self.L):
            feats0[kk] = _normalize_tensor(outs0[kk])

        return feats0

    def train(self, mode: bool) -> "LPIPExtractor":  # type: ignore[override]
        """Force network to always be in evaluation mode."""
        return super().train(False)

class ORBLoopDetector:
    def __init__(self) -> None:
        self.extractor = bow.ORBextractor(1000, 1.2, 8, 20, 7) 
        self.voc = bow.OrbVocabulary()
        self.voc.loadFromTextFile("/home/wen/Projects/ORB_SLAM2/Vocabulary/ORBvoc.txt")

        param = bow.LoopDetectorParam(480, 640, 0.5, True, 0.15, 1, bow.LoopDetectorGeometricalCheck.GEOM_FLANN, 6)
        param.max_neighbor_ratio = 0.8

        self.loop_detector = bow.OrbLoopDetector(self.voc, param)
    
    def addImage(self, image: np.ndarray, image_id: int) -> None:
        # switch Axis if necessary
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
            image = np.ascontiguousarray(image)
        
        image = np.clip(image, a_min=0., a_max=1.)
        image = (image * 255).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        kp, des = self.extractor.extract(image, mask)
        des = np.array(des)

        result_pkg = self.loop_detector.detectLoop(kp, des, image_id)
        print("Result: ", result_pkg[0].status)

        if result_pkg[0].status == bow.LoopDetectorDetectionStatus.LOOP_DETECTED:
            result:bow.LoopDetectorResult = result_pkg[0]
            Log("Loop found: query {} match {}".format(result.query, result.match))
        
        return result_pkg[0]

class BRIEFLoopDetector:
    def __init__(self) -> None:
        self.extractor = bow.BRIEFextractor("/home/wen/Projects/MonoGS/resources/brief_pattern.yml") 
        self.voc = bow.BRIEFVocabulary("/home/wen/Projects/MonoGS/resources/brief_k10L6.voc")

        param = bow.BRIEFLoopDetectorParam(480, 640, .5, True, 0.15, 1, bow.LoopDetectorGeometricalCheck.GEOM_EXHAUSTIVE, 2)
        param.max_neighbor_ratio = 0.8
        param.max_distance_between_groups = 3
        param.max_distance_between_queries = 2

        self.loop_detector = bow.BRIEFLoopDetector(self.voc, param)
    
    def addImage(self, image: np.ndarray) -> None:
        # switch Axis if necessary
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
            image = np.ascontiguousarray(image)
        
        image = np.clip(image, a_min=0., a_max=1.)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        result_pkg = self.loop_detector.detectLoop(self.extractor, image)
        print("Result: ", result_pkg[0].status)

        if result_pkg[0].status == bow.LoopDetectorDetectionStatus.LOOP_DETECTED:
            result:bow.LoopDetectorResult = result_pkg[0]
            Log("Loop found: query {} match {}".format(result.query, result.match))
        
        return result_pkg

class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints: Dict[int, Camera] = {}
        self.current_window: List[int] = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.covisibility_score = self.config["Training"]["covi_score"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        self.covisible_method = "n_touch"
        self.loop_detection_method = "BoW"

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            # No random point
            # for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
            #     viewpoint = random_viewpoint_stack[cam_idx]
            #     render_pkg = render(
            #         viewpoint, self.gaussians, self.pipeline_params, self.background
            #     )
            #     (
            #         image,
            #         viewspace_point_tensor,
            #         visibility_filter,
            #         radii,
            #         depth,
            #         opacity,
            #         n_touched,
            #     ) = (
            #         render_pkg["render"],
            #         render_pkg["viewspace_points"],
            #         render_pkg["visibility_filter"],
            #         render_pkg["radii"],
            #         render_pkg["depth"],
            #         render_pkg["opacity"],
            #         render_pkg["n_touched"],
            #     )
            #     loss_mapping += get_loss_mapping(
            #         self.config, image, depth, viewpoint, opacity
            #     )
            #     viewspace_point_tensor_acm.append(viewspace_point_tensor)
            #     visibility_filter_acm.append(visibility_filter)
            #     radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        # init image descriptor extractor
        self.init_loop_detection_desc()
            
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")
                    
                    # add image
                    self.extractor.addImage(viewpoint.original_image.cpu().numpy())

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map)
                    
                    # Do Loop Detection
                    result, match_kps, query_kps = self.extractor.addImage(viewpoint.original_image.cpu().numpy())

                    # visualize the loop detection result
                    if result.status == bow.LoopDetectorDetectionStatus.LOOP_DETECTED:
                        self.visualize_loop_detection(result, match_kps, query_kps, cur_frame_idx)
                        
                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    local_window = self.construct_local_graph(cur_frame_idx, 0.2)
                    Log(" Local Map Window: ", local_window)

                    # extract frames in current window
                    for cam_num, cam_idx in enumerate(local_window):
                        if cam_idx == 0:
                            continue
                        viewpoint = self.viewpoints[cam_idx]
                        if cam_num < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5 if cur_frame_idx - cam_idx < 50 else 1e-4,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5 if cur_frame_idx - cam_idx < 50 else 1e-4,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01 if cur_frame_idx - cam_idx < 50 else 1e-3,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01 if cur_frame_idx - cam_idx < 50 else 1e-3,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(local_window, iters=iter_per_kf)
                    self.map(local_window, prune=True)                    
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return

    def init_loop_detection_desc(self):
        if self.loop_detection_method == "lpips":
            # lpips extractor
            self.extractor = LPIPExtractor(net="alex")
            self.extractor.to(self.device)

        elif self.loop_detection_method == "BoW":
            # Bag of Words extractor
            self.extractor = BRIEFLoopDetector()
        else:
            raise NotImplementedError("Method not implemented")
        
    def visualize_loop_detection(self, result:bow.DetectionResult, match_kps:List[bow.KeyPoint], 
                                 query_kps:List[bow.KeyPoint], cur_frame_idx:int):
        """ Visualize the loop detection result """
        keyframe_idx = list(self.viewpoints.keys())
        keyframe_idx.sort()

        query_idx = keyframe_idx[result.query]
        match_idx = keyframe_idx[result.match]

        query_img = self.viewpoints[query_idx].original_image.permute(1, 2, 0).cpu().numpy()
        match_img = self.viewpoints[match_idx].original_image.permute(1, 2, 0).cpu().numpy()

        # visualize the loop detection result
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query Image {query_idx}")
        axes[1].imshow(match_img)
        axes[1].set_title(f"Match Image {match_idx}")
        fig.savefig(f"loop_detection_{cur_frame_idx}.png")

        H, W = query_img.shape[:2]
        canvas = np.concatenate([query_img, match_img], axis=1)
        canvas = np.clip(canvas * 255, 0, 255).astype(np.uint8)
        for query_kp, match_kp in zip(query_kps, match_kps):
            # generate random color
            random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            canvas = cv2.circle(canvas, (int(query_kp.pt.x), int(query_kp.pt.y)), 5, random_color, 1)
            canvas = cv2.circle(canvas, (int(match_kp.pt.x) + W, int(match_kp.pt.y)), 5, random_color, 1)
            canvas = cv2.line(canvas, (int(query_kp.pt.x), int(query_kp.pt.y)), (int(match_kp.pt.x) + W, int(match_kp.pt.y)), random_color, 2)

        cv2.imwrite(f"loop_detection_{cur_frame_idx}_match.png", canvas)

    def pose_graph_optimization(self, local_graph=False):
        # optimize for the pose graph
        pass

    def loop_detection(self):
        """ Detect loops in the new keyframe """
        pass

    def ntouch_covisibility(self, f_idx1, f_idx2):
        # compute the covisibility score between two frames
        render_pkg1 = render(
            self.viewpoints[f_idx1], self.gaussians, self.pipeline_params, self.background
        )
        f1_visibility = (render_pkg1["n_touched"] > 0).long()

        render_pkg2 = render(
            self.viewpoints[f_idx2], self.gaussians, self.pipeline_params, self.background
        )
        f2_visibility = (render_pkg2["n_touched"] > 0).long()

        union = torch.logical_or(
            f1_visibility, f2_visibility
        ).count_nonzero()
        intersection = torch.logical_and(
            f1_visibility, f2_visibility
        ).count_nonzero()

        return intersection / union
    
    def ntouch_kl_covisibility(self, f_idx1, f_idx2):
        # compute the covisibility score between two frames
        # using the KL divergence of n_touch
        render_pkg1 = render(
            self.viewpoints[f_idx1], self.gaussians, self.pipeline_params, self.background
        )
        visibility_dist1 = render_pkg1["n_touch"].float() / render_pkg1["n_touched"].sum()

        render_pkg2 = render(
            self.viewpoints[f_idx2], self.gaussians, self.pipeline_params, self.background
        )
        visibility_dist2 = render_pkg2["n_touch"].float() / render_pkg2["n_touched"].sum()

        score = ( - visibility_dist2 * torch.log(visibility_dist1 / visibility_dist2)).sum()

        return score
    
    def viewdir_covisibility(self, f_idx1, f_idx2):
        """ Use the viewdir of gaussians as a measure of covisibility """
        pass
    
    def compute_covisibility(self, f_idx1, f_idx2):
        # compute the covisibility score between two frames
        if self.covisible_method == "n_touch":
            return self.ntouch_covisibility(f_idx1, f_idx2)
        else:
            raise NotImplementedError("Method not implemented")

    def construct_local_graph(self, cur_frame_idx, covisible_thresh=0.9):
        """ Construct the local graph for the current frame using the covisibility score """
        # construct the local graph
        local_window = set()
        local_window.add(cur_frame_idx)

        frame_idx = []
        score = []

        # store visited nodes
        visited = set()
        visited.add(cur_frame_idx)

        bfs_queue = Queue()
        for f_idx in self.current_window[1:]:
            bfs_queue.put(f_idx)
        
        while not bfs_queue.empty():
            f_idx = bfs_queue.get()
            visited.add(f_idx)
            
            covisibility = self.compute_covisibility(cur_frame_idx, f_idx)
            # record the covisibility
            self.viewpoints[cur_frame_idx].covisibile_frame_idx[f_idx] = covisibility.item()
            self.viewpoints[f_idx].covisibile_frame_idx[cur_frame_idx] = covisibility.item()
            
            if covisibility > covisible_thresh:
                if f_idx not in local_window:
                    local_window.add(f_idx)

                    frame_idx.append(f_idx)
                    score.append(covisibility.item())
                    
                    for f_idx2, covisibility in self.viewpoints[f_idx].covisibile_frame_idx.items():
                        if covisibility > covisible_thresh and f_idx2 not in visited:
                            bfs_queue.put(f_idx2)

        frame_idx = np.array(frame_idx)
        score = np.array(score)

         # sort the frames based on the covisibility score
        sorted_idx = np.argsort(score)[::-1]
        frame_idx = frame_idx[sorted_idx][:self.window_size]
        local_window = [cur_frame_idx] + frame_idx.tolist()

        frame_diff = cur_frame_idx - frame_idx
        loop_frame = frame_idx[frame_diff > 50]
        if len(loop_frame) > 0:
            Log("Loop detected: ", loop_frame)
            for l_frame in loop_frame:
                covisibility = [[f_idx, score] for f_idx, score in self.viewpoints[l_frame].covisibile_frame_idx.items()]
                covisibility.sort(key=lambda x: x[1], reverse=True)
                
                neighbor = 0
                for f_idx, score in covisibility:
                    if f_idx not in local_window:
                        local_window.append(f_idx)
                        neighbor += 1
                        if neighbor >= 1:
                            break
       
        return local_window
