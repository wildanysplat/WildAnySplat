import os
from copy import deepcopy
import time
from typing import Optional, Tuple, Union, List, Dict, Any
from einops import rearrange
import huggingface_hub
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass

from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg
from src.model.encoder.anysplat import EncoderAnySplat, EncoderAnySplatCfg, OpacityMappingCfg
from src.utils.bilateral_grids import BilateralGrid

class AnySplat(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(
        self,
        encoder_cfg: EncoderAnySplatCfg,
        decoder_cfg: DecoderSplattingCUDACfg,
    ):  
        super(AnySplat, self).__init__()
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.build_encoder(encoder_cfg)
        self.build_decoder(decoder_cfg)
        self.bg = BilateralGrid()

    def convert_nested_config(self, cfg_dict: dict, target_class: type):
        """Convert nested dictionary config to dataclass instance
        
        Args:
            cfg_dict: Configuration dictionary or already converted object
            target_class: Target dataclass type to convert to
            
        Returns:
            Instance of target_class
        """
        if isinstance(cfg_dict, dict):
            # Convert dict to dataclass
            return target_class(**cfg_dict)
        elif isinstance(cfg_dict, target_class):
            # Already converted, return as is
            return cfg_dict
        elif cfg_dict is None:
            # Handle None case
            return None
        else:
            raise ValueError(f"Cannot convert {type(cfg_dict)} to {target_class}")

    # def convert_config_recursively(self, cfg_obj, conversion_map: dict):
    #     """Convert nested configurations recursively using a conversion map
        
    #     Args:
    #         cfg_obj: Configuration object to convert
    #         conversion_map: Dict mapping field names to their target classes
    #                        e.g., {'gaussian_adapter': GaussianAdapterCfg}
        
    #     Returns:
    #         Converted configuration object
    #     """
    #     if not hasattr(cfg_obj, '__dict__'):
    #         return cfg_obj
            
    #     cfg_dict = cfg_obj.__dict__.copy()
        
    #     for field_name, target_class in conversion_map.items():
    #         if field_name in cfg_dict:
    #             cfg_dict[field_name] = self.convert_nested_config(
    #                 cfg_dict[field_name], 
    #                 target_class
    #             )
        
    #     # Return new instance of the same type
    #     return type(cfg_obj)(**cfg_dict)

    

    def convert_config_recursively(self, cfg_obj, conversion_map: dict):
        if isinstance(cfg_obj, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg_obj, resolve=False)
            for field_name, target_class in conversion_map.items():
                if field_name in cfg_dict:
                    cfg_dict[field_name] = self.convert_nested_config(cfg_dict[field_name], target_class)
            return OmegaConf.create(cfg_dict)

        if not hasattr(cfg_obj, '__dict__'):
            return cfg_obj

        cfg_dict = cfg_obj.__dict__.copy()
        for field_name, target_class in conversion_map.items():
            if field_name in cfg_dict:
                cfg_dict[field_name] = self.convert_nested_config(cfg_dict[field_name], target_class)

        return type(cfg_obj)(**cfg_dict)


    def convert_encoder_config(self, encoder_cfg: EncoderAnySplatCfg) -> EncoderAnySplatCfg:
        """Convert all nested configurations in encoder_cfg"""
        conversion_map = {
            'gaussian_adapter': GaussianAdapterCfg,
            'opacity_mapping': OpacityMappingCfg,
        }
        
        return self.convert_config_recursively(encoder_cfg, conversion_map)

    def build_encoder(self, encoder_cfg: EncoderAnySplatCfg):
        # Convert nested configurations using the helper method
        encoder_cfg = self.convert_encoder_config(encoder_cfg)
        self.encoder = EncoderAnySplat(encoder_cfg)

    def build_decoder(self, decoder_cfg: DecoderSplattingCUDACfg):
        self.decoder = DecoderSplattingCUDA(decoder_cfg)
    
    @torch.no_grad()
    def inference(self,
        context_image: torch.Tensor,
        visualization_dump: Optional[dict] = None,
    ):
        self.encoder.distill = False
        print("visualization_dump", visualization_dump, flush=True)
        if visualization_dump is None:
            encoder_output = self.encoder(context_image, global_step=0, visualization_dump=visualization_dump)
        else:
            encoder_output, visualization_dump = self.encoder(context_image, global_step=0, visualization_dump=visualization_dump)
        gaussians, pred_context_pose, bg_dict = encoder_output.gaussians, encoder_output.pred_context_pose, encoder_output.bg_dict

        if visualization_dump is not None:
            return gaussians, pred_context_pose, bg_dict, visualization_dump
        else:
            return gaussians, pred_context_pose, bg_dict
    
    def _bg_scale(self, global_step: int) -> float:
        vals = [0.0, 0.05, self.bg.s_max]
        steps = [0, 500, 2000]
        if global_step < steps[0]:
            return vals[0]
        if global_step < steps[1]:
            return vals[1]
        if global_step < steps[2]:
            t = (global_step - steps[1]) / (steps[2] - steps[1])
            return vals[1] + t * (vals[2] - vals[1])
        return vals[2]
    
    def forward(self, 
        context_image: torch.Tensor, # augmented for student
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        near: float = 0.01,
        far: float = 100.0,
        target_image: Optional[torch.Tensor] = None, # target novel views
    ):
        b, v, c, h, w = context_image.shape
        device = context_image.device
        # print("[forward] global_step", global_step, "context_image", context_image.shape)
        # if target_image is not None:
        #     print("[forward] target_image", target_image.shape)

        if visualization_dump is not None:
            encoder_output, visualization_dump = self.encoder(context_image, global_step, visualization_dump=visualization_dump, target_image=target_image)
        else:
            encoder_output = self.encoder(context_image, global_step, visualization_dump=visualization_dump, target_image=target_image)
        gaussians, pred_context_pose, bg_dict = encoder_output.gaussians, encoder_output.pred_context_pose, encoder_output.bg_dict
        output = self.decoder.forward(
            gaussians,
            pred_context_pose['extrinsic'], #c2w
            pred_context_pose["intrinsic"],
            torch.ones(1, v, device=device) * near,
            torch.ones(1, v, device=device) * far,
            (h, w),
            "depth",
        )

        # Apply bilateral grids
        # s = 0
        # if global_step > 0:
        #     s = self._bg_scale(global_step)
        s = 1.0
        output.color_aug, output.conf = self.bg.apply_bilateral_grids(bg_dict, output.color[:,:v], s=s)

        if visualization_dump is not None:
            return encoder_output, output, visualization_dump
        else:
            return encoder_output, output
    
