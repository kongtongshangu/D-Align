import logging

import torch
import torch.nn as nn
import torch.optim as optim

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from adapt_algorithm import CorA
import d_align
from conf import cfg, load_cfg_fom_args
from utils.visual_prompt import LoR_VP
import math
import numpy as np
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate(description):
    args = load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "D-Align":
        logger.info("test-time adaptation: D-Align")
        Visual_Promopt = LoR_VP(cfg)
        Visual_Promopt_1 = LoR_VP(cfg)
        model = setup_d_align(base_model, Visual_Promopt, Visual_Promopt_1)

    # evaluate on each severity and type of corruption in turn
    all_error = 0.0
    for ii, severity in enumerate(cfg.CORRUPTION.SEVERITY):
        for i_x, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            # reset adaptation for each combination of corruption x severity
            # note: for evaluation protocol, but not necessarily needed
            try:
                if i_x == 0:
                    model.reset()
                    logger.info("resetting model")
                else:
                    logger.warning("not resetting model")
            except:
                logger.warning("not resetting model")
            x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = D_Align_clean_accuracy(model, x_test, y_test, Visual_Promopt, cfg.TEST.BATCH_SIZE, None, logger)
            err = 1. - acc
            logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            all_error += err
    all_error = all_error / 15
    logger.info(f"Avg_error % [{cfg.MODEL.ADAPTATION}]: {all_error:.2%}")


def setup_optimizer_d_align(params, params_vp):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": cfg.OPTIM.LR},
                           {"params": params_vp, "lr": cfg.VP.VP_lr1}],
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": cfg.OPTIM.LR},
                          {"params": params_vp, "lr": cfg.VP.VP_lr1}],
                         lr=cfg.OPTIM.LR,
                         momentum=0.9,
                         dampening=0,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=True)
    else:
        raise NotImplementedError

def setup_d_align(model, visual_promopt, visual_promopt_1):
    model = d_align.configure_model(model)
    params, params_vp = d_align.collect_params(model, visual_promopt_1)
    optimizer = setup_optimizer_d_align(params, params_vp)
    d_align_model = d_align.D_Align(model, optimizer, visual_promopt, visual_promopt_1,
                           len_num_keep=cfg.OPTIM.KEEP,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC,
                           m = cfg.OPTIM.M,
                           n = cfg.OPTIM.N,
                           lamb = cfg.OPTIM.LAMB,
                           margin = cfg.OPTIM.MARGIN,
                           )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return d_align_model

def D_Align_clean_accuracy(model: nn.Module,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   visual_prompt = None,
                   batch_size: int = 100,
                   device: torch.device = None,
                   logger=None
                   ):
    if device is None:
        device = x.device
    acc = 0.
    n_batches = math.ceil(x.shape[0] / batch_size)

    label_counts = torch.bincount(y)
    num_classes = len(label_counts)
    proportion_vector = torch.zeros(num_classes)

    total_count = label_counts.sum().item()
    if total_count > 0:
        for i in range(num_classes):
            proportion_vector[i] = label_counts[i].item() / total_count
    memory_bank = MemoryBank(num_classes=num_classes, capacity_per_class=1)
    CorA_ = CorA(model, memory_bank, filter_K=cfg.CorA.filter_K_TCA, W_num_iterations=cfg.CorA.W_num_iterations,
                  W_lr=cfg.CorA.W_lr, VP_lr=cfg.VP.VP_lr)

    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x[counter * batch_size:(counter + 1) * batch_size].to(device)
            y_curr = y[counter * batch_size:(counter + 1) * batch_size].to(device)
            embeddings, logits, score = model(x_curr)
            embeddings = embeddings[:, 0, :]
            outputs_CorA = CorA_.calculate(num_classes=num_classes, embeddings_arr=embeddings, logits_arr=logits,
                                         score=score, proportion_vector=proportion_vector, visual_prompt=visual_prompt)
            acc += (outputs_CorA.max(1)[1] == y_curr).float().sum()
    return acc.item() / x.shape[0]

if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')
