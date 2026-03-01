"""Tests for compute_iou and dice_loss_from_logits (utils)."""
from __future__ import annotations

import torch

from utils import compute_iou, dice_loss_from_logits


def test_compute_iou_perfect_match() -> None:
    """When prediction matches target exactly, IoU should be 1.0."""
    # logits large positive -> sigmoid ~1, same as target 1
    B, C, D, H, W = 2, 1, 4, 4, 4
    target = torch.ones(B, C, D, H, W)
    logits = torch.ones(B, C, D, H, W) * 10.0  # sigmoid(10) ~ 1
    iou = compute_iou(logits, target, threshold=0.5)
    assert abs(iou - 1.0) < 1e-5


def test_compute_iou_no_overlap() -> None:
    """When prediction and target don't overlap, IoU should be 0."""
    B, C, D, H, W = 2, 1, 4, 4, 4
    target = torch.zeros(B, C, D, H, W)
    target[:, :, :2, :, :] = 1.0  # first half
    logits = torch.ones(B, C, D, H, W) * (-10.0)  # sigmoid(-10) ~ 0 -> pred all 0
    # pred=0, target has 1s -> intersection 0, union = target sum -> IoU 0
    iou = compute_iou(logits, target, threshold=0.5)
    assert abs(iou) < 1e-5


def test_compute_iou_half_overlap() -> None:
    """Fifty-fifty overlap gives IoU around 1/3 (intersection/union)."""
    B, C, D, H, W = 1, 1, 4, 4, 4
    # target: first 32 voxels = 1
    target = torch.zeros(B, C, D, H, W)
    target[:, :, 0, :, :] = 1.0
    # pred: same first 32 = 1 -> intersection 32, union 32, IoU 1.0
    # pred: second 32 = 1 -> intersection 0, union 64, IoU 0
    # pred: first 16 and second 16 -> intersection 16, union 48, IoU = 16/48 = 1/3
    pred_logits = torch.zeros(B, C, D, H, W)
    pred_logits[:, :, 0, :2, :] = 10.0  # first 2 rows of slice 0
    pred_logits[:, :, 1, :2, :] = 10.0  # first 2 rows of slice 1
    # target has slice 0 all 1 (16 voxels). pred has slice 0 two rows (8) + slice 1 two rows (8) = 16 pred 1s
    # intersection: slice 0 two rows = 8. union: 16 + 16 - 8 = 24. IoU = 8/24 = 1/3
    iou = compute_iou(pred_logits, target, threshold=0.5)
    assert 0.32 < iou < 0.34


def test_dice_loss_perfect_match() -> None:
    """When prediction matches target, dice loss should be ~0."""
    B, C, D, H, W = 2, 1, 4, 4, 4
    target = torch.ones(B, C, D, H, W)
    logits = torch.ones(B, C, D, H, W) * 10.0
    loss = dice_loss_from_logits(logits, target)
    assert loss.dim() == 0
    assert loss.item() < 0.01


def test_dice_loss_no_overlap() -> None:
    """When prediction and target don't overlap, dice loss should be ~1."""
    B, C, D, H, W = 2, 1, 4, 4, 4
    target = torch.zeros(B, C, D, H, W)
    target[:, :, :2, :, :] = 1.0
    logits = torch.ones(B, C, D, H, W) * (-10.0)  # pred all 0
    loss = dice_loss_from_logits(logits, target)
    assert loss.dim() == 0
    assert loss.item() > 0.99


def test_dice_loss_from_logits_gradient_flow() -> None:
    """Dice loss is differentiable (for training)."""
    logits = torch.randn(2, 1, 4, 4, 4, requires_grad=True)
    target = (torch.rand(2, 1, 4, 4, 4) > 0.5).float()
    loss = dice_loss_from_logits(logits, target)
    loss.backward()
    assert logits.grad is not None
