import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import List, Tuple

from models.utils import augmentation_layers as augl


class LearnableAugPolicy(nn.Module):
    """
    Learnable policy over a pool of augmentations.

    - Maintains two categorical distributions (positives, negatives) over the
      same augmentation candidate list.
    - At each call, samples n_pos and n_neg transforms (with replacement)
      and applies them to the given anchor batch.
    - Provides log-probabilities for REINFORCE-style updates and entropy for
      regularization.
    """

    def __init__(
        self,
        aug_names: List[str],
        severity: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.aug_names = list(aug_names)
        self.severity = severity
        self.device = device

        aug_list = augl.get_augmentation_list()

        augs = []
        for aug_name in self.aug_names:
            for aug in aug_list:
                if aug.lower() == aug_name.lower().replace('_', ''):
                    augs.append(augl.return_aug(aug, severity=self.severity))
        # Build transform modules
        self.transforms = nn.ModuleList(augs)
        self.to(device)

        # Learnable logits for positive and negative policies
        n = len(self.aug_names)
        self.pos_logits = nn.Parameter(torch.zeros(n))
        self.neg_logits = nn.Parameter(torch.zeros(n))

        # Moving baseline for REINFORCE advantage
        self.register_buffer("baseline", torch.tensor(0.0))
        self.register_buffer("_baseline_initialized", torch.tensor(0, dtype=torch.uint8))

    @property
    def pos_probs(self) -> torch.Tensor:
        return F.softmax(self.pos_logits, dim=0)

    @property
    def neg_probs(self) -> torch.Tensor:
        return F.softmax(self.neg_logits, dim=0)

    def entropy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shannon entropy of positive/negative distributions."""
        p_pos = self.pos_probs
        p_neg = self.neg_probs
        # add small epsilon for numerical stability
        eps = 1e-12
        H_pos = -torch.sum(p_pos * torch.log(p_pos + eps))
        H_neg = -torch.sum(p_neg * torch.log(p_neg + eps))
        return H_pos, H_neg

    def update_baseline(self, loss_value: torch.Tensor, momentum: float = 0.9) -> None:
        """
        Update exponential moving average baseline using the provided scalar loss.
        """
        with torch.no_grad():
            if self._baseline_initialized.item() == 0:
                self.baseline.copy_(loss_value.detach())
                self._baseline_initialized.fill_(1)
            else:
                self.baseline.mul_(momentum).add_(loss_value.detach() * (1 - momentum))

    def _sample_indices(self, logits: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample k indices (with replacement) from a categorical with given logits.
        Returns (indices, log_prob_sum).
        """
        dist = Categorical(logits=logits)
        idx = dist.sample((k,)) if k > 1 else dist.sample().unsqueeze(0)
        logp = dist.log_prob(idx).sum()
        return idx, logp

    @torch.no_grad()
    def selected_names(self, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> Tuple[List[str], List[str]]:
        pnames = [self.aug_names[i] for i in pos_idx.view(-1).tolist()]
        nnames = [self.aug_names[i] for i in neg_idx.view(-1).tolist()]
        return pnames, nnames

    def forward(
        self,
        anchor: torch.Tensor,
        n_pos: int = 1,
        n_neg: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply sampled positive and negative transforms to the anchor batch.

        Returns:
          pos_views: (n_pos*B, C, H, W)
          neg_views: (n_neg*B, C, H, W)
          logp_pos:  scalar tensor (sum of log-probs of selected positives)
          logp_neg:  scalar tensor (sum of log-probs of selected negatives)
          pos_idx:   (n_pos,) tensor of selected indices
          neg_idx:   (n_neg,) tensor of selected indices
        """
        B = anchor.size(0)

        # Sample indices
        pos_idx, logp_pos = self._sample_indices(self.pos_logits, max(1, n_pos))
        neg_idx, logp_neg = self._sample_indices(self.neg_logits, max(1, n_neg))

        # Apply transforms
        pos_views_list = [self.transforms[i](anchor) for i in pos_idx]
        neg_views_list = [self.transforms[i](anchor) for i in neg_idx]

        pos_views = torch.cat(pos_views_list, dim=0) if len(pos_views_list) > 0 else anchor.new_empty((0,) + anchor.shape[1:])
        neg_views = torch.cat(neg_views_list, dim=0) if len(neg_views_list) > 0 else anchor.new_empty((0,) + anchor.shape[1:])

        return pos_views, neg_views, logp_pos, logp_neg, pos_idx, neg_idx

