import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CBAdaptiveFocalLoss(nn.Module):
    """Class‑Balanced Focal Loss with cosine‑scheduled γ.

    This implementation supports both binary and multi‑class classification:
    * **Binary** – ``logits`` may be raw scores or sigmoid probabilities.
    * **Multi‑class** – ``logits`` should be raw scores (softmax applied inside).

    Parameters
    ----------
    class_counts : List[int]
        Number of training samples for each class, ordered exactly the same as
        the model’s output dimension.
    total_epochs : int
        Number of epochs used for cosine scheduling of γ.
    beta : float, default 0.999
        Hyper‑parameter for *effective‑number* class re‑weighting.
    gamma_min : float, default 0.5
        Initial focusing parameter at the very first epoch.
    gamma_max : float, default 2.5
        Final focusing parameter reached at the last epoch.
    eps : float, default 1e‑8
        Numerical stability constant.
    """

    def __init__(
        self,
        class_counts: List[int],
        total_epochs: int,
        beta: float = 0.999,
        gamma_min: float = 0.5,
        gamma_max: float = 2.5,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
        effective_num = 1.0 - torch.pow(torch.tensor(beta, device=device), counts)
        alpha = (1.0 - beta) / (effective_num + eps)  # shape [C]
        self.register_buffer("alpha_buf", alpha)  # [C]

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.total_epochs = total_epochs
        self.eps = eps
        self.cur_epoch = 0

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def step_epoch(self) -> None:
        """Call once at the **end** of every epoch to update γ."""
        self.cur_epoch += 1

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    @property
    def _gamma(self) -> float:
        """Cosine‑scheduled γ for the *current* epoch."""
        ratio = (1.0 - math.cos(math.pi * self.cur_epoch / self.total_epochs)) / 2.0
        return self.gamma_min + (self.gamma_max - self.gamma_min) * ratio

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        logits : torch.Tensor
            * Binary*: shape ``[B]`` or ``[B,1]``; raw logits **or** sigmoid probs.
            * Multi‑class*: shape ``[B,C]``; raw logits.
        target : torch.Tensor
            Long tensor of shape ``[B]`` with class indices.
        sample_weight : torch.Tensor, optional
            Extra per‑sample weights (e.g. label‑aware dynamic weights). Shape
            ``[B]`` or ``[B,1]``.
        """
        if logits.dim() == 1 or logits.shape[1] == 1:
            # ------------------------------ Binary ----------------------
            p = torch.sigmoid(logits).clamp_(self.eps, 1.0 - self.eps)  # [B]
            pt = torch.where(target == 1, p, 1.0 - p)  # [B]

            if self.alpha_buf.numel() == 1:  # safety: only one class weight
                alpha_t = self.alpha_buf[0]
            else:
                alpha_t = torch.where(target == 1, self.alpha_buf[1], self.alpha_buf[0])
        else:
            # --------------------------- Multi‑class ---------------------
            p = F.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)  # [B,C]
            pt = p[torch.arange(target.size(0), device=logits.device), target]  # [B]
            alpha_t = self.alpha_buf[target]  # [B]

        loss = -alpha_t * ((1.0 - pt) ** self._gamma) * pt.log()

        if sample_weight is not None:
            loss = loss * sample_weight.view(-1)

        return loss.mean()

