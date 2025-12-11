# util/image_pool.py
# TODO Nov 01 hardened ImagePool

import random
from typing import List, Optional
import torch


class ImagePool:
    """
    Image replay buffer for GAN training.
    Stores previously generated *batches* and, on query(), returns a mix of
    old and current samples to stabilize D learning.

    - Works with tensors of shape (B, ...) on any device/dtype.
    - Detaches stored tensors from the graph to avoid GPU memory leaks.
    - Probabilistic swap controlled by shuffle_odds (default 0.5).
    - Checkpointable via state_dict()/load_state_dict().
    """

    def __init__(self, pool_size: int, shuffle_odds: float = 0.5):
        """
        Args:
            pool_size: max number of items stored. If 0, buffer is disabled.
            shuffle_odds: probability of returning a past sample (and swapping)
                          instead of the current one.
        """
        self.pool_size = int(pool_size)
        self.shuffle_odds = float(shuffle_odds)
        self.num_imgs = 0
        self.images: List[torch.Tensor] = []

    def __len__(self) -> int:
        return self.num_imgs

    def clear(self):
        """Remove all cached items."""
        self.images.clear()
        self.num_imgs = 0

    def state_dict(self):
        """Minimal checkpoint (stores CPU copies to keep checkpoints portable)."""
        return {
            "pool_size": self.pool_size,
            "shuffle_odds": self.shuffle_odds,
            "num_imgs": self.num_imgs,
            "images": [t.detach().cpu() for t in self.images],
        }

    def load_state_dict(self, state):
        self.pool_size = int(state.get("pool_size", self.pool_size))
        self.shuffle_odds = float(state.get("shuffle_odds", self.shuffle_odds))
        self.images = [t.clone() for t in state.get("images", [])]
        self.num_imgs = min(len(self.images), self.pool_size)

    @torch.no_grad()
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) or (B, C, D, H, W) etc.

        Returns:
            Tensor of same shape and device as `images`.
        """
        if self.pool_size == 0:
            # Pass-through when disabled
            return images

        if not torch.is_tensor(images):
            raise TypeError("ImagePool.query expects a torch.Tensor batch.")

        # We will build a list of per-item tensors to stack at the end
        b = images.shape[0]
        device = images.device
        out_items: List[torch.Tensor] = []

        for i in range(b):
            item = images[i].detach()  # drop graph for storage; keeps device/dtype

            if self.num_imgs < self.pool_size:
                # Pool still filling: store and return the current item
                self.images.append(item.clone())
                self.num_imgs += 1
                out_items.append(item)
            else:
                if random.random() < self.shuffle_odds:
                    # Return a random past item, replace it with current one
                    ridx = random.randint(0, self.pool_size - 1)
                    old = self.images[ridx]
                    self.images[ridx] = item.clone()
                    out_items.append(old.to(device=device, dtype=item.dtype))
                else:
                    # Return current item (no swap)
                    out_items.append(item)

        return torch.stack(out_items, dim=0)
