import unittest
import torch
import numpy as np

# Assuming strategy_train.py is in the same directory or accessible via PYTHONPATH
# and can be imported without side effects that break testing.
from strategy_train import info_nce_multi

class TestInfoNCEMulti(unittest.TestCase):

    def test_basic_case_low_loss(self):
        """Test with a simple case where loss is expected to be very small."""
        z_anchor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        # Ensure views are distinct enough for sim_max to work as expected
        z_views = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False]], dtype=torch.bool)
        tau = 0.1
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        
        # Manual calculation for this specific case:
        # sim = [[10.0, -10.0]]
        # sim_max = 10.0
        # sim_exp = [[exp(0), exp(-20)]] = [[1.0, exp(-20)]]
        # denom = 1.0 + exp(-20)
        # pos_sim_exp = 1.0
        # pos_count = 1.0
        # loss = -log(1.0 / (1.0 + exp(-20))) / 1.0 = log(1.0 + exp(-20))
        expected_loss_val = np.log(1.0 + np.exp(-20.0))
        self.assertAlmostEqual(loss.item(), expected_loss_val, delta=1e-9)

    def test_no_positive_samples_for_any_anchor(self):
        """Test case where pos_mask is all False. Expected loss is 0.0."""
        z_anchor = torch.randn(2, 4, dtype=torch.float32)
        z_views = torch.randn(2, 3, 4, dtype=torch.float32)
        pos_mask = torch.zeros((2, 3), dtype=torch.bool) # All False
        tau = 0.2
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        self.assertEqual(loss.item(), 0.0)

    def test_all_views_are_positive(self):
        """Test case where all views for an anchor are positive. Expected loss is 0.0."""
        z_anchor = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        z_views = torch.tensor([[[1.0, 0.0], [0.7, 0.7]]], dtype=torch.float32) # Both positive-like
        pos_mask = torch.tensor([[True, True]], dtype=torch.bool)
        tau = 0.1
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        # If all views are positive, num = sum(exp(sim_i/tau)), denom = sum(exp(sim_i/tau))
        # log(num/denom) = log(1) = 0.
        self.assertAlmostEqual(loss.item(), 0.0, delta=1e-7)

    def test_multiple_anchors_all_valid(self):
        """Test with multiple anchors, all having at least one positive view."""
        z_anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_views = torch.tensor([
            [[1.0, 0.0], [-1.0, 0.0]],  # For anchor 0
            [[0.0, 1.0], [0.0, -1.0]]   # For anchor 1
        ], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False], [True, False]], dtype=torch.bool)
        tau = 0.1
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        
        # Each anchor is like the 'test_basic_case_low_loss'
        expected_loss_per_anchor = np.log(1.0 + np.exp(-20.0))
        # Loss is mean over valid samples
        self.assertAlmostEqual(loss.item(), expected_loss_per_anchor, delta=1e-9)

    def test_one_valid_one_invalid_anchor(self):
        """Test with one anchor having positives, another having no positives."""
        z_anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        z_views = torch.tensor([
            [[1.0, 0.0], [-1.0, 0.0]],  # For anchor 0
            [[0.0, 1.0], [0.0, -1.0]]   # For anchor 1
        ], dtype=torch.float32)
        pos_mask = torch.tensor([[True, False], [False, False]], dtype=torch.bool) # Anchor 1 has no positives
        tau = 0.1
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        
        # Only anchor 0 contributes to the loss
        expected_loss_val = np.log(1.0 + np.exp(-20.0))
        self.assertAlmostEqual(loss.item(), expected_loss_val, delta=1e-9)

    def test_batch_size_one_view_one(self):
        """Test with B=1, K=1."""
        z_anchor = torch.randn(1, 10, dtype=torch.float32)
        z_views = torch.randn(1, 1, 10, dtype=torch.float32) # K=1
        tau = 0.2

        # Case 1: The single view is positive
        pos_mask_true = torch.tensor([[True]], dtype=torch.bool)
        loss_true_positive = info_nce_multi(z_anchor, z_views, pos_mask_true, tau)
        # If K=1 and it's positive, sim_exp/denom = 1, log(1)=0. Loss = 0.
        self.assertAlmostEqual(loss_true_positive.item(), 0.0, delta=1e-7)

        # Case 2: The single view is negative (no valid anchors)
        pos_mask_false = torch.tensor([[False]], dtype=torch.bool)
        loss_false_positive = info_nce_multi(z_anchor, z_views, pos_mask_false, tau)
        self.assertEqual(loss_false_positive.item(), 0.0) # valid_mask.sum() == 0

    def test_different_dimensions_and_non_negativity(self):
        """Test with larger, random dimensions and check for non-negative loss."""
        B, K, D = 4, 5, 64
        z_anchor = torch.randn(B, D, dtype=torch.float32)
        z_views = torch.randn(B, K, D, dtype=torch.float32)
        # Random pos_mask, ensuring at least one positive if possible for a more general test
        pos_mask = torch.randint(0, 2, (B, K), dtype=torch.bool)
        if not pos_mask.any(): # If all are False by chance
            pos_mask[0,0] = True # Make at least one positive to avoid trivial 0.0 loss
            
        tau = 0.15
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.ndim, 0) # Scalar tensor
        self.assertGreaterEqual(loss.item(), 0.0) # InfoNCE loss should be non-negative

    def test_output_device_cpu(self):
        """Test if output tensor is on the same device as input (CPU)."""
        device = torch.device("cpu")
        z_anchor = torch.randn(2, 4, device=device, dtype=torch.float32)
        z_views = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
        pos_mask = torch.tensor([[True, False, True], [False, True, False]], device=device, dtype=torch.bool)
        tau = 0.2
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        self.assertEqual(loss.device.type, "cpu")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_output_device_cuda(self):
        """Test if output tensor is on the same device as input (CUDA)."""
        device = torch.device("cuda")
        z_anchor = torch.randn(2, 4, device=device, dtype=torch.float32)
        z_views = torch.randn(2, 3, 4, device=device, dtype=torch.float32)
        pos_mask = torch.tensor([[True, False, True], [False, True, False]], device=device, dtype=torch.bool)
        tau = 0.2
        
        loss = info_nce_multi(z_anchor, z_views, pos_mask, tau)
        self.assertEqual(loss.device.type, "cuda")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)