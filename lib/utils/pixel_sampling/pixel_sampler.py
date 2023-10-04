import numpy as np

class PixelSampler:
    @staticmethod
    def sample_patches(num_patch, patch_size, patch_ratio, H, W, msk_sample=None):
        assert(patch_size % 2 == 0)
        true_patch_size = patch_size
        patch_size = int(patch_size * patch_ratio)
        half_true_patch_size = true_patch_size // 2
        half_patch_size = patch_size // 2
        
        if msk_sample is not None and msk_sample.sum() > 0:
            num_fg_patch = num_patch
            non_zero = msk_sample.nonzero()
            permutation = np.random.permutation(msk_sample.sum())[:num_fg_patch].astype(np.int32)
            X_, Y_ = non_zero[1][permutation], non_zero[0][permutation]
            X_ = np.clip(X_, half_patch_size, W-half_patch_size)
            Y_ = np.clip(Y_, half_patch_size, H-half_patch_size)
        else:
            num_fg_patch = 0
            
        num_patch = num_patch - num_fg_patch
        X = np.random.randint(low=half_patch_size, high=W-half_patch_size, size=num_patch)
        Y = np.random.randint(low=half_patch_size, high=H-half_patch_size, size=num_patch)
        if num_fg_patch > 0:
            X = np.concatenate([X, X_]).astype(np.int32)
            Y = np.concatenate([Y, Y_]).astype(np.int32)
        grid = np.meshgrid((np.arange(true_patch_size)-half_true_patch_size)*patch_ratio, (np.arange(true_patch_size)-half_true_patch_size)*patch_ratio)
        return np.concatenate([grid[0].reshape(-1) + x for x in X]), np.concatenate([grid[1].reshape(-1) + y for y in Y])