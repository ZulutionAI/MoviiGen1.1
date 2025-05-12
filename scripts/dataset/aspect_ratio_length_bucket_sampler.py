"""
aspect_ratio_length_bucket_sampler.py

This module implements a specialized sampler that groups samples by aspect ratio
and frame length for efficient training of video or image models in distributed
environments.
"""

import math
from collections import Counter, defaultdict
from typing import List, Optional, Tuple, Dict

import torch
from torch.utils.data import Sampler


def get_world_size() -> int:
    """Get the number of processes in the distributed training."""
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


class AspectRatioLengthBucketSampler(Sampler):
    """
    Sampler that groups data by both aspect ratio and frame length buckets to ensure:

    1. All samples in a global batch have the same aspect ratio and similar frame lengths
    2. All samples are included in each epoch (no samples are skipped)
    3. Load is balanced across all GPUs

    This sampler is ideal for video or image datasets where maintaining consistent
    shapes within batches improves training efficiency without sacrificing data diversity.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        lengths: Optional[List[int]] = None,
        aspect_ratios: Optional[List[int]] = None,
        num_length_bins: int = 8,
        drop_last: bool = False,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the AspectRatioLengthBucketSampler.

        Args:
            batch_size: Number of samples per batch per GPU
            dataset_size: Total size of the dataset
            rank: Rank of current process (defaults to get_rank())
            world_size: Number of processes participating (defaults to get_world_size())
            lengths: List of frame lengths for each sample in the dataset
            aspect_ratios: List of aspect ratio bins for each sample in the dataset
            num_length_bins: Number of bins to use for grouping frame lengths
            drop_last: Whether to drop samples that don't fill a complete global batch
            seed: Random seed for reproducibility
            verbose: Whether to print detailed statistics during initialization and iteration
        """
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.rank = get_rank() if rank is None else rank
        self.world_size = get_world_size() if world_size is None else world_size
        self.lengths = lengths
        self.aspect_ratios = aspect_ratios
        self.num_length_bins = num_length_bins
        self.drop_last = drop_last
        self.seed = seed
        self.verbose = verbose
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.world_size

        # Validate inputs
        if self.lengths is None or self.aspect_ratios is None:
            raise ValueError("Both 'lengths' and 'aspect_ratios' must be provided")

        if len(self.lengths) != self.dataset_size or len(self.aspect_ratios) != self.dataset_size:
            raise ValueError("Length of 'lengths' and 'aspect_ratios' must match dataset_size")

        # Generate frame length bin boundaries
        self.length_bins = self._create_length_bins()

        if self.verbose and self.rank == 0:
            self._log_initialization_info()

    def _create_length_bins(self) -> List[int]:
        """
        Create bin boundaries for frame lengths to ensure balanced distribution.
        Uses quantiles to create bins with roughly equal number of samples.

        Returns:
            List of bin boundary values
        """
        min_len, max_len = min(self.lengths), max(self.lengths)

        if self.num_length_bins <= 1:
            return [min_len, max_len + 1]  # Only one bin

        # Use quantiles for more balanced distribution of samples across bins
        sorted_lengths = sorted(self.lengths)
        bins = [min_len]

        for i in range(1, self.num_length_bins):
            idx = i * len(sorted_lengths) // self.num_length_bins
            bins.append(sorted_lengths[idx])

        bins.append(max_len + 1)  # Ensure max length is included
        return bins

    def _get_length_bin(self, length: int) -> int:
        """
        Determine which length bin a sample belongs to.

        Args:
            length: Frame length of the sample

        Returns:
            Bin index (0 to num_length_bins-1)
        """
        for i in range(len(self.length_bins) - 1):
            if self.length_bins[i] <= length < self.length_bins[i + 1]:
                return i
        return 0  # Default to first bin if not found

    def _log_initialization_info(self) -> None:
        """Log information about the sampler initialization."""
        print("Initializing AspectRatioLengthBucketSampler:")
        print(f"  Rank: {self.rank}, World Size: {self.world_size}")
        print(f"  Batch Size: {self.batch_size}, Global Batch Size: {self.global_batch_size}")
        print(f"  Dataset Size: {self.dataset_size}")
        print(f"  Number of Length Bins: {self.num_length_bins}")
        print(f"  Frame Length Bin Boundaries: {self.length_bins}")

        # Log aspect ratio distribution
        ar_counts = Counter(self.aspect_ratios)
        print(f"  Aspect Ratio Distribution: {sorted(ar_counts.items())}")

        # Log length bin distribution
        length_bins = [self._get_length_bin(length) for length in self.lengths]
        length_bin_counts = Counter(length_bins)
        print(f"  Length Bin Distribution: {sorted(length_bin_counts.items())}")

        # Log combined bucket distribution
        bucket_counts = Counter([(self.aspect_ratios[i], self._get_length_bin(self.lengths[i]))
                                 for i in range(self.dataset_size)])
        print(f"  Number of (AR, Length) Buckets: {len(bucket_counts)}")

        # Calculate expected batches
        total_complete_batches = sum([count // self.global_batch_size * self.global_batch_size
                                      for count in bucket_counts.values()])
        print(f"  Expected samples per epoch: {total_complete_batches} " +
              f"({total_complete_batches/self.dataset_size:.2%} of dataset)")

    def __len__(self) -> int:
        """Return the expected number of samples for this rank."""
        return self.dataset_size // self.world_size

    def __iter__(self):
        """
        Create an iterator that yields indices for the current rank.
        The indices are grouped by aspect ratio and frame length to ensure
        consistent shapes within each batch, while maintaining full dataset coverage.
        """
        # Set random seed for this epoch to ensure different shuffling each epoch
        # but consistency across ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Step 1: Group indices by aspect ratio and length bin
        bucket_indices: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx in range(self.dataset_size):
            ar = self.aspect_ratios[idx]
            length_bin = self._get_length_bin(self.lengths[idx])
            bucket_key = (ar, length_bin)
            bucket_indices[bucket_key].append(idx)

        # Step 2: Shuffle indices within each bucket
        for bucket_key in bucket_indices:
            indices = bucket_indices[bucket_key]
            bucket_indices[bucket_key] = [
                indices[i] for i in torch.randperm(len(indices), generator=g).tolist()
            ]

        # Step 3: Create global batches from each bucket
        global_batches = []
        bucket_usage_stats = {}

        for bucket_key, indices in bucket_indices.items():
            if self.drop_last:
                # Only keep samples that form complete global batches
                num_complete_batches = len(indices) // self.global_batch_size
                num_to_keep = num_complete_batches * self.global_batch_size
                useful_indices = indices[:num_to_keep]
            else:
                useful_indices = indices.copy()

                # If not dropping incomplete batches, pad to full global batch
                remainder = len(useful_indices) % self.global_batch_size
                if remainder > 0 and len(useful_indices) > 0:
                    padding_needed = self.global_batch_size - remainder
                    # Use samples from the same bucket for padding to maintain shape consistency
                    padding = [
                        indices[i % len(indices)]
                        for i in range(padding_needed)
                    ]
                    useful_indices.extend(padding)

            # Track bucket usage statistics
            bucket_usage_stats[bucket_key] = {
                "total": len(indices),
                "used": len(useful_indices),
                "batches": len(useful_indices) // self.global_batch_size
            }

            # Create global batches from this bucket
            for i in range(0, len(useful_indices), self.global_batch_size):
                batch = useful_indices[i:i + self.global_batch_size]
                if len(batch) == self.global_batch_size:  # Only add complete batches
                    global_batches.append(batch)

        # Step 4: Shuffle the order of global batches
        batch_indices = torch.randperm(len(global_batches), generator=g).tolist()
        shuffled_global_batches = [global_batches[i] for i in batch_indices]

        # Step 5: Extract samples for the current rank
        rank_indices = []
        for global_batch in shuffled_global_batches:
            # Each rank gets a slice of the global batch
            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size
            rank_batch = global_batch[start_idx:end_idx]
            rank_indices.extend(rank_batch)

        # Log statistics if verbose
        if self.verbose and self.rank == 0:
            self._log_epoch_stats(bucket_usage_stats, len(global_batches), len(rank_indices))

        return iter(rank_indices)

    def _log_epoch_stats(self, bucket_stats, num_global_batches, num_rank_samples):
        """Log statistics about the current epoch."""
        total_samples = self.dataset_size
        used_samples = num_global_batches * self.global_batch_size
        # unique_samples = len(set(sample for batch in bucket_stats.values() for sample in range(batch["total"])))

        print(f"\nEpoch {self.epoch} Statistics:")
        print(f"  Total dataset samples: {total_samples}")
        print(f"  Samples used in batches: {used_samples} ({used_samples/total_samples:.2%} of dataset)")
        print(f"  Global batches created: {num_global_batches}")
        print(f"  Samples per rank: {num_rank_samples}")

        # Print bucket statistics summary
        ar_stats = defaultdict(int)
        length_bin_stats = defaultdict(int)

        for (ar, length_bin), stats in bucket_stats.items():
            ar_stats[ar] += stats["used"]
            length_bin_stats[length_bin] += stats["used"]

        print(f"  Number of buckets: {len(bucket_stats)}")
        print(f"  Aspect ratio distribution in batches: {sorted(ar_stats.items())}")
        print(f"  Length bin distribution in batches: {sorted(length_bin_stats.items())}")

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler to ensure different shuffling order for each epoch.

        Args:
            epoch: Epoch number
        """
        self.epoch = epoch


class AspectRatioLengthBucketDistributedSampler(AspectRatioLengthBucketSampler):
    """
    Extended version of AspectRatioLengthBucketSampler that's specifically designed
    for distributed training scenarios.

    This version handles edge cases in distributed environments and provides
    additional capabilities needed for multi-node training.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        lengths: Optional[List[int]] = None,
        aspect_ratios: Optional[List[int]] = None,
        num_length_bins: int = 8,
        drop_last: bool = False,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the distributed sampler with options for multi-node training.

        Args:
            dataset_size: Total size of the dataset
            batch_size: Number of samples per batch per GPU
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process within num_replicas
            lengths: List of frame lengths for each sample
            aspect_ratios: List of aspect ratio bins for each sample
            num_length_bins: Number of bins to use for grouping frame lengths
            drop_last: Whether to drop samples that don't fill a complete global batch
            seed: Random seed for reproducibility
            verbose: Whether to print detailed statistics
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        super().__init__(
            batch_size=batch_size,
            dataset_size=dataset_size,
            rank=rank,
            world_size=num_replicas,
            lengths=lengths,
            aspect_ratios=aspect_ratios,
            num_length_bins=num_length_bins,
            drop_last=drop_last,
            seed=seed,
            verbose=verbose
        )

        # Make sure all processes have the same number of batches
        # This is critical for distributed training with gradient accumulation
        self.num_samples_per_replica = math.ceil(self.dataset_size / self.world_size)

        if verbose and rank == 0:
            print("Distributed sampler initialized with:")
            print(f"  Num replicas: {num_replicas}")
            print(f"  Samples per replica: {self.num_samples_per_replica}")

    def __len__(self) -> int:
        """Return the number of samples for this replica."""
        return self.num_samples_per_replica


class SPAwareAspectRatioLengthBucketDistributedSampler(AspectRatioLengthBucketDistributedSampler):
    """
    A Sequence Parallel aware distributed sampler that extends AspectRatioLengthBucketDistributedSampler.

    This sampler ensures:
    1. Each sequence parallel group receives unique data samples (different SP groups process different data)
    2. All GPUs within the same sequence parallel group receive identical data indices
       (same SP group processes different parts of the same data)
    3. Maintains aspect ratio and length bucketing for training efficiency

    This sampler coordinates with sequence parallel data transformers to ensure no data redundancy
    while supporting efficient sequence parallel training.

    The key concept is to treat each sequence parallel group as a single unit for data sampling purposes,
    and then let the sequence parallel mechanism handle splitting the sequences across GPUs within the group.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        sp_size: int,  # sequence parallel group size
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        lengths: Optional[List[int]] = None,
        aspect_ratios: Optional[List[int]] = None,
        num_length_bins: int = 8,
        drop_last: bool = False,
        seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize the Sequence Parallel aware distributed sampler.

        Args:
            dataset_size: Total size of the dataset
            batch_size: Batch size per GPU
            sp_size: Size of each sequence parallel group (number of GPUs that process one sequence)
            num_replicas: Total number of processes participating in distributed training
            rank: Global rank of the current process
            lengths: List of frame lengths for each sample
            aspect_ratios: List of aspect ratios for each sample
            num_length_bins: Number of bins for grouping frame lengths
            drop_last: Whether to drop samples that don't fill a complete batch
            seed: Random seed for reproducibility
            verbose: Whether to print detailed statistics
        """
        self.sp_size = sp_size

        # Get world size and rank if not provided
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        # Calculate data parallel group size and rank
        # Total world size = data parallel size * sequence parallel size
        if num_replicas % sp_size != 0:
            raise ValueError(f"Sequence parallel size ({sp_size}) must evenly divide world size ({num_replicas})")

        # Compute the data parallel and sequence parallel dimensions
        self.data_parallel_size = num_replicas // sp_size
        self.data_parallel_rank = rank // sp_size
        self.sp_rank = rank % sp_size

        # Store original values for reference
        self.global_world_size = num_replicas
        self.global_rank = rank

        # Log initial setup information
        if verbose and self.sp_rank == 0:
            print(f"Initializing Sequence Parallel Aware Sampler:")
            print(f"  Total world size: {num_replicas}")
            print(f"  Sequence parallel size: {sp_size}")
            print(f"  Data parallel size: {self.data_parallel_size}")

        # Initialize parent class using data parallel size/rank instead of global size/rank
        # This ensures each SP group gets unique data
        super().__init__(
            dataset_size=dataset_size,
            batch_size=batch_size,
            num_replicas=self.data_parallel_size,  # Use data parallel size
            rank=self.data_parallel_rank,          # Use data parallel rank
            lengths=lengths,
            aspect_ratios=aspect_ratios,
            num_length_bins=num_length_bins,
            drop_last=drop_last,
            seed=seed,
            # Only log on first GPU of each SP group to avoid duplicate messages
            verbose=verbose and self.sp_rank == 0
        )

        # After parent initialization, add clarification about batch sizes
        # This is the batch_size * data_parallel_size (unique samples)
        self.dp_global_batch_size = self.global_batch_size
        # This is the total across all GPUs including SP replicas
        self.total_global_batch_size = batch_size * num_replicas

        if verbose and self.sp_rank == 0:
            print(f"  Batch size per GPU: {batch_size}")
            print(f"  Data parallel global batch size: {self.dp_global_batch_size} (unique samples)")
            print(f"  Total global batch size: {self.total_global_batch_size} (including SP replicas)")

    def _log_sp_info(self):
        """Log sequence parallel information for debugging purposes."""
        world_size = get_world_size()
        rank = get_rank()

        print(f"Process {rank}/{world_size} Sequence Parallel Info:")
        print(f"  Global rank: {rank}")
        print(f"  SP size: {self.sp_size}")
        print(f"  SP rank: {self.sp_rank}")
        print(f"  DP size: {self.data_parallel_size}")
        print(f"  DP rank: {self.data_parallel_rank}")

    def __iter__(self):
        """
        Create an iterator that yields indices for the current data parallel rank.
        All GPUs within the same sequence parallel group get identical indices.

        This method inherits behavior from parent class but uses the data parallel rank
        instead of global rank to determine which samples to process. This ensures
        that each SP group processes unique data.

        Returns:
            Iterator yielding indices for this GPU to process
        """
        # Log SP info on first epoch if verbose is enabled
        if self.epoch == 0 and self.verbose and self.sp_rank == 0:
            self._log_sp_info()

        # Call parent class __iter__ which will use the configured data parallel rank
        return super().__iter__()
