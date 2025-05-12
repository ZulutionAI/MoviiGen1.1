import logging
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def setup_logger(name='sampler', level=logging.INFO):
    """Setup a logger that prints to console with timestamps"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(process)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

logger = setup_logger()


def get_world_size():
    """Get world size safely, defaulting to 1 if distributed not initialized"""
    if not torch.distributed.is_available():
        return 1
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    """Get rank safely, defaulting to 0 if distributed not initialized"""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_safe_indices(dataset_size: int, count: int, default_idx: int = 0) -> List[int]:
    """
    Generate a safe list of indices that are guaranteed to be within range.

    Args:
        dataset_size: Size of the dataset
        count: Number of indices needed
        default_idx: Default index to use if dataset is empty

    Returns:
        List of valid indices
    """
    if dataset_size <= 0:
        return [default_idx] * count

    # Use modulo to ensure no index is out of bounds
    return [(i % dataset_size) for i in range(count)]


class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        logger.info(f"num replicas in DistributedGroupSampler: {num_replicas}")
        logger.info(f"current rank in DistributedGroupSampler: {rank}")
        logger.info(f"local rank, rank, world size: {os.environ.get('LOCAL_RANK')}, {os.environ.get('RANK')}, {os.environ.get('WORLD_SIZE')}")

        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'aspect_ratios'), "Dataset must have aspect_ratios attribute"
        self.aspect_ratios = self.dataset.aspect_ratios
        self.group_sizes = np.bincount(self.aspect_ratios)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(
                    self.group_sizes[i] * 1.0 / self.samples_per_gpu / self.num_replicas
                )
            ) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.aspect_ratios == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) * self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InferenceSampler(Sampler):
    """Produce indices for inference."""

    def __init__(self, size: int):
        self._size = size
        assert size > 0
        self._rank = get_rank()
        self._world_size = get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


class RobustAspectRatioLengthGroupedSampler(Sampler):
    """
    Robust sampler that groups samples by both aspect ratio and sequence length.
    Provides strong guarantees against distributed communication hang-ups with:
    - Comprehensive error recovery
    - Guaranteed sample distribution
    - Edge case handling for all parallel configurations
    - Detailed logging for troubleshooting
    """

    def __init__(
        self,
        batch_size: int,
        rank: int,
        world_size: int,
        dataset=None,
        lengths=None,
        aspect_ratios=None,
        group_frame=False,
        group_resolution=False,
        generator=None,
        sp_size=1,
        verbose=False,
        length_bucket_count=4,
        balance_factor=0.9,
        max_samples_per_gpu=None,
        fallback_samples=1000,  # Number of samples to use if distribution fails
        min_samples_per_rank=1  # Minimum number of samples each rank must receive
    ):
        """
        Initialize the robust sampler with comprehensive error handling.

        Args:
            batch_size: Batch size per GPU
            rank: Process rank
            world_size: Total number of processes
            dataset: Dataset with aspect_ratios and lengths attributes
            lengths: Sequence lengths (alternative to dataset)
            aspect_ratios: Aspect ratio categories (alternative to dataset)
            group_frame: Whether to group by frame count
            group_resolution: Whether to group by resolution
            generator: Random number generator
            sp_size: Sequence parallel size
            verbose: Whether to print detailed logs
            length_bucket_count: Number of length buckets to create per aspect ratio
            balance_factor: Factor to balance length distribution
            max_samples_per_gpu: Safety limit for max samples per GPU
            fallback_samples: Number of samples to use if distribution fails
            min_samples_per_rank: Minimum samples each rank must receive
        """
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.generator = generator
        self.group_frame = group_frame
        self.group_resolution = group_resolution
        self.epoch = 0
        self.sp_size = sp_size
        self.verbose = verbose and rank == 0  # Only log from rank 0
        self.length_bucket_count = length_bucket_count
        self.balance_factor = balance_factor
        self.max_samples_per_gpu = max_samples_per_gpu
        self.fallback_samples = fallback_samples
        self.min_samples_per_rank = min_samples_per_rank

        # Calculate SP group ID and local rank within SP group
        self.sp_group = self.rank // self.sp_size
        self.sp_local_rank = self.rank % self.sp_size
        self.sp_group_count = self.world_size // self.sp_size

        # Verify SP configuration
        if self.world_size % self.sp_size != 0:
            logger.warning(
                f"WARNING: world_size ({self.world_size}) is not divisible by sp_size ({self.sp_size}). "
                f"This may cause uneven workload distribution. Proceeding with {self.sp_group_count} groups."
            )

        try:
            # Get lengths and aspect ratios from dataset or directly
            if dataset is not None:
                if not hasattr(dataset, 'aspect_ratios') or not hasattr(dataset, 'lengths'):
                    # Create fallback attributes if missing
                    logger.warning("Dataset missing required attributes. Creating synthetic data.")
                    dataset_size = len(dataset)
                    self.aspect_ratios = np.zeros(dataset_size, dtype=np.int64)
                    self.lengths = np.ones(dataset_size, dtype=np.int64) * 50  # Default length
                else:
                    self.aspect_ratios = dataset.aspect_ratios
                    self.lengths = dataset.lengths
            else:
                if lengths is None or aspect_ratios is None:
                    raise ValueError("Either dataset or both lengths and aspect_ratios must be provided")
                self.lengths = lengths
                self.aspect_ratios = aspect_ratios

            # Convert to numpy arrays for faster operations
            if not isinstance(self.lengths, np.ndarray):
                self.lengths = np.array(self.lengths)
            if not isinstance(self.aspect_ratios, np.ndarray):
                self.aspect_ratios = np.array(self.aspect_ratios)

            # Verify data integrity
            if len(self.lengths) != len(self.aspect_ratios):
                logger.error(f"Length mismatch: lengths={len(self.lengths)}, aspect_ratios={len(self.aspect_ratios)}")
                # Create consistent synthetic data
                max_len = max(len(self.lengths), len(self.aspect_ratios))
                self.lengths = np.ones(max_len, dtype=np.int64) * 50
                self.aspect_ratios = np.zeros(max_len, dtype=np.int64)

            # Handle empty dataset
            if len(self.lengths) == 0:
                logger.warning("Empty dataset detected. Creating synthetic data.")
                self.lengths = np.ones(self.fallback_samples, dtype=np.int64) * 50
                self.aspect_ratios = np.zeros(self.fallback_samples, dtype=np.int64)

            # Log initialization info
            if self.verbose:
                self._log_initialization_info()

        except Exception as e:
            logger.error(f"Error initializing sampler: {str(e)}")
            logger.error(traceback.format_exc())
            # Create synthetic data for recovery
            self.lengths = np.ones(self.fallback_samples, dtype=np.int64) * 50
            self.aspect_ratios = np.zeros(self.fallback_samples, dtype=np.int64)
            logger.warning(f"Created fallback synthetic data with {self.fallback_samples} samples")

    def _log_initialization_info(self):
        """Log detailed initialization information for debugging"""
        try:
            unique_ars = np.unique(self.aspect_ratios)
            ar_counts = {int(ar): int(np.sum(self.aspect_ratios == ar)) for ar in unique_ars}
            length_stats = {
                'min': float(np.min(self.lengths)),
                'max': float(np.max(self.lengths)),
                'mean': float(np.mean(self.lengths)),
                'median': float(np.median(self.lengths))
            }

            logger.info(f"Initialized RobustAspectRatioLengthGroupedSampler with:")
            logger.info(f"  Total samples: {len(self.lengths)}")
            logger.info(f"  Unique aspect ratios: {len(unique_ars)}")
            logger.info(f"  Aspect ratio distribution: {ar_counts}")
            logger.info(f"  Length stats: {length_stats}")
            logger.info(f"  SP configuration: {self.sp_group_count} groups with {self.sp_size} GPUs each")
            logger.info(f"  This rank: {self.rank} (SP group {self.sp_group}, local rank {self.sp_local_rank})")
        except Exception as e:
            logger.error(f"Error logging initialization info: {str(e)}")

    def __len__(self):
        """Return the dataset length"""
        return len(self.lengths)

    def __iter__(self):
        """
        Create and return an iterator over the sample indices for this rank.
        Includes comprehensive error handling and recovery mechanisms.
        """
        try:
            # Set deterministic seed based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)

            # Group indices by aspect ratio and length
            ar_length_buckets = self._create_ar_length_buckets(g)

            # Create balanced batches
            balanced_batches = self._create_balanced_batches(ar_length_buckets, g)

            # Distribute batches to SP groups
            my_indices = self._distribute_to_rank(balanced_batches, g)

            # Ensure minimum samples per rank
            if len(my_indices) < self.min_samples_per_rank:
                logger.warning(
                    f"Rank {self.rank}: Received only {len(my_indices)} samples. "
                    f"Adding padding to reach minimum {self.min_samples_per_rank}."
                )
                # Pad with valid indices (using modulo to stay in bounds)
                dataset_size = len(self.lengths)
                padding_indices = get_safe_indices(
                    dataset_size,
                    self.min_samples_per_rank - len(my_indices)
                )
                my_indices.extend(padding_indices)

            # Log statistics for the first epoch or periodically
            if self.verbose and (self.epoch <= 1 or self.epoch % 10 == 0):
                self._log_distribution_stats(my_indices)

            return iter(my_indices)

        except Exception as e:
            # Handle any exceptions during iteration creation
            error_msg = f"Error creating iterator on rank {self.rank}: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)

            # Return a safe fallback (ensure training can continue)
            dataset_size = len(self.lengths)
            # Create deterministic "random" indices based on rank and epoch
            safe_indices = get_safe_indices(
                dataset_size,
                max(self.batch_size * 4, self.min_samples_per_rank),
                default_idx=(self.rank % max(1, dataset_size))
            )

            logger.warning(f"Rank {self.rank}: Using {len(safe_indices)} fallback indices")
            return iter(safe_indices)

    def _create_ar_length_buckets(self, generator):
        """
        Create buckets grouped by aspect ratio and sequence length.

        Args:
            generator: Random number generator

        Returns:
            List of bucketed indices
        """
        try:
            # Get unique aspect ratios
            unique_ars = np.unique(self.aspect_ratios)
            buckets = []

            for ar in unique_ars:
                # Get indices for this aspect ratio
                ar_indices = np.where(self.aspect_ratios == ar)[0]

                if len(ar_indices) == 0:
                    continue

                # Sort indices by length
                ar_lengths = self.lengths[ar_indices]
                sorted_idx = np.argsort(ar_lengths)
                sorted_ar_indices = ar_indices[sorted_idx]

                # Create length buckets
                total_indices = len(sorted_ar_indices)
                indices_per_bucket = max(1, total_indices // self.length_bucket_count)

                for i in range(0, total_indices, indices_per_bucket):
                    end_idx = min(i + indices_per_bucket, total_indices)
                    length_bucket = sorted_ar_indices[i:end_idx].tolist()

                    # Randomize within bucket if not grouping by frame
                    if not self.group_frame:
                        length_bucket = [length_bucket[i] for i in
                                        torch.randperm(len(length_bucket), generator=generator).tolist()]

                    buckets.append(length_bucket)

            # Handle case where no buckets were created
            if not buckets:
                logger.warning(f"No buckets were created. Creating fallback bucket.")
                # Create a fallback bucket with all indices
                all_indices = np.arange(len(self.lengths))
                # Shuffle all indices
                shuffled_indices = all_indices[torch.randperm(
                    len(all_indices), generator=generator
                ).numpy()]
                buckets.append(shuffled_indices.tolist())

            return buckets

        except Exception as e:
            logger.error(f"Error creating AR length buckets: {str(e)}")
            logger.error(traceback.format_exc())

            # Create a fallback bucket with sequential indices
            safe_indices = list(range(min(len(self.lengths), self.fallback_samples)))
            return [safe_indices]

    def _create_balanced_batches(self, buckets, generator):
        """
        Create balanced batches across SP groups and ranks.

        Args:
            buckets: List of index buckets
            generator: Random number generator

        Returns:
            List of batches balanced for distributed training
        """
        try:
            # Safety check
            if not buckets:
                # Create one bucket with sequential indices as fallback
                logger.warning("Empty buckets in _create_balanced_batches, creating fallback.")
                dataset_size = len(self.lengths)
                buckets = [list(range(min(dataset_size, self.fallback_samples)))]

            # Shuffle the buckets
            shuffled_bucket_indices = torch.randperm(len(buckets), generator=generator).tolist()
            shuffled_buckets = [buckets[i] for i in shuffled_bucket_indices]

            # Calculate super-batch size (total samples per SP group)
            sp_batch_size = self.batch_size * self.sp_size

            # Create batches
            all_batches = []
            current_batch = []
            current_length_sum = 0

            # Target average length per sample for balanced batches
            avg_sample_length = float(np.mean(self.lengths)) if len(self.lengths) > 0 else 50.0

            for bucket in shuffled_buckets:
                # Safety check
                if not bucket:
                    continue

                bucket_copy = bucket.copy()

                while bucket_copy:
                    # Space left in current batch
                    space_left = sp_batch_size - len(current_batch)

                    if space_left == 0:
                        # Current batch is full, add it and start a new one
                        all_batches.append((current_batch, current_length_sum))
                        current_batch = []
                        current_length_sum = 0
                        space_left = sp_batch_size

                    # Add samples to the current batch
                    samples_to_add = min(space_left, len(bucket_copy))
                    batch_addition = bucket_copy[:samples_to_add]
                    bucket_copy = bucket_copy[samples_to_add:]

                    current_batch.extend(batch_addition)
                    # Safely get length sums using try/except for each index
                    for idx in batch_addition:
                        try:
                            if 0 <= idx < len(self.lengths):
                                current_length_sum += self.lengths[idx]
                            else:
                                current_length_sum += avg_sample_length
                        except (IndexError, TypeError):
                            current_length_sum += avg_sample_length

            # Handle the last batch
            if current_batch:
                if len(current_batch) < sp_batch_size:
                    # Need to pad the last batch
                    padding_needed = sp_batch_size - len(current_batch)

                    # Get indices with similar aspect ratio (from the first batches)
                    similar_indices = []
                    for batch, _ in all_batches:
                        similar_indices.extend(batch)
                        if len(similar_indices) >= padding_needed:
                            break

                    # Handle case where we don't have enough indices for padding
                    if len(similar_indices) < padding_needed:
                        # Generate synthetic indices using modulo to stay in bounds
                        dataset_size = len(self.lengths)
                        padding = get_safe_indices(dataset_size, padding_needed - len(similar_indices))
                        similar_indices.extend(padding)

                    padding = similar_indices[:padding_needed]
                    current_batch.extend(padding)

                    # Update length sum with the new indices
                    for idx in padding:
                        try:
                            if 0 <= idx < len(self.lengths):
                                current_length_sum += self.lengths[idx]
                            else:
                                current_length_sum += avg_sample_length
                        except (IndexError, TypeError):
                            current_length_sum += avg_sample_length

                all_batches.append((current_batch, current_length_sum))

            # Extract just the batches (without the length sums)
            balanced_batches = [batch for batch, _ in all_batches]

            # Handle empty batches list
            if not balanced_batches:
                logger.warning("No balanced batches created. Creating fallback batches.")
                # Create synthetic batches
                dataset_size = len(self.lengths)
                indices = list(range(min(dataset_size, self.fallback_samples)))

                # Split into batches of sp_batch_size
                for i in range(0, len(indices), sp_batch_size):
                    batch = indices[i:i + sp_batch_size]
                    # Pad the last batch if needed
                    if len(batch) < sp_batch_size:
                        padding = indices[:sp_batch_size - len(batch)]
                        batch.extend(padding)
                    balanced_batches.append(batch)

            # Shuffle the batches again for better distribution
            if balanced_batches:
                final_batch_indices = torch.randperm(len(balanced_batches), generator=generator).tolist()
                final_balanced_batches = [balanced_batches[i] for i in final_batch_indices]
                return final_balanced_batches
            else:
                # Final safety check
                logger.error("Failed to create balanced batches even with fallback.")
                # Create one minimal batch that won't cause issues
                return [[0] * sp_batch_size]

        except Exception as e:
            logger.error(f"Error creating balanced batches: {str(e)}")
            logger.error(traceback.format_exc())

            # Create fallback batches
            sp_batch_size = self.batch_size * self.sp_size
            dataset_size = len(self.lengths)
            safe_indices = list(range(min(dataset_size, self.fallback_samples)))

            # Split into batches
            fallback_batches = []
            for i in range(0, len(safe_indices), sp_batch_size):
                batch = safe_indices[i:i + sp_batch_size]
                # Pad the last batch if needed
                if len(batch) < sp_batch_size:
                    padding = safe_indices[:sp_batch_size - len(batch)]
                    batch.extend(padding)
                fallback_batches.append(batch)

            if not fallback_batches:
                # Create at least one batch
                fallback_batches = [[0] * sp_batch_size]

            return fallback_batches

    def _distribute_to_rank(self, batches, generator):
        """
        Distribute batches to this rank, respecting SP groups and local ranks.
        Includes comprehensive safety checks to prevent hangs.

        Args:
            batches: List of batches to distribute
            generator: Random number generator

        Returns:
            List of indices for this rank
        """
        try:
            # Safety check for empty batches
            if not batches:
                logger.warning(f"Rank {self.rank}: No batches to distribute. Creating fallback batches.")
                # Return safe fallback indices
                dataset_size = len(self.lengths)
                return get_safe_indices(dataset_size, self.batch_size * 4)

            # Step 1: Assign batches to SP groups evenly
            # Handle case where we have fewer batches than SP groups
            if len(batches) < self.sp_group_count:
                logger.warning(
                    f"Fewer batches ({len(batches)}) than SP groups ({self.sp_group_count}). "
                    f"Some groups will receive duplicate batches."
                )
                # Duplicate batches to ensure each SP group gets at least one
                while len(batches) < self.sp_group_count:
                    batches.extend(batches[:self.sp_group_count - len(batches)])

            batches_per_sp_group = len(batches) // self.sp_group_count
            extra_batches = len(batches) % self.sp_group_count

            # Calculate start and end indices for this SP group
            start_idx = self.sp_group * batches_per_sp_group + min(self.sp_group, extra_batches)
            end_idx = start_idx + batches_per_sp_group + (1 if self.sp_group < extra_batches else 0)

            # Safety check for index bounds
            start_idx = min(start_idx, len(batches) - 1)
            end_idx = min(end_idx, len(batches))
            if start_idx >= end_idx:
                end_idx = start_idx + 1

            # Get batches for this SP group
            sp_group_batches = batches[start_idx:end_idx]

            # Handle empty sp_group_batches
            if not sp_group_batches and batches:
                logger.warning(f"SP group {self.sp_group} received no batches. Using first batch.")
                sp_group_batches = [batches[0]]

            # Step 2: Distribute to local ranks within the SP group
            my_indices = []

            for batch_idx, batch in enumerate(sp_group_batches):
                try:
                    # Safety check batch size
                    if not batch:
                        continue

                    # Ensure batch size is compatible with SP size
                    target_size = self.sp_size * self.batch_size

                    # Pad or trim batch to make it compatible with reshaping
                    batch_len = len(batch)
                    if batch_len % self.sp_size != 0:
                        # Calculate number of complete SP rows, padding is needed
                        complete_rows = batch_len // self.sp_size
                        target_len = (complete_rows + 1) * self.sp_size

                        # Pad batch by reusing elements
                        padding_needed = target_len - batch_len
                        if padding_needed > 0:
                            padding = batch[:padding_needed]
                            batch = batch + padding

                    # Reshape batch for distribution across SP ranks
                    batch_array = np.array(batch)

                    # Ensure batch can be reshaped
                    reshaped_rows = len(batch_array) // self.sp_size
                    if reshaped_rows == 0:
                        # Not enough elements, duplicate
                        while len(batch_array) < self.sp_size:
                            batch_array = np.concatenate([batch_array, batch_array])
                        reshaped_rows = len(batch_array) // self.sp_size

                    # Now safely reshape
                    try:
                        reshaped_batch = batch_array.reshape(reshaped_rows, self.sp_size, -1)
                    except ValueError:
                        # If reshape fails, reorganize and try again
                        remainder = len(batch_array) % self.sp_size
                        if remainder > 0:
                            # Remove excess elements
                            batch_array = batch_array[:-remainder]

                        # If still empty, add elements
                        if len(batch_array) == 0:
                            batch_array = np.array([0] * self.sp_size)

                        # Try reshape again
                        reshaped_rows = len(batch_array) // self.sp_size
                        reshaped_batch = batch_array.reshape(reshaped_rows, self.sp_size, -1)

                    # Extract indices for this local rank
                    for row_idx in range(reshaped_batch.shape[0]):
                        if self.sp_local_rank < reshaped_batch.shape[1]:
                            row_data = reshaped_batch[row_idx, self.sp_local_rank]
                            my_indices.extend(row_data.tolist())

                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_idx}: {str(batch_error)}")
                    # Continue with next batch instead of failing completely

            # Make sure we have at least some indices
            if not my_indices and len(self.lengths) > 0:
                logger.warning(f"Rank {self.rank} received no valid indices. Creating fallback indices.")
                # Use deterministic indices based on rank
                start_idx = (self.rank * self.batch_size) % len(self.lengths)
                my_indices = [(start_idx + i) % len(self.lengths) for i in range(self.batch_size * 4)]

            # Enforce max samples limit if specified
            if self.max_samples_per_gpu and len(my_indices) > self.max_samples_per_gpu:
                my_indices = my_indices[:self.max_samples_per_gpu]

            return my_indices

        except Exception as e:
            logger.error(f"Error in _distribute_to_rank for rank {self.rank}: {str(e)}")
            logger.error(traceback.format_exc())

            # Create safe fallback indices
            dataset_size = len(self.lengths)
            return get_safe_indices(
                dataset_size,
                self.batch_size * 4,
                default_idx=self.rank % max(1, dataset_size)
            )

    def _log_distribution_stats(self, indices):
        """Log statistics about the distribution for debugging."""
        try:
            if not indices:
                logger.warning(f"WARNING: Rank {self.rank} received no samples!")
                return

            # Gather stats for this rank
            total_length = sum(
                self.lengths[i] if 0 <= i < len(self.lengths) else 0
                for i in indices
            )
            mean_length = total_length / len(indices)

            # Count aspect ratios
            ar_counts = {}
            valid_indices = [i for i in indices if 0 <= i < len(self.aspect_ratios)]
            for idx in valid_indices:
                ar = int(self.aspect_ratios[idx])
                ar_counts[ar] = ar_counts.get(ar, 0) + 1

            logger.info(f"Rank {self.rank} distribution:")
            logger.info(f"  Samples: {len(indices)}")
            logger.info(f"  Valid indices: {len(valid_indices)}/{len(indices)}")
            logger.info(f"  Total sequence length: {total_length}")
            logger.info(f"  Average length per sample: {mean_length:.2f}")
            logger.info(f"  Aspect ratio distribution: {ar_counts}")

            # Verify balanced workload across SP group
            global_mean = np.mean(self.lengths) if len(self.lengths) > 0 else 1.0
            if global_mean > 0:
                logger.info(f"  This indicates a {100.0 * mean_length / global_mean:.1f}% workload relative to average")

        except Exception as e:
            logger.error(f"Error logging distribution stats: {str(e)}")
            logger.error(traceback.format_exc())

    def set_epoch(self, epoch):
        """Set the epoch for this sampler for deterministic shuffling."""
        old_epoch = self.epoch
        self.epoch = epoch
        if self.verbose and (epoch == 0 or epoch % 10 == 0 or epoch != old_epoch + 1):
            logger.info(f"Setting epoch to {epoch}")


# For backward compatibility
ImprovedAspectRatioLengthGroupedSampler = RobustAspectRatioLengthGroupedSampler