# Curriculum Learning & CumAcc Sample Difficulty Scoring

## Paper Summary

**Title:** "Does the Definition of Difficulty Matter? Scoring Functions and their Role for Curriculum Learning"  
**Authors:** Rampp et al. (TUM, Imperial College)  
**Source:** arXiv:2411.00973v1, November 2024

### Core Concept

Curriculum learning (CL) trains models by presenting samples in order of difficulty—easy samples first, then progressively harder ones. The key challenge is defining and measuring "difficulty."

### Key Findings

1. **Scoring functions are sensitive to training settings** - Model architecture, hyperparameters, and even random seeds significantly affect difficulty orderings

2. **Ensemble scoring improves robustness** - Averaging difficulty scores across multiple random seeds produces more stable orderings

3. **Easy-to-hard ordering outperforms hard-to-easy** - Especially with slowly saturating pacing functions

4. **CL doesn't always beat standard training** - No universal advantage, but models trained with different orderings learn complementary concepts (useful for late fusion)

5. **Faster pacing functions perform better** - Quickly incorporating harder samples and saturating on full dataset yields best results

### Scoring Functions Compared

| Method | Description | Compute Cost | Granularity |
|--------|-------------|--------------|-------------|
| C-score | Consistency across training subsets (held-out) | Very High | Fine |
| CVLoss | Average loss in k-fold cross-validation | High | Fine |
| **CumAcc** | Sum of correct classifications over epochs | **Low** | Coarse |
| FIT | First epoch correctly classified (and stays correct) | Low | Coarse |
| CELoss | Final cross-entropy loss per sample | Low | Fine |
| Transfer Teacher | SVM margins on pre-trained features | Medium | Fine |
| Prediction Depth | Layer at which KNN probe matches prediction | Medium | Coarse |

### Datasets Tested

- **CIFAR-10** (Computer Vision): 50k training images, 10 classes
- **DCASE2020** (Computer Audition): 14k audio samples, acoustic scene classification

---

## CumAcc (Cumulative Accuracy)

### Definition

```
CumAcc(sample_i) = (Number of epochs sample_i was correctly classified) / (Total epochs)
```

- **Higher score** → Easier sample (learned early, stays correct)
- **Lower score** → Harder sample (frequently misclassified)

### Why Choose CumAcc

1. **Computationally cheap** - Single training run required
2. **Works well for audio** - Validated on DCASE2020 (acoustic scene classification)
3. **High correlation with other methods** - 0.85+ correlation with CVLoss and FIT
4. **Benefits from ensembling** - Robustness increases significantly with ensemble size

### Limitations

- **Coarse granularity** - Only 32-48 unique values (single seed), 151+ with ensembles
- **Ties handled by dataset order** - Samples with same score sorted by original index

---

## Implementation Guide

### Step 1: Dataset Wrapper (Must Return Indices)

```python
import torch
from torch.utils.data import Dataset, DataLoader

class IndexedDataset(Dataset):
    """Wraps any dataset to also return sample indices."""
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __getitem__(self, idx):
        data, target = self.base_dataset[idx]
        return idx, data, target
    
    def __len__(self):
        return len(self.base_dataset)
```

### Step 2: CumAcc Tracker

```python
import numpy as np

class CumAccTracker:
    """Tracks per-sample correctness across training epochs."""
    
    def __init__(self, num_samples: int, num_epochs: int):
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.correct_matrix = np.zeros((num_samples, num_epochs), dtype=np.int8)
        self.current_epoch = 0
    
    def update_batch(self, indices: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor):
        """Update tracking after each batch. Call during training."""
        predictions = logits.argmax(dim=1)
        correct = (predictions == targets).cpu().numpy().astype(np.int8)
        batch_indices = indices.cpu().numpy()
        self.correct_matrix[batch_indices, self.current_epoch] = correct
    
    def end_epoch(self):
        """Call at the end of each training epoch."""
        self.current_epoch += 1
    
    def get_cumacc_scores(self) -> np.ndarray:
        """Compute CumAcc score for each sample."""
        return self.correct_matrix.sum(axis=1) / self.num_epochs
    
    def get_difficulty_ranking(self, easy_to_hard: bool = True) -> np.ndarray:
        """
        Get sample indices sorted by difficulty.
        
        Args:
            easy_to_hard: If True, easiest samples first (for curriculum learning)
                         If False, hardest samples first (for anti-curriculum)
        
        Returns:
            Array of sample indices in difficulty order
        """
        scores = self.get_cumacc_scores()
        if easy_to_hard:
            return np.argsort(-scores)  # Descending: high score (easy) first
        else:
            return np.argsort(scores)   # Ascending: low score (hard) first
    
    def save_scores(self, filepath: str):
        """Save scores to file for later use."""
        scores = self.get_cumacc_scores()
        np.save(filepath, scores)
    
    def get_difficulty_distribution(self) -> dict:
        """Get statistics about difficulty distribution."""
        scores = self.get_cumacc_scores()
        return {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'num_unique': len(np.unique(scores)),
            'easiest_samples': np.where(scores == scores.max())[0].tolist()[:10],
            'hardest_samples': np.where(scores == scores.min())[0].tolist()[:10],
        }
```

### Step 3: Training Loop for Score Collection

```python
import torch
import torch.nn as nn
from tqdm import tqdm

def collect_cumacc_scores(
    model: nn.Module,
    train_dataset: Dataset,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> CumAccTracker:
    """
    Train model and collect CumAcc scores.
    
    Args:
        model: PyTorch model
        train_dataset: Training dataset (will be wrapped with IndexedDataset)
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        CumAccTracker with computed scores
    """
    # Wrap dataset to return indices
    indexed_dataset = IndexedDataset(train_dataset)
    train_loader = DataLoader(
        indexed_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    tracker = CumAccTracker(len(train_dataset), num_epochs)
    
    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Collecting CumAcc scores"):
        model.train()
        
        for indices, inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Track correctness (no grad needed)
            with torch.no_grad():
                tracker.update_batch(indices, outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tracker.end_epoch()
    
    return tracker
```

### Step 4: Ensemble Scoring (Recommended)

```python
def collect_ensemble_cumacc_scores(
    model_class,
    model_kwargs: dict,
    train_dataset: Dataset,
    num_seeds: int = 5,
    num_epochs: int = 50,
    **train_kwargs
) -> np.ndarray:
    """
    Collect CumAcc scores across multiple random seeds and average.
    
    This improves robustness significantly (see paper Figure 1).
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model constructor
        train_dataset: Training dataset
        num_seeds: Number of random seeds to use
        num_epochs: Training epochs per seed
        **train_kwargs: Additional arguments for collect_cumacc_scores
    
    Returns:
        Ensemble-averaged CumAcc scores
    """
    all_scores = []
    
    for seed in range(num_seeds):
        print(f"\n=== Seed {seed + 1}/{num_seeds} ===")
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Create fresh model
        model = model_class(**model_kwargs)
        
        # Collect scores
        tracker = collect_cumacc_scores(
            model=model,
            train_dataset=train_dataset,
            num_epochs=num_epochs,
            **train_kwargs
        )
        
        all_scores.append(tracker.get_cumacc_scores())
    
    # Average across seeds
    ensemble_scores = np.mean(all_scores, axis=0)
    return ensemble_scores
```

---

## Curriculum Learning Training

### Pacing Functions

The paper tests four pacing functions that control how quickly new samples are introduced:

```python
import numpy as np

def get_pacing_function(name: str):
    """
    Get pacing function by name.
    
    Pacing functions determine dataset size at each training step.
    Input: progress in [0, 1] (0 = start, 1 = saturation point)
    Output: fraction of dataset to use in [b, 1] where b = initial fraction
    
    Faster saturation (log, root) generally performs better.
    """
    functions = {
        'logarithmic': lambda t, b: b + (1 - b) * np.log(1 + t * (np.e - 1)),
        'root': lambda t, b: b + (1 - b) * np.sqrt(t),
        'linear': lambda t, b: b + (1 - b) * t,
        'exponential': lambda t, b: b + (1 - b) * (np.exp(t) - 1) / (np.e - 1),
    }
    return functions[name]
```

### Curriculum DataLoader

```python
class CurriculumSampler(torch.utils.data.Sampler):
    """
    Sampler that introduces samples according to difficulty ordering and pacing.
    
    Args:
        difficulty_ranking: Sample indices sorted by difficulty (easy first)
        num_samples: Total number of samples
        initial_fraction: Starting fraction of dataset (b in paper)
        saturation_fraction: Training progress at which full dataset is used (a in paper)
        pacing_function: Name of pacing function ('logarithmic', 'root', 'linear', 'exponential')
        current_progress: Current training progress in [0, 1]
    """
    
    def __init__(
        self,
        difficulty_ranking: np.ndarray,
        num_samples: int,
        initial_fraction: float = 0.2,
        saturation_fraction: float = 0.5,
        pacing_function: str = 'logarithmic',
        current_progress: float = 0.0
    ):
        self.difficulty_ranking = difficulty_ranking
        self.num_samples = num_samples
        self.initial_fraction = initial_fraction
        self.saturation_fraction = saturation_fraction
        self.pacing_fn = get_pacing_function(pacing_function)
        self.current_progress = current_progress
    
    def set_progress(self, progress: float):
        """Update training progress (call each epoch)."""
        self.current_progress = min(progress, 1.0)
    
    def _get_current_size(self) -> int:
        """Calculate how many samples to use at current progress."""
        if self.current_progress >= self.saturation_fraction:
            return self.num_samples
        
        # Normalize progress to [0, 1] within the saturation window
        normalized = self.current_progress / self.saturation_fraction
        fraction = self.pacing_fn(normalized, self.initial_fraction)
        return int(fraction * self.num_samples)
    
    def __iter__(self):
        current_size = self._get_current_size()
        # Use only the easiest `current_size` samples
        active_indices = self.difficulty_ranking[:current_size].copy()
        np.random.shuffle(active_indices)
        return iter(active_indices.tolist())
    
    def __len__(self):
        return self._get_current_size()
```

### Full Curriculum Training Loop

```python
def train_with_curriculum(
    model: nn.Module,
    train_dataset: Dataset,
    difficulty_ranking: np.ndarray,
    num_epochs: int = 50,
    batch_size: int = 32,
    initial_fraction: float = 0.2,
    saturation_fraction: float = 0.5,
    pacing_function: str = 'logarithmic',
    learning_rate: float = 0.001,
    device: str = 'cuda'
) -> nn.Module:
    """
    Train model using curriculum learning.
    
    Args:
        model: PyTorch model
        train_dataset: Training dataset
        difficulty_ranking: Sample indices sorted easy-to-hard
        num_epochs: Total training epochs
        batch_size: Batch size
        initial_fraction: Starting fraction of dataset (paper uses 0.2)
        saturation_fraction: Progress at which full dataset is used (paper uses 0.5 or 0.8)
        pacing_function: 'logarithmic' (best), 'root', 'linear', or 'exponential'
        learning_rate: Learning rate
        device: Training device
    
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create curriculum sampler
    sampler = CurriculumSampler(
        difficulty_ranking=difficulty_ranking,
        num_samples=len(train_dataset),
        initial_fraction=initial_fraction,
        saturation_fraction=saturation_fraction,
        pacing_function=pacing_function
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    for epoch in range(num_epochs):
        # Update curriculum progress
        progress = epoch / num_epochs
        sampler.set_progress(progress)
        current_size = len(sampler)
        
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Samples: {current_size}/{len(train_dataset)} | "
              f"Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model
```

---

## Application to Audio Datasets

### SpeechCommands

```python
import torchaudio

# Load dataset
train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root='./data',
    subset='training',
    download=True
)

# Typical settings
NUM_EPOCHS = 50
BATCH_SIZE = 64
NUM_CLASSES = 35  # or 12 for subset
```

### AudioSet

```python
# AudioSet is large - consider:
# 1. Using fewer epochs (20-30)
# 2. Subset for initial experiments
# 3. Pre-computed spectrograms for speed

# Recommended ensemble size: 3-5 seeds
# Memory for tracking: ~100MB+ for full AudioSet
```

### Audio-Specific Considerations

1. **Feature extraction** - Use log-Mel spectrograms (64 bins typical)
2. **Model architecture** - CNN10/CNN14 (PANNs) or EfficientNet work well
3. **Data augmentation** - SpecAugment, time shifting, noise injection
4. **Class imbalance** - AudioSet is heavily imbalanced; consider balanced sampling within curriculum

---

## Expected Results

From the paper (DCASE2020 acoustic scene classification):

| Method | Accuracy |
|--------|----------|
| Baseline (no CL) | 0.583 |
| Best CL (TT) | 0.577 |
| Best Late Fusion (CL+ACL+RCL+B1) | 0.636 |

**Key insight:** CL alone may not beat baseline, but combining models trained with different orderings via late fusion yields significant improvements.

---

## Quick Start Checklist

1. [ ] Wrap dataset with `IndexedDataset`
2. [ ] Train model with `CumAccTracker` for 50 epochs
3. [ ] (Recommended) Repeat with 3-5 different seeds for ensemble
4. [ ] Average scores and compute difficulty ranking
5. [ ] Train final model with `CurriculumSampler`
6. [ ] Use logarithmic pacing, b=0.2, a=0.5 as starting point
7. [ ] (Optional) Train with different orderings and late-fuse for best results

---

## References

- Paper: https://arxiv.org/abs/2411.00973
- Code: https://github.com/autrainer/aucurriculum
- autrainer: https://github.com/autrainer/autrainer
