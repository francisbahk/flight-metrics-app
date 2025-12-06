import numpy as np
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
from Logistic import LinearLogisticModel 
from GP import GaussianProcessUtilityModel

@dataclass
class ComparisonResult:
    """Store result of a pairwise comparison"""
    option_a: int
    option_b: int
    winner: int  # Should be either option_a or option_b
    features_a: np.ndarray
    features_b: np.ndarray


from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

@dataclass
class BatchComparisonResult:
    """Store result of a batch comparison where one option beats all others"""
    batch_indices: List[int]
    winner_idx: int
    features_batch: np.ndarray  # Features for all options in batch
    feature_deltas: np.ndarray  # Winner features - loser features for each comparison


class BatchDuelingBanditOptimizer:
    """
    Dueling bandit optimizer for batch comparisons where one winner is selected from each batch.
    Uses EUBO (Expected Utility of Best Option) or simpler acquisition functions.
    """
    
    def __init__(
        self,
        all_options: List[int],
        features: np.ndarray,
        batch_size: int = 4,
        acquisition: str = "eubo",
        C: float = 1.0,
        n_samples_eubo: int = 100,
        exploration_weight: float = 2.0,
        min_appearances_for_model: int = 5,
        model_type: str = "gp",
        random_seed: Optional[int] = None
    ):
        """
        Args:
            all_options: List of all option indices
            features: Feature matrix (n_options x n_features)
            batch_size: Number of options to show per batch
            acquisition: Acquisition function type ("eubo", "ucb", "thompson", "random")
            C: Regularization parameter for logistic regression
            n_samples_eubo: Number of samples for EUBO computation
            exploration_weight: Weight for exploration in UCB
            min_appearances_for_model: Minimum data points before using model
            random_seed: Random seed for reproducibility
        """
        self.all_options = all_options
        self.features = features
        self.batch_size = batch_size
        self.acquisition = acquisition.lower()
        self.C = C
        self.n_samples_eubo = n_samples_eubo
        self.exploration_weight = exploration_weight
        self.min_appearances_for_model = min_appearances_for_model
        self.model_type = model_type
        # Set random seeds
        if random_seed is not None:
            np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize tracking
        self.shown_options: Set[int] = set()
        self.unshown_options: List[int] = list(all_options)
        self.comparison_history: List[BatchComparisonResult] = []
        
        # Initialize model
        #self.model = LinearLogisticModel(C=self.C, rng=self.rng)
        if self.model_type== 'logistic':
            self.model = LinearLogisticModel(C=self.C)
        else:
            self.model = GaussianProcessUtilityModel()
        
        # Track utilities and batch history
        self.utility_history: List[np.ndarray] = []
        self.batch_history: List[List[int]] = []
        
        # Track appearance and win counts
        self.appearance_counts = np.zeros(len(all_options))
        self.win_counts = np.zeros(len(all_options))
        
        # Cache for model predictions
        self._cached_utilities = None
        self._cached_uncertainties = None
    
    def select_next_batch(self) -> List[int]:
        """
        Select next batch using specified acquisition function.
        """
        # If not enough data, use exploration
        if len(self.comparison_history) < self.min_appearances_for_model:
            return self.select_exploration_batch()
        
        # Use specified acquisition function
        if self.acquisition == "eubo":
            return self.select_batch_eubo()
        elif self.acquisition == "ucb":
            return self.select_batch_ucb()
        elif self.acquisition == "thompson":
            return self.select_batch_thompson()
        elif self.acquisition == "random":
            return self.select_random_batch()
        else:  # random or fallback
            return self.select_exploration_batch()
    
    def select_exploration_batch(self) -> List[int]:
        """
        Select batch for pure exploration, prioritizing unshown options.
        """
        batch = []
        
        # First, include unshown options
        if self.unshown_options:
            n_unshown = min(self.batch_size, len(self.unshown_options))
            selected_unshown = self.rng.choice(
                self.unshown_options,
                size=n_unshown,
                replace=False
            ).tolist()
            batch.extend(selected_unshown)
            
            # Mark as shown
            for idx in selected_unshown:
                self.unshown_options.remove(idx)
                self.shown_options.add(idx)
        
        # Fill remaining slots with least-shown options
        if len(batch) < self.batch_size:
            remaining_needed = self.batch_size - len(batch)
            available = [idx for idx in self.all_options if idx not in batch]
            
            if available:
                # Sort by appearance count (ascending) and select least shown
                appearance_scores = [(idx, self.appearance_counts[idx]) for idx in available]
                appearance_scores.sort(key=lambda x: x[1])
                
                for idx, _ in appearance_scores[:remaining_needed]:
                    batch.append(idx)
        
        # Shuffle for random presentation order
        self.rng.shuffle(batch)
        self.batch_history.append(batch)
        
        return batch
    def select_random_batch(self) -> List[int]:
        """
        Select a random batch but ensure the winner of the previous iteration is included.
        """
        print('USING RANDOM')
        batch = []
        
        # Include the winner of the previous iteration if available
        if self.comparison_history:
            last_winner = self.comparison_history[-1].winner_idx
            batch.append(last_winner)
        
        # Fill the rest of the batch with random options
        remaining_needed = self.batch_size - len(batch)
        available = [idx for idx in self.all_options if idx not in batch]
        
        if available and remaining_needed > 0:
            selected_random = self.rng.choice(
                available,
                size=remaining_needed,
                replace=False
            ).tolist()
            batch.extend(selected_random)
        
        # Shuffle for random presentation order
        self.rng.shuffle(batch)
        self.batch_history.append(batch)
        
        return batch
    
    def select_batch_ucb(self) -> List[int]:
        """
        Select batch using Upper Confidence Bound (UCB) acquisition.
        """
        if not self.model.ready():
            return self.select_exploration_batch()
        
        # Compute UCB scores for all options
        means = self.model.posterior_mean_util(self.features)
        stds = self.model.posterior_std_util(self.features)
        ucb_scores = means + self.exploration_weight * stds
        
        # Add bonus for unshown options
        for idx in range(len(self.all_options)):
            if idx not in self.shown_options:
                ucb_scores[idx] += 1.0  # Exploration bonus
        
        # Select top-k by UCB score
        top_indices = np.argsort(ucb_scores)[::-1][:self.batch_size]
        batch = top_indices.tolist()
        
        # Mark new options as shown
        for idx in batch:
            if idx not in self.shown_options:
                self.shown_options.add(idx)
                if idx in self.unshown_options:
                    self.unshown_options.remove(idx)
        
        # Shuffle for random presentation
        self.rng.shuffle(batch)
        self.batch_history.append(batch)
        
        return batch
    
    def select_batch_thompson(self) -> List[int]:
        """
        Select batch using Thompson Sampling.
        """
        if not self.model.ready():
            return self.select_exploration_batch()
        
        # Sample utilities from posterior
        sampled_utils = self.model.sample_utilities(self.features, n_samples=1).flatten()
        
        # Add bonus for unshown options
        for idx in range(len(self.all_options)):
            if idx not in self.shown_options:
                sampled_utils[idx] += self.rng.normal(0.5, 0.2)  # Exploration bonus
        
        # Select top-k by sampled utility
        top_indices = np.argsort(sampled_utils)[::-1][:self.batch_size]
        batch = top_indices.tolist()
        
        # Mark new options as shown
        for idx in batch:
            if idx not in self.shown_options:
                self.shown_options.add(idx)
                if idx in self.unshown_options:
                    self.unshown_options.remove(idx)
        
        # Shuffle for random presentation
        self.rng.shuffle(batch)
        self.batch_history.append(batch)
        
        return batch
    
    def select_batch_eubo(self) -> List[int]:
        """
        Select batch using simplified EUBO (greedy approximation).
        Since true EUBO is computationally expensive, we use a greedy approach
        that sequentially selects options that maximize expected information gain.
        """
        print('USING EUBO')
        if not self.model.ready():
            return self.select_exploration_batch()
        
        batch = []
        available = list(range(len(self.all_options)))
        
        # Greedy selection: add options one by one
        for _ in range(self.batch_size):
            best_score = -np.inf
            best_idx = None
            
            # Evaluate each candidate
            for candidate_idx in available:
                if candidate_idx not in batch:
                    # Score this candidate given current batch
                    test_batch = batch + [candidate_idx]
                    score = self.compute_batch_information_gain(test_batch)
                    
                    # Add exploration bonus for unshown options
                    if candidate_idx not in self.shown_options:
                        score += 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_idx = candidate_idx
            
            if best_idx is not None:
                batch.append(best_idx)
                available.remove(best_idx)
        
        # Mark new options as shown
        for idx in batch:
            if idx not in self.shown_options:
                self.shown_options.add(idx)
                if idx in self.unshown_options:
                    self.unshown_options.remove(idx)
        
        # Shuffle for random presentation
        self.rng.shuffle(batch)
        self.batch_history.append(batch)
        
        return batch
    
    def compute_batch_information_gain(self, batch_indices: List[int]) -> float:
        """
        Compute approximate information gain from observing this batch.
        Uses a simplified version that measures reduction in uncertainty.
        """
        if not self.model.ready() or not batch_indices:
            return 0.0
        
        # Get current uncertainties
        current_stds = self.model.posterior_std_util(self.features)
        current_uncertainty = np.sum(current_stds)
        
        # Estimate reduction in uncertainty from this batch
        # Higher variance options provide more information
        batch_stds = current_stds[batch_indices]
        batch_uncertainty = np.sum(batch_stds)
        
        # Also consider diversity within batch (more diverse = more informative)
        if len(batch_indices) > 1:
            batch_features = self.features[batch_indices]
            # Compute pairwise distances
            diversity = 0.0
            for i in range(len(batch_indices)):
                for j in range(i + 1, len(batch_indices)):
                    diversity += np.linalg.norm(
                        batch_features[i] - batch_features[j]
                    )
            diversity /= (len(batch_indices) * (len(batch_indices) - 1) / 2)
        else:
            diversity = 0.0
        
        # Information gain combines uncertainty reduction and diversity
        info_gain = batch_uncertainty / current_uncertainty + 0.1 * diversity
        
        return info_gain
    
    def update_with_winner(self, batch_indices: List[int], winner_idx: int):
        """
        Update model with batch comparison result where winner beats all others.
        
        Args:
            batch_indices: Indices of all options in the batch
            winner_idx: Index of the winning option
        """
        if winner_idx not in batch_indices:
            raise ValueError(f"Winner {winner_idx} not in batch {batch_indices}")
        
        # Update counts
        for idx in batch_indices:
            self.appearance_counts[idx] += 1
        self.win_counts[winner_idx] += 1
        
        # Create comparison result
        batch_features = self.features[batch_indices]
        winner_features = self.features[winner_idx]
        
        # Compute feature deltas (winner - each loser)
        feature_deltas = []
        for idx in batch_indices:
            if idx != winner_idx:
                delta = winner_features - self.features[idx]
                feature_deltas.append(delta)
        
        if feature_deltas:
            feature_deltas = np.vstack(feature_deltas)
        else:
            feature_deltas = np.empty((0, winner_features.shape[0]))
        
        # Store comparison
        result = BatchComparisonResult(
            batch_indices=batch_indices,
            winner_idx=winner_idx,
            features_batch=batch_features,
            feature_deltas=feature_deltas
        )
        self.comparison_history.append(result)
        
        # Retrain model on all data unless acquisition is pure random
        if self.acquisition != "random":
            self._train_model_on_all_data()
            # Clear cache
            self._cached_utilities = None
            self._cached_uncertainties = None
    
    def _train_model_on_all_data(self):
        """Train model on all historical comparison data."""
        if not self.comparison_history:
            return
        
        # Prepare training data from all batch comparisons
        X_deltas = []
        y_outcomes = []
        
        for comp in self.comparison_history:
            # For each comparison, winner beat all others
            for delta in comp.feature_deltas:
                if delta.size > 0:  # Only add if delta exists
                    X_deltas.append(delta)
                    y_outcomes.append(1)  # Winner always wins
        
        if X_deltas:
            # Stack into arrays
            X_delta = np.vstack(X_deltas)
            y = np.array(y_outcomes)
            
            # Train new model from scratch on all data
            self.model = self._create_new_model()
            self.model.fit_on_duels(X_delta, y)
            
            # Store utility estimates
            if self.model.ready():
                utilities = self.model.posterior_mean_util(self.features)
                self.utility_history.append(utilities)
    
    def _create_new_model(self):
        """Create a new model instance respecting the configured model_type."""
        if self.model_type == 'logistic':
            from Logistic import LinearLogisticModel
            return LinearLogisticModel(C=self.C, rng=self.rng)
        else:
            from GP import GaussianProcessUtilityModel
            return GaussianProcessUtilityModel(rng=self.rng)
    
    def get_current_ranking(self, top_k: int = 10) -> Dict[str, Any]:
        """
        Get current ranking of options based on model utilities or win rates.
        """
        n_comparisons = len(self.comparison_history)
        
        if self.model.ready() and n_comparisons >= self.min_appearances_for_model:
            # Use model utilities
            utilities = self.model.posterior_mean_util(self.features)
            uncertainties = self.model.posterior_std_util(self.features)
            ranking = np.argsort(utilities)[::-1]
        else:
            # Use win rates
            win_rates = np.divide(
                self.win_counts,
                np.maximum(self.appearance_counts, 1)
            )
            ranking = np.argsort(win_rates)[::-1]
            utilities = win_rates
            uncertainties = np.zeros_like(win_rates)
        
        # Ensure top_k doesn't exceed number of options
        top_k = min(top_k, len(self.all_options))
        top_indices = ranking[:top_k]
        
        return {
            'ranking': top_indices.tolist(),
            'utilities': utilities[top_indices].tolist(),
            'uncertainties': uncertainties[top_indices].tolist(),
            'win_rates': (self.win_counts[top_indices] / 
                         np.maximum(self.appearance_counts[top_indices], 1)).tolist(),
            'appearances': self.appearance_counts[top_indices].tolist(),
            'wins': self.win_counts[top_indices].tolist(),
            'model_ready': self.model.ready()
        }
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """
        Get metrics to assess convergence of the optimization.
        """
        if len(self.utility_history) < 2:
            return {
                'converged': False,
                'utility_change': np.inf,
                'ranking_stability': 0.0
            }
        
        # Compare last two utility estimates
        util_change = np.mean(np.abs(
            self.utility_history[-1] - self.utility_history[-2]
        ))
        
        # Check ranking stability (Kendall's tau)
        ranking_current = np.argsort(self.utility_history[-1])[::-1][:10]
        ranking_previous = np.argsort(self.utility_history[-2])[::-1][:10]
        
        # Simple overlap metric
        overlap = len(set(ranking_current) & set(ranking_previous)) / 10.0
        
        return {
            'converged': util_change < 0.01 and overlap > 0.8,
            'utility_change': float(util_change),
            'ranking_stability': float(overlap),
            'n_comparisons': len(self.comparison_history),
            'n_shown': len(self.shown_options),
            'n_total': len(self.all_options)
        }