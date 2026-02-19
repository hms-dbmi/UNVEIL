"""
Unified Demographic-informed Patch Filtering Agent.

This agent implements a multi-signal approach to patch filtering
with parametric filtering rate selection and ensemble strategies.

Strategy:
- Phase 1: Enhanced multi-signal feature engineering
- Phase 2: Parametric filtering rate selection (no validation data required)
- Phase 3: Ensemble strategy generation with equal-weight averaging

"""

import torch
import numpy as np
import json
import h5py
from pathlib import Path
from datetime import datetime
from demographic_agent import DemographicPatchAgent


class UnifiedDemographicAgent(DemographicPatchAgent):
    """
    Unified agent with enhanced feature engineering, parametric filtering,
    and ensemble strategies.
    """
    
    def __init__(self, *args, use_v6_routing=True, **kwargs):
        """
        Initialize unified agent with multi-factor routing.
        Inherits all base functionality from DemographicPatchAgent.
        
        Args:
            use_v6_routing: If True, use multi-factor routing (default: True, recommended)
        """
        # Store routing strategy parameter before parent init
        self.use_v6_routing = use_v6_routing
        
        # Initialize parent class
        super().__init__(*args, **kwargs)
        
        # Add unified-specific components if strategy is 'unified'
        if self.strategy == 'unified':
            self._init_unified_components()
    
    def _init_unified_components(self):
        """Initialize unified agent specific components."""
        # Ensemble logging
        self.ensemble_log = {}
        self.features_log = {}
        
        # Weighting parameters for composite scoring
        self.weight_attention = 0.5
        self.weight_confidence = 0.2
        self.weight_uncertainty = 0.1
        self.weight_heterogeneity = 0.1
        self.weight_spatial = 0.1
        
        # Parametric filtering rate parameters
        self.base_rate = 0.15  # Base 15% filtering rate
        self.min_rate = 0.05   # Minimum 5%
        self.max_rate = 0.40   # Maximum 40%
        
        # Multi-factor routing
        self.use_v6_routing = getattr(self, 'use_v6_routing', True)
        
        if self.use_v6_routing:
            # Uses dynamic routing based on multiple factors
            # Factors computed WITHOUT data leakage:
            # 1. Training progress (total_slides_processed)
            # 2. Group imbalance ratio (from demographic attention data)
            # 3. Demographic model accuracy (from prediction data, excluding current patient)
            
            self.v6_threshold = 200  # Primary threshold for training progress
            
            # Compute group imbalance from demographic attention data
            # This is computed once at init, reflects overall dataset balance
            self.group_imbalance_ratio = self._compute_group_imbalance_ratio()
            
            # Compute demographic model accuracy from training data only
            self.demo_model_accuracy = self._compute_demo_accuracy()
            
            print(f"[DemographicAgent] Multi-Factor Routing enabled")
            print(f"[DemographicAgent] Dataset: {self.cancer.upper()}")
            print(f"[DemographicAgent] Group imbalance ratio: {self.group_imbalance_ratio:.2f}")
            print(f"[DemographicAgent] Demographic model accuracy: {self.demo_model_accuracy:.3f}")
            print(f"[DemographicAgent] Dynamic routing based on:")
            print(f"[DemographicAgent]   - Training progress (threshold: {self.v6_threshold} slides)")
            print(f"[DemographicAgent]   - Group imbalance (threshold: 5.0)")
            print(f"[DemographicAgent]   - Demo accuracy (threshold: 0.60)")
        
        else:
            print(f"[DemographicAgent] Initialized without multi-factor routing")
    
    def get_patch_filter_mask(self, slide_id, num_patches, mutation_coords=None, slide_type='FS'):
        """
        Override parent method to use unified approach when strategy='unified'.
        
        Args:
            slide_id: Slide identifier
            num_patches: Number of patches in mutation prediction data
            mutation_coords: Optional coordinates for coordinate-based matching
            slide_type: 'FS' or 'PM'
            
        Returns:
            Boolean mask (True = keep patch)
        """
        if self.strategy != 'unified':
            # Use parent implementation for other strategies
            return super().get_patch_filter_mask(slide_id, num_patches, mutation_coords, slide_type)
        
        # Unified approach implementation
        self.stats['total_slides_processed'] += 1
        self.stats['total_patches_original'] += num_patches
        
        # If no demographic model available, don't filter
        if not self.demographic_model_available:
            keep_mask = np.ones(num_patches, dtype=bool)
            self.stats['total_patches_kept'] += num_patches
            return keep_mask
        
        # Get demographic data
        patient_barcode = self._normalize_slide_id(slide_id, is_demographic=False)
        demographic_slide_keys = self._normalize_slide_id(patient_barcode, is_demographic=True)
        
        slide_key = None
        for key in demographic_slide_keys:
            if key in self.attention_data:
                slide_key = key
                break
        
        if slide_key is None:
            # No demographic data - don't filter
            self.stats['slides_with_no_demographic_data'] += 1
            keep_mask = np.ones(num_patches, dtype=bool)
            self.stats['total_patches_kept'] += num_patches
            return keep_mask
        
        attention_list = self.attention_data[slide_key]
        pred_info = self.prediction_data.get(slide_key, {})
        prediction = pred_info.get('prediction', [0.5])[0]
        
        # Handle length mismatch with coordinate matching
        if len(attention_list) != num_patches:
            attention_list = self._handle_length_mismatch(
                attention_list, num_patches, slide_key, mutation_coords, slide_type
            )
        
        # Phase 1: Compute enhanced features
        features = self._compute_enhanced_features(
            attention_list, prediction, mutation_coords, pred_info
        )
        
        # Determine filtering strategy based on prediction correctness and version
        is_correct = features.get('prediction_correct', None)
        
        # Multi-factor routing (highest precedence)
        if self.use_v6_routing and is_correct is False:
            # Prediction is WRONG - route based on multiple factors
            current_slides = self.stats['total_slides_processed']
            
            # Decision logic based on three factors:
            # 1. Group imbalance > 5.0 AND small dataset -> conservative random
            # 2. Demo accuracy < 0.60 -> always random (unreliable model)
            # 3. Otherwise: dynamic threshold based on training progress
            
            if self.demo_model_accuracy < 0.60:
                # Demographic model is unreliable - use random always
                final_mask = self._generate_random_filter_mask_adaptive(
                    num_patches, features, slide_id
                )
                routing_reason = 'low_demo_accuracy'
                
            elif self.group_imbalance_ratio > 5.0 and current_slides < 150:
                # Severe imbalance + small dataset - preserve minority data
                final_mask = self._generate_random_filter_mask_adaptive(
                    num_patches, features, slide_id
                )
                routing_reason = 'severe_imbalance_small'
                
            elif current_slides < self.v6_threshold:
                # Early training - use conservative random
                final_mask = self._generate_random_filter_mask_adaptive(
                    num_patches, features, slide_id
                )
                routing_reason = 'early_training'
                
            else:
                # Late training - use signal-leveraging
                scores = self._compute_composite_scores(features, num_patches)
                final_mask = self._generate_ensemble_masks(scores, features, num_patches, slide_id)
                routing_reason = 'late_training'
            
            # Log multi-factor routing decision
            if slide_id not in self.decisions_log:
                self.decisions_log[slide_id] = {}
            self.decisions_log[slide_id]['v6_mode'] = routing_reason
            self.decisions_log[slide_id]['v6_current_slides'] = int(current_slides)
            self.decisions_log[slide_id]['v6_imbalance'] = float(self.group_imbalance_ratio)
            self.decisions_log[slide_id]['v6_demo_acc'] = float(self.demo_model_accuracy)
        
        # Default: Attention-based filtering (when prediction is correct or V6 not enabled)
        else:
            # Prediction is CORRECT or advanced strategies not enabled
            scores = self._compute_composite_scores(features, num_patches)
            final_mask = self._generate_ensemble_masks(scores, features, num_patches, slide_id)
        
        self.stats['total_patches_kept'] += final_mask.sum()
        
        # Log decision
        self._log_unified_decision(slide_id, num_patches, final_mask, features)
        
        return final_mask
    
    def _handle_length_mismatch(self, attention_list, num_patches, slide_key, mutation_coords, slide_type):
        """
        Handle length mismatch using coordinate-based matching or resampling.
        This is inherited from parent but called here for clarity.
        """
        self.stats['slides_with_length_mismatch'] += 1
        
        # Try coordinate-based matching
        if mutation_coords is None:
            mutation_coords = self._load_coordinates(slide_key, slide_type)
        
        if mutation_coords is not None and len(mutation_coords) == num_patches:
            demo_coords = self._load_coordinates(slide_key, slide_type)
            
            if demo_coords is not None and len(demo_coords) == len(attention_list):
                # Perform coordinate-based matching
                attention_list = self._match_patches_by_coordinates(
                    attention_list, demo_coords, mutation_coords
                )
                self.stats['coordinate_matched_successfully'] += 1
                return attention_list
        
        # Fallback: position-based resampling
        self.stats['slides_with_coordinate_mismatch'] += 1
        if len(attention_list) > num_patches:
            indices = np.random.choice(len(attention_list), num_patches, replace=False)
            attention_list = [attention_list[i] for i in sorted(indices)]
        else:
            median_attention = np.median(attention_list)
            attention_list = list(attention_list) + [median_attention] * (num_patches - len(attention_list))
        
        return attention_list
    
    # ==========================================================================
    # MULTI-FACTOR ROUTING HELPER METHODS (NO DATA LEAKAGE)
    # ==========================================================================
    
    def _compute_group_imbalance_ratio(self):
        """
        Compute group imbalance ratio from demographic attention data.
        NO DATA LEAKAGE: Uses only demographic model's training distribution.
        
        Returns:
            Float: Ratio of majority to minority group size
        """
        if not hasattr(self, 'prediction_data') or len(self.prediction_data) == 0:
            return 1.0  # Default: balanced
        
        # Count predictions by group (class 0 vs class 1 from demographic model)
        group_counts = {0: 0, 1: 0}
        
        for slide_id, pred_info in self.prediction_data.items():
            prediction = pred_info.get('prediction', [0.5])[0]
            predicted_class = 1 if prediction > 0.5 else 0
            group_counts[predicted_class] += 1
        
        # Calculate imbalance ratio (majority / minority)
        minority_size = min(group_counts[0], group_counts[1])
        majority_size = max(group_counts[0], group_counts[1])
        
        if minority_size == 0:
            return 10.0  # Severe imbalance
        
        imbalance_ratio = majority_size / minority_size
        return imbalance_ratio
    
    def _compute_demo_accuracy(self):
        """
        Compute demographic model accuracy from prediction data.
        NO DATA LEAKAGE: Uses only slides with ground truth in demographic data.
        
        Returns:
            Float: Accuracy of demographic model (0-1)
        """
        if not hasattr(self, 'prediction_data') or len(self.prediction_data) == 0:
            return 0.5  # Default: random guessing
        
        correct = 0
        total = 0
        
        for slide_id, pred_info in self.prediction_data.items():
            if 'target' in pred_info and 'prediction' in pred_info:
                prediction = pred_info.get('prediction', [0.5])[0]
                target = pred_info.get('target', [0.5])[0]
                
                predicted_class = 1 if prediction > 0.5 else 0
                true_class = int(target)
                
                if predicted_class == true_class:
                    correct += 1
                total += 1
        
        if total == 0:
            return 0.5  # Default: random guessing
        
        accuracy = correct / total
        return accuracy
    
    # ==========================================================================
    # PHASE 1: ENHANCED FEATURE ENGINEERING
    # ==========================================================================
    
    def _compute_enhanced_features(self, attention, prediction, coords=None, pred_info=None):
        """
        Phase 1: Multi-signal feature engineering.
        NO DATA LEAKAGE - computed from slide-level data only.
        
        Args:
            attention: List or array of attention scores
            prediction: Slide-level prediction (probability)
            coords: Optional patch coordinates for spatial analysis
            pred_info: Optional prediction info dict with 'target' for correctness weighting
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Convert attention to numpy array
        attention_array = np.array(attention, dtype=float)
        
        # 1. Attention normalization (existing)
        max_att = np.max(attention_array)
        features['attention_norm'] = attention_array / (max_att + 1e-8)
        features['attention_mean'] = float(np.mean(attention_array))
        features['attention_max'] = float(max_att)
        
        # 2. Slide-level confidence with optional correctness weighting
        base_confidence = float(abs(prediction - 0.5) * 2)  # [0, 1]
        features['prediction'] = float(prediction)
        features['base_confidence'] = base_confidence
        
        # Apply correctness weighting if enabled
        if self.use_correctness_weighting and pred_info and 'target' in pred_info:
            target = pred_info.get('target', [0.5])[0]
            predicted_class = 1 if prediction > 0.5 else 0
            true_class = int(target)
            is_correct = (predicted_class == true_class)
            
            if is_correct:
                # Correct prediction - use full confidence
                effective_confidence = base_confidence
                features['prediction_correct'] = True
            else:
                # Wrong prediction - heavily reduce confidence (30% weight)
                effective_confidence = base_confidence * 0.3
                features['prediction_correct'] = False
            
            features['confidence'] = float(effective_confidence)
        else:
            # No correctness weighting
            features['confidence'] = base_confidence
            features['prediction_correct'] = None
        
        # 3. NEW: Prediction uncertainty (entropy-based)
        p = np.clip(prediction, 1e-8, 1 - 1e-8)
        entropy = -p * np.log2(p) - (1-p) * np.log2(1-p)
        features['uncertainty'] = float(entropy)  # [0, 1]
        
        # 4. NEW: Attention heterogeneity (coefficient of variation)
        attention_std = np.std(attention_array)
        features['heterogeneity'] = float(attention_std / (features['attention_mean'] + 1e-8))
        features['attention_std'] = float(attention_std)
        
        # 5. NEW: Spatial clustering (if coordinates available)
        if coords is not None and len(coords) == len(attention):
            features['spatial_clustering'] = self._compute_spatial_autocorrelation(
                attention_array, coords
            )
        else:
            features['spatial_clustering'] = 0.0
        
        # 6. NEW: Patch-level attention entropy
        features['patch_entropy'] = self._compute_patch_entropy(attention_array, k=10)
        
        return features
    
    def _compute_spatial_autocorrelation(self, attention, coords, k=5):
        """
        Moran's I statistic for spatial autocorrelation.
        Measures if high-attention patches cluster spatially.
        
        Args:
            attention: Array of attention scores
            coords: Array of (x, y) coordinates
            k: Number of nearest neighbors
            
        Returns:
            float: Moran's I statistic in [-1, 1]
        """
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("[UnifiedAgent] Warning: scipy not available, skipping spatial analysis")
            return 0.0
        
        n = len(attention)
        if n < k + 1:
            return 0.0
        
        try:
            # Build KD-tree for efficient nearest neighbor search
            tree = cKDTree(coords)
            
            # For each patch, find k nearest neighbors
            attention_mean = np.mean(attention)
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n):
                distances, indices = tree.query(coords[i], k=k+1)
                neighbors = indices[1:]  # Exclude self
                
                for j in neighbors:
                    weight = 1.0  # Simple binary weight
                    numerator += weight * (attention[i] - attention_mean) * (attention[j] - attention_mean)
                
                denominator += (attention[i] - attention_mean) ** 2
            
            # Moran's I
            if denominator > 1e-8:
                moran_i = (n / (n * k)) * (numerator / denominator)
                return float(np.clip(moran_i, -1, 1))  # Normalize to [-1, 1]
            else:
                return 0.0
        except Exception as e:
            print(f"[UnifiedAgent] Warning: Spatial autocorrelation failed: {e}")
            return 0.0
    
    def _compute_patch_entropy(self, attention, k=10):
        """
        Local entropy: measure attention diversity in local neighborhoods.
        
        Args:
            attention: Array of attention scores
            k: Window size for entropy computation
            
        Returns:
            float: Mean local entropy
        """
        n = len(attention)
        if n < k:
            return 0.0
        
        # Sort attention values
        sorted_att = np.sort(attention)
        
        # Compute local entropy in sliding windows
        entropies = []
        for i in range(n - k + 1):
            window = sorted_att[i:i+k]
            # Normalize to probability distribution
            window_sum = np.sum(window)
            if window_sum > 1e-8:
                window_norm = window / window_sum
                # Compute Shannon entropy
                entropy = -np.sum(window_norm * np.log2(window_norm + 1e-8))
                entropies.append(entropy)
        
        if entropies:
            return float(np.mean(entropies))
        else:
            return 0.0
    
    def _compute_composite_scores(self, features, num_patches):
        """
        Compute composite patch scores from multiple signals.
        
        Args:
            features: Dictionary of computed features
            num_patches: Number of patches
            
        Returns:
            Array of composite scores (higher = more demographic signal)
        """
        attention_norm = features['attention_norm']
        confidence = features['confidence']
        uncertainty = features['uncertainty']
        heterogeneity = np.clip(features['heterogeneity'], 0, 1)
        spatial = (features['spatial_clustering'] + 1) / 2  # Normalize to [0, 1]
        
        # Broadcast scalar values to match patch dimensions
        confidence_vec = np.full(num_patches, confidence)
        uncertainty_vec = np.full(num_patches, 1 - uncertainty)  # Invert: high uncertainty = low score
        heterogeneity_vec = np.full(num_patches, heterogeneity)
        spatial_vec = np.full(num_patches, spatial)
        
        # Weighted combination
        scores = (self.weight_attention * attention_norm +
                 self.weight_confidence * confidence_vec +
                 self.weight_uncertainty * uncertainty_vec +
                 self.weight_heterogeneity * heterogeneity_vec +
                 self.weight_spatial * spatial_vec)
        
        return scores
    
    # ==========================================================================
    # PHASE 2: PARAMETRIC FILTERING RATE SELECTION
    # ==========================================================================
    
    def _compute_parametric_filtering_rate(self, features):
        """
        Phase 2: Theoretically-grounded filtering rate.
        NO VALIDATION DATA REQUIRED - pure mathematical formula.
        
        Principles:
        1. High confidence → filter more (strong demographic signal)
        2. High uncertainty → filter less (unreliable demographic signal)
        3. High heterogeneity → filter more selectively (mixed patches)
        4. High spatial clustering → filter more (localized bias)
        
        Args:
            features: Dictionary of computed features
            
        Returns:
            float: Filtering rate in [0.05, 0.40]
        """
        confidence = features['confidence']
        uncertainty = features['uncertainty']
        heterogeneity = features['heterogeneity']
        spatial = features.get('spatial_clustering', 0.0)
        
        # Confidence factor: linear scaling [0.5, 1.0]
        # High confidence → factor approaches 1.0 (keep base rate or increase)
        confidence_factor = 0.5 + 0.5 * confidence
        
        # Uncertainty penalty: reduce filtering for uncertain predictions
        # High uncertainty → penalty approaches 0.5 (reduce filtering)
        uncertainty_penalty = 1.0 - 0.5 * uncertainty
        
        # Heterogeneity boost: increase filtering for heterogeneous slides
        # High variance suggests some patches are highly biased
        heterogeneity_boost = 1.0 + np.tanh(heterogeneity)  # [1.0, ~2.0]
        
        # Spatial boost: increase filtering if bias is spatially clustered
        # Positive spatial correlation → increase filtering
        spatial_boost = 1.0 + 0.5 * max(spatial, 0)  # [1.0, 1.5]
        
        # Multiplicative combination (conservative)
        filtering_rate = (self.base_rate *
                         confidence_factor *
                         uncertainty_penalty *
                         heterogeneity_boost *
                         spatial_boost)
        
        # Safety bounds: [5%, 40%]
        filtering_rate = np.clip(filtering_rate, self.min_rate, self.max_rate)
        
        return float(filtering_rate)
    
    # ==========================================================================
    # PHASE 3: ENSEMBLE STRATEGY GENERATION
    # ==========================================================================
    
    def _generate_ensemble_masks(self, scores, features, num_patches, slide_id):
        """
        Phase 3: Generate multiple candidate filtering strategies.
        Equal-weighted ensemble for robustness.
        
        Args:
            scores: Composite patch scores
            features: Dictionary of computed features
            num_patches: Number of patches
            slide_id: Slide identifier for logging
            
        Returns:
            Boolean mask (ensemble average > 0.5)
        """
        masks = []
        strategy_names = []
        filtering_rates_used = []
        
        # Strategy 1: Conservative (5% filtering)
        mask1 = self._generate_mask_at_rate(scores, 0.05, num_patches)
        masks.append(mask1)
        strategy_names.append('conservative_5pct')
        filtering_rates_used.append(0.05)
        
        # Strategy 2: Moderate (15% filtering)
        mask2 = self._generate_mask_at_rate(scores, 0.15, num_patches)
        masks.append(mask2)
        strategy_names.append('moderate_15pct')
        filtering_rates_used.append(0.15)
        
        # Strategy 3: Aggressive (30% filtering)
        mask3 = self._generate_mask_at_rate(scores, 0.30, num_patches)
        masks.append(mask3)
        strategy_names.append('aggressive_30pct')
        filtering_rates_used.append(0.30)
        
        # Strategy 4: Attention-only (ignore confidence, use parametric rate)
        attention_scores = features['attention_norm']
        rate4 = self._compute_parametric_filtering_rate(features)
        mask4 = self._generate_mask_at_rate(attention_scores, rate4, num_patches)
        masks.append(mask4)
        strategy_names.append('attention_only_parametric')
        filtering_rates_used.append(rate4)
        
        # Strategy 5: Parametric (full multi-signal with parametric rate)
        rate5 = self._compute_parametric_filtering_rate(features)
        mask5 = self._generate_mask_at_rate(scores, rate5, num_patches)
        masks.append(mask5)
        strategy_names.append('parametric_multi_signal')
        filtering_rates_used.append(rate5)
        
        # Equal-weight ensemble (most conservative, no overfitting)
        masks_array = np.array(masks, dtype=float)
        ensemble_probs = np.mean(masks_array, axis=0)
        final_mask = ensemble_probs > 0.5  # Majority voting
        
        # Log individual strategy decisions
        self.ensemble_log[slide_id] = {
            'strategy_names': strategy_names,
            'individual_kept_counts': [int(m.sum()) for m in masks],
            'ensemble_kept_count': int(final_mask.sum()),
            'filtering_rates_used': filtering_rates_used,
            'ensemble_agreement': float(np.mean([
                np.mean(m == final_mask) for m in masks
            ]))
        }
        
        return final_mask
    
    def _generate_mask_at_rate(self, scores, rate, num_patches):
        """
        Generate binary mask by filtering top 'rate' fraction of scores.
        
        Args:
            scores: Patch scores (higher = more demographic signal)
            rate: Filtering rate (fraction to remove)
            num_patches: Total number of patches
            
        Returns:
            Boolean mask (True = keep patch)
        """
        n_keep = int(num_patches * (1 - rate))
        n_keep = max(1, n_keep)  # Keep at least 1 patch
        
        # Sort by score (ascending) and keep LOWEST n_keep
        # (LOW score = LOW demographic signal = KEEP for mutation prediction)
        sorted_indices = np.argsort(scores)
        keep_indices = sorted_indices[:n_keep]
        
        mask = np.zeros(num_patches, dtype=bool)
        mask[keep_indices] = True
        
        return mask
    
    # ==========================================================================
    # LOGGING AND DIAGNOSTICS
    # ==========================================================================
    
    def _log_unified_decision(self, slide_id, num_patches, keep_mask, features):
        """Log filtering decision with unified features for this slide."""
        # Compute parametric filtering rate for logging
        parametric_rate = self._compute_parametric_filtering_rate(features)
        
        self.decisions_log[slide_id] = {
            'num_patches_original': int(num_patches),
            'num_patches_kept': int(keep_mask.sum()),
            'filtering_mode': 'unified_ensemble',
            'parametric_filtering_rate': float(parametric_rate),
            'filtered_patch_indices': np.where(~keep_mask)[0].tolist(),
            'features': {
                'confidence': float(features['confidence']),
                'uncertainty': float(features['uncertainty']),
                'heterogeneity': float(features['heterogeneity']),
                'spatial_clustering': float(features['spatial_clustering']),
                'attention_mean': float(features['attention_mean']),
                'patch_entropy': float(features['patch_entropy'])
            }
        }
        
        # Store features for later analysis
        self.features_log[slide_id] = features
    
    def _generate_random_filter_mask_adaptive(self, num_patches, features, slide_id):
        """
        Generate random filter mask with adaptive rate (for wrong predictions).
        
        When demographic prediction is wrong, use random selection instead of
        attention to avoid introducing demographic bias.
        
        Adaptive strategy based on estimated dataset size:
        - Small datasets (< 100 total slides): Filter less (preserve critical data)
        - Medium datasets (100-300 slides): Moderate filtering
        - Large datasets (> 300 slides): Standard filtering
        
        Args:
            num_patches: Number of patches
            features: Feature dictionary (contains prediction info)
            slide_id: Slide identifier for logging
            
        Returns:
            Boolean mask (True = keep patch)
        """
        # Estimate dataset size from total slides processed so far
        # This is a rough proxy - smaller datasets need more data preservation
        total_slides = self.stats['total_slides_processed']
        
        # Adaptive filtering rate based on dataset size (proxy for minority group size)
        base_conf = features.get('base_confidence', 0.05)
        
        if total_slides < 100:
            # Small dataset - very conservative filtering (0x when wrong)
            wrong_factor = 0.0
        elif total_slides < 300:
            # Medium dataset - moderate reduction (0.1x when wrong)
            wrong_factor = 0.1
        else:
            # Large dataset - standard reduction (0.3x when wrong)
            wrong_factor = 0.3
        
        # Calculate adaptive filtering rate
        effective_conf = base_conf * wrong_factor
        filter_rate = self.base_rate * effective_conf / base_conf if base_conf > 0 else 0
        filter_rate = np.clip(filter_rate, 0.0, self.max_rate)
        
        # Number of patches to remove
        n_remove = int(num_patches * filter_rate)
        
        # RANDOM selection of patches to remove (no demographic bias)
        keep_mask = np.ones(num_patches, dtype=bool)
        
        if n_remove > 0 and n_remove < num_patches:
            # Randomly select patches to remove
            remove_indices = np.random.choice(num_patches, size=n_remove, replace=False)
            keep_mask[remove_indices] = False
        
        # Log random filtering decision
        if 'random_filtering' not in self.stats:
            self.stats['v3_random_filtering'] = 0
        self.stats['v3_random_filtering'] += 1
        
        if slide_id not in self.decisions_log:
            self.decisions_log[slide_id] = {}
        
        self.decisions_log[slide_id].update({
            'v3_mode': 'random_when_wrong',
            'wrong_factor': float(wrong_factor),
            'dataset_size_estimate': int(total_slides),
            'random_filter_rate': float(filter_rate)
        })
        
        return keep_mask
    
    def save_logs(self):
        """Save decision logs, features, and ensemble information."""
        if not self.save_dir:
            return
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save decisions (inherited method, but call it)
        torch.save(self.decisions_log, self.save_dir / 'agent_decisions.pt')
        print(f"[UnifiedAgent] Decisions saved: {len(self.decisions_log)} slides")
        
        # Save ensemble log (NEW for unified agent)
        if self.ensemble_log:
            with open(self.save_dir / 'unified_ensemble_log.json', 'w') as f:
                json.dump(self.ensemble_log, f, indent=2)
            print(f"[UnifiedAgent] Ensemble log saved: {len(self.ensemble_log)} slides")
        
        # Save features log (NEW for unified agent)
        if self.features_log:
            torch.save(self.features_log, self.save_dir / 'unified_features_log.pt')
            print(f"[UnifiedAgent] Features log saved: {len(self.features_log)} slides")
        
        # Compute and save summary
        summary = {
            'agent_type': 'unified',
            'total_slides_processed': int(self.stats['total_slides_processed']),
            'total_patches_original': int(self.stats['total_patches_original']),
            'total_patches_kept': int(self.stats['total_patches_kept']),
            'overall_filtering_rate': float(1 - (self.stats['total_patches_kept'] /
                                           max(self.stats['total_patches_original'], 1))),
            'slides_with_no_demographic_data': int(self.stats['slides_with_no_demographic_data']),
            'slides_with_length_mismatch': int(self.stats['slides_with_length_mismatch']),
            'slides_with_coordinate_mismatch': int(self.stats['slides_with_coordinate_mismatch']),
            'coordinate_matched_successfully': int(self.stats['coordinate_matched_successfully']),
            'filtering_mode': 'unified_parametric_ensemble',
            'base_rate': float(self.base_rate),
            'rate_bounds': [float(self.min_rate), float(self.max_rate)],
            'ensemble_strategies': 5,
            'coordinate_matching_enabled': True
        }
        
        # Add ensemble statistics
        if self.ensemble_log:
            agreements = [log['ensemble_agreement'] for log in self.ensemble_log.values()]
            summary['ensemble_statistics'] = {
                'mean_agreement': float(np.mean(agreements)),
                'std_agreement': float(np.std(agreements)),
                'min_agreement': float(np.min(agreements)),
                'max_agreement': float(np.max(agreements))
            }
        
        # Add feature statistics
        if self.features_log:
            confidences = [f['confidence'] for f in self.features_log.values()]
            uncertainties = [f['uncertainty'] for f in self.features_log.values()]
            heterogeneities = [f['heterogeneity'] for f in self.features_log.values()]
            
            summary['feature_statistics'] = {
                'confidence': {
                    'mean': float(np.mean(confidences)),
                    'std': float(np.std(confidences)),
                    'quartiles': [float(q) for q in np.percentile(confidences, [25, 50, 75])]
                },
                'uncertainty': {
                    'mean': float(np.mean(uncertainties)),
                    'std': float(np.std(uncertainties)),
                    'quartiles': [float(q) for q in np.percentile(uncertainties, [25, 50, 75])]
                },
                'heterogeneity': {
                    'mean': float(np.mean(heterogeneities)),
                    'std': float(np.std(heterogeneities)),
                    'quartiles': [float(q) for q in np.percentile(heterogeneities, [25, 50, 75])]
                }
            }
        
        with open(self.save_dir / 'agent_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[UnifiedAgent] Summary saved to {self.save_dir / 'agent_summary.json'}")
        print(f"[UnifiedAgent] Total patches: {summary['total_patches_original']}, "
              f"Kept: {summary['total_patches_kept']}, "
              f"Filtered: {summary['overall_filtering_rate']:.2%}")

