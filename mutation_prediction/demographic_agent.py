"""
Demographic-informed patch filtering agent.
Filters patches based on demographic classifier attention and confidence.
Uses coordinate-based patch matching to handle pipeline differences.
"""

import torch
import numpy as np
import json
import h5py
from pathlib import Path
from datetime import datetime
from demographic_config_lookup import get_best_demographic_config


class DemographicPatchAgent:
    def __init__(self, attribute, cancer, foundation_model, gene,
                 strategy='attention_confidence',
                 base_filter_percentile=25,
                 adaptive_filtering=False,
                 use_correctness_weighting=False,
                 demographic_inference_base_path='./data/demographic_inferences/',
                 coord_base_path='./data/coords/',
                 save_dir=None):
        """
        Agent that filters patches based on demographic signals.
        
        Args:
            attribute: Demographic attribute ('Age', 'Race', 'Sex')
            cancer: Cancer type ('BRCA', 'LGG', 'GBM', 'KIRC')
            foundation_model: Foundation model name
            gene: Gene name for mutation prediction
            strategy: 'attention_confidence', 'random', or 'none'
            base_filter_percentile: Base percentage for filtering (10, 25, 50)
            adaptive_filtering: If True, scale percentile by slide confidence
            use_correctness_weighting: If True, weight confidence by prediction correctness
            demographic_inference_base_path: Path to demographic inference data (default: './data/demographic_inferences/')
            coord_base_path: Path to WSI coordinate files (default: './data/coords/')
            save_dir: Directory to save agent logs
        """
        
        self.strategy = strategy
        self.base_filter_percentile = base_filter_percentile
        self.adaptive_filtering = adaptive_filtering
        self.use_correctness_weighting = use_correctness_weighting
        self.alpha_attention = 0.7
        self.alpha_confidence = 0.3
        self.save_dir = Path(save_dir) if save_dir else None
        self.coord_base_path = Path(coord_base_path)
        self.cancer = cancer.upper()
        self.foundation_model = foundation_model
        
        # Get best demographic model config (with graceful fallback)
        try:
            config = get_best_demographic_config(cancer, attribute, foundation_model)
            self.demo_config = config
            self.demo_config['attribute'] = attribute
            self.demo_config['cancer'] = cancer
            self.demo_config['foundation_model'] = foundation_model
            self.demographic_model_available = True
        except ValueError as e:
            print(f"[Agent] WARNING: No demographic model found for Cancer={cancer}, Attribute={attribute}, Model={foundation_model}")
            print(f"[Agent] Agent will NOT filter patches (fallback to baseline)")
            self.demographic_model_available = False
            self.demo_config = {'attribute': attribute, 'cancer': cancer, 'foundation_model': foundation_model}
            self.attention_data = {}
            self.prediction_data = {}
            print(f"[Agent] Strategy: {strategy} (DISABLED - no demographic model)")
            # Skip the rest of initialization
            self.patient_to_slides = {}
            self.stats = {
                'total_slides_processed': 0,
                'total_patches_original': 0,
                'total_patches_kept': 0,
                'slides_with_no_demographic_data': 0,
                'slides_with_length_mismatch': 0,
                'slides_with_coordinate_mismatch': 0,
                'coordinate_matched_successfully': 0,
                'adaptive_percentiles': [],
                'confidence_scores': []
            }
            self.decisions_log = {}
            return
        
        # Build path to demographic data
        attr_lower = attribute.lower()
        cancer_code = config['cancer_code']
        exp = config['exp_id']
        fold = config['fold_id']
        metric = config['metric']
        
        base_path = Path(demographic_inference_base_path)
        demo_dir = base_path / attr_lower / cancer_code / 'TCGA' / \
                   f"{exp}_cv_fold_{fold}_{metric}"
        
        if not demo_dir.exists():
            raise FileNotFoundError(f"Demographic inference directory not found: {demo_dir}")
        
        self.demo_config['inference_path'] = str(demo_dir)
        
        # Load demographic data
        self.attention_data = self._load_attention(demo_dir)
        self.prediction_data = self._load_predictions(demo_dir)
        
        # Build patient-to-slides mapping for patient-level ID matching
        self.patient_to_slides = self._build_patient_mapping()
        
        print(f"[Agent] Loaded demographic data: {len(self.attention_data)} slides")
        print(f"[Agent] Unique patients in demographic data: {len(self.patient_to_slides)}")
        print(f"[Agent] Strategy: {strategy}, Base percentile: {base_filter_percentile}%, Adaptive: {adaptive_filtering}")
        
        # Initialize logging
        self.decisions_log = {}
        self.stats = {
            'total_slides_processed': 0,
            'total_patches_original': 0,
            'total_patches_kept': 0,
            'slides_with_no_demographic_data': 0,
            'slides_with_length_mismatch': 0,
            'slides_with_coordinate_mismatch': 0,
            'coordinate_matched_successfully': 0,
            'adaptive_percentiles': [],
            'confidence_scores': []
        }
        
        # Save config
        if self.save_dir:
            self._save_config(gene)
        
    def _load_attention(self, demo_dir):
        """Load intermediate_deep_attention.pt"""
        path = demo_dir / 'intermediate_deep_attention.pt'
        if not path.exists():
            raise FileNotFoundError(f"Attention file not found: {path}")
        return torch.load(path, map_location='cpu')
        
    def _load_predictions(self, demo_dir):
        """Load inference_log.json"""
        path = demo_dir / 'inference_log.json'
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        return data['inferences']
    
    def _build_patient_mapping(self):
        """
        Build mapping from patient-level barcode (TCGA-XX-XXXX) to full slide IDs.
        Demographic data uses full slide IDs like: TCGA-05-4244-01A-01-BS1.{uuid}.pt
        Mutation data uses patient IDs like: TCGA-4H-AAAK
        """
        patient_to_slides = {}
        
        for slide_id in self.attention_data.keys():
            # Extract patient barcode from demographic slide ID
            # Format: TCGA-05-4244-01A-01-BS1.UUID.pt -> TCGA-05-4244
            # Or: TCGA-XX-XXXX-XXX-XXX-XXX.UUID.pt -> TCGA-XX-XXXX
            parts = slide_id.split('-')
            if len(parts) >= 3:
                # Take first 3 parts: TCGA-XX-XXXX
                patient_barcode = '-'.join(parts[:3])
                
                if patient_barcode not in patient_to_slides:
                    patient_to_slides[patient_barcode] = []
                patient_to_slides[patient_barcode].append(slide_id)
        
        return patient_to_slides
        
    def _normalize_slide_id(self, slide_id, is_demographic=False):
        """
        Normalize slide ID to extract patient barcode.
        
        Args:
            slide_id: Input slide ID (various formats)
            is_demographic: If True, return list of demographic slide IDs for this patient
        
        Returns:
            If is_demographic=False: patient barcode (TCGA-XX-XXXX)
            If is_demographic=True: list of full demographic slide IDs
        """
        # Extract patient-level barcode (TCGA-XX-XXXX)
        if slide_id.endswith('.pt'):
            slide_id = slide_id[:-3]
        
        # Handle various formats
        parts = slide_id.split('-')
        if len(parts) >= 3:
            patient_barcode = '-'.join(parts[:3])  # TCGA-XX-XXXX
        else:
            patient_barcode = slide_id
        
        if is_demographic:
            # Return list of demographic slides for this patient
            return self.patient_to_slides.get(patient_barcode, [])
        else:
            # Return patient barcode
            return patient_barcode
    
    def _load_coordinates(self, slide_id, slide_type):
        """
        Load patch coordinates from h5 file.
        
        Args:
            slide_id: Slide identifier (with .pt extension)
            slide_type: 'FS' or 'PM'
            
        Returns:
            np.ndarray: Array of (x, y) coordinates, or None if not found
        """
        # Build coordinate file path
        coord_filename = slide_id.replace('.pt', '.h5')
        cancer_dir = f"TCGA-{self.cancer}-{slide_type}"
        coord_path = self.coord_base_path / cancer_dir / "coords" / coord_filename
        
        if not coord_path.exists():
            return None
        
        try:
            with h5py.File(coord_path, 'r') as f:
                coords = f['coords'][:]
                # Convert structured array to regular array of tuples for easier matching
                return np.array([(int(c['x']), int(c['y'])) for c in coords])
        except Exception as e:
            print(f"[Agent] Warning: Could not load coordinates from {coord_path}: {e}")
            return None
    
    def _match_patches_by_coordinates(self, demo_attention, demo_coords, mutation_coords):
        """
        Match demographic attention scores to mutation patches using coordinates.
        
        Args:
            demo_attention: List of attention scores from demographic model
            demo_coords: np.ndarray of (x, y) coordinates from demographic features
            mutation_coords: np.ndarray of (x, y) coordinates from mutation features
            
        Returns:
            np.ndarray: Matched attention scores aligned with mutation patches
        """
        matched_attention = []
        
        # Create a dictionary for fast lookup: coord -> attention
        demo_coord_dict = {tuple(coord): att for coord, att in zip(demo_coords, demo_attention)}
        
        # Match each mutation coordinate
        for mut_coord in mutation_coords:
            mut_coord_tuple = tuple(mut_coord)
            if mut_coord_tuple in demo_coord_dict:
                matched_attention.append(demo_coord_dict[mut_coord_tuple])
            else:
                # No match found - use median attention as fallback
                matched_attention.append(np.median(demo_attention))
        
        return np.array(matched_attention)
        
    def get_patch_filter_mask(self, slide_id, num_patches, mutation_coords=None, slide_type='FS'):
        """
        Main interface - returns boolean mask for patches to keep.
        
        Args:
            slide_id: Slide identifier
            num_patches: Number of patches in mutation prediction data
            mutation_coords: Optional np.ndarray of (x, y) coordinates for mutation patches
            slide_type: 'FS' or 'PM' slide type
            
        Returns:
            np.ndarray: Boolean mask of shape (num_patches,)
        """
        self.stats['total_slides_processed'] += 1
        self.stats['total_patches_original'] += num_patches
        
        # If no demographic model available, don't filter
        if not self.demographic_model_available:
            keep_mask = np.ones(num_patches, dtype=bool)
            self.stats['total_patches_kept'] += num_patches
            return keep_mask
        
        if self.strategy == 'none':
            keep_mask = np.ones(num_patches, dtype=bool)
            self.stats['total_patches_kept'] += num_patches
            return keep_mask
            
        if self.strategy == 'random':
            keep_mask = self._random_filter(num_patches)
            self._log_decision(slide_id, num_patches, keep_mask, None, None, None)
            return keep_mask
            
        if self.strategy == 'attention_confidence':
            keep_mask = self._attention_confidence_filter(slide_id, num_patches, mutation_coords, slide_type)
            return keep_mask
        
        # Default: no filtering
        keep_mask = np.ones(num_patches, dtype=bool)
        self.stats['total_patches_kept'] += num_patches
        return keep_mask
            
    def _random_filter(self, num_patches):
        """Random control filtering (always uses base percentile)"""
        n_keep = int(num_patches * (1 - self.base_filter_percentile / 100))
        n_keep = max(1, n_keep)  # Keep at least 1 patch
        keep_indices = np.random.choice(num_patches, n_keep, replace=False)
        keep_mask = np.zeros(num_patches, dtype=bool)
        keep_mask[keep_indices] = True
        
        self.stats['total_patches_kept'] += n_keep
        return keep_mask
        
    def _attention_confidence_filter(self, slide_id, num_patches, mutation_coords=None, slide_type='FS'):
        """Filter based on attention + confidence with optional adaptive k and coordinate matching"""
        # Extract patient barcode from mutation slide ID
        patient_barcode = self._normalize_slide_id(slide_id, is_demographic=False)
        
        # Get list of demographic slides for this patient
        demographic_slide_keys = self._normalize_slide_id(patient_barcode, is_demographic=True)
        
        # Find first available demographic slide for this patient
        slide_key = None
        for key in demographic_slide_keys:
            if key in self.attention_data:
                slide_key = key
                break
        
        # Check if patient has demographic data
        if slide_key is None:
            self.stats['slides_with_no_demographic_data'] += 1
            keep_mask = np.ones(num_patches, dtype=bool)
            self.stats['total_patches_kept'] += num_patches
            return keep_mask
            
        attention_list = self.attention_data[slide_key]
        
        # Handle length mismatch with coordinate-based matching
        if len(attention_list) != num_patches:
            self.stats['slides_with_length_mismatch'] += 1
            
            # Try coordinate-based matching
            if mutation_coords is None:
                # Load coordinates from file
                mutation_coords = self._load_coordinates(slide_key, slide_type)
            
            if mutation_coords is not None and len(mutation_coords) == num_patches:
                # Load demographic coordinates
                demo_coords = self._load_coordinates(slide_key, slide_type)
                
                if demo_coords is not None and len(demo_coords) == len(attention_list):
                    # Perform coordinate-based matching
                    attention_list = self._match_patches_by_coordinates(
                        attention_list, demo_coords, mutation_coords
                    )
                    self.stats['coordinate_matched_successfully'] += 1
                else:
                    # Fallback: use position-based resampling
                    self.stats['slides_with_coordinate_mismatch'] += 1
                    if len(attention_list) > num_patches:
                        indices = np.random.choice(len(attention_list), num_patches, replace=False)
                        attention_list = [attention_list[i] for i in sorted(indices)]
                    else:
                        median_attention = np.median(attention_list)
                        attention_list = list(attention_list) + [median_attention] * (num_patches - len(attention_list))
            else:
                # Fallback: use position-based resampling
                self.stats['slides_with_coordinate_mismatch'] += 1
                if len(attention_list) > num_patches:
                    indices = np.random.choice(len(attention_list), num_patches, replace=False)
                    attention_list = [attention_list[i] for i in sorted(indices)]
                else:
                    median_attention = np.median(attention_list)
                    attention_list = list(attention_list) + [median_attention] * (num_patches - len(attention_list))
            
        # Get slide-level prediction and confidence
        pred_info = self.prediction_data.get(slide_key, {})
        prediction = pred_info.get('prediction', [0.5])[0]
        slide_confidence = abs(prediction - 0.5) * 2  # Range [0, 1]
        
        # Correctness-weighted confidence (Strategy 2)
        if self.use_correctness_weighting and 'target' in pred_info:
            target = pred_info.get('target', [0.5])[0]
            predicted_class = 1 if prediction > 0.5 else 0
            true_class = int(target)
            is_correct = (predicted_class == true_class)
            
            if is_correct:
                # Correct prediction - use full confidence
                effective_confidence = slide_confidence
            else:
                # Wrong prediction - heavily reduce confidence (30% weight)
                effective_confidence = slide_confidence * 0.3
        else:
            # No correctness weighting
            effective_confidence = slide_confidence
        
        # ADAPTIVE FILTERING: Scale percentile by effective confidence
        if self.adaptive_filtering:
            adaptive_percentile = self.base_filter_percentile * effective_confidence
        else:
            adaptive_percentile = self.base_filter_percentile
        
        self.stats['adaptive_percentiles'].append(adaptive_percentile)
        self.stats['confidence_scores'].append(slide_confidence)
        if self.use_correctness_weighting:
            if 'effective_confidence_scores' not in self.stats:
                self.stats['effective_confidence_scores'] = []
            self.stats['effective_confidence_scores'].append(effective_confidence)
        
        # Weighted scoring for patch ranking
        attention_array = np.array(attention_list)
        attention_norm = attention_array / (np.max(attention_array) + 1e-8)
        
        # Score combines attention (which patches) + effective confidence (slide-level signal)
        scores = self.alpha_attention * attention_norm + \
                 self.alpha_confidence * effective_confidence
        
        # Remove adaptive_percentile% HIGHEST attention patches (keep lowest demographic signal)
        if adaptive_percentile > 0:
            n_keep = int(num_patches * (1 - adaptive_percentile / 100))
            n_keep = max(1, n_keep)  # Keep at least 1 patch
            
            # Sort by score and keep BOTTOM n_keep patches (lowest demographic scores)
            # HIGH attention = high demographic signal = REMOVE to reduce bias
            # LOW attention = low demographic signal = KEEP for mutation prediction
            sorted_indices = np.argsort(scores)  # Ascending order (lowest first)
            keep_indices = sorted_indices[:n_keep]  # Take FIRST n_keep (lowest scores)
            keep_mask = np.zeros(num_patches, dtype=bool)
            keep_mask[keep_indices] = True
        else:
            keep_mask = np.ones(num_patches, dtype=bool)
        
        self.stats['total_patches_kept'] += keep_mask.sum()
        
        # Log decision
        self._log_decision(slide_id, num_patches, keep_mask, 
                          attention_array, slide_confidence, adaptive_percentile)
        
        return keep_mask
        
    def _log_decision(self, slide_id, num_patches, keep_mask, 
                     attention, confidence, adaptive_k):
        """Log filtering decision for this slide"""
        self.decisions_log[slide_id] = {
            'num_patches_original': int(num_patches),
            'num_patches_kept': int(keep_mask.sum()),
            'base_filter_percentile': self.base_filter_percentile,
            'slide_confidence': float(confidence) if confidence is not None else None,
            'adaptive_percentile': float(adaptive_k) if adaptive_k is not None else None,
            'filtering_mode': 'adaptive' if self.adaptive_filtering else 'fixed',
            'mean_attention': float(attention.mean()) if attention is not None else None,
            'filtered_patch_indices': np.where(~keep_mask)[0].tolist()
        }
        
    def _save_config(self, gene):
        """Save agent configuration"""
        if not self.save_dir:
            return
            
        config = {
            'agent_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'strategy': self.strategy,
            'filtering_mode': 'adaptive' if self.adaptive_filtering else 'fixed',
            'base_filter_percentile': self.base_filter_percentile,
            'adaptive_scaling': 'confidence_based' if self.adaptive_filtering else None,
            'alpha_attention': self.alpha_attention,
            'alpha_confidence': self.alpha_confidence,
            'demographic_source': self.demo_config,
            'mutation_task': {
                'cancer': self.demo_config['cancer'],
                'gene': gene,
                'sensitive_attribute': self.demo_config['attribute'],
                'foundation_model': self.demo_config['foundation_model']
            },
            'filtering_scope': 'training_only',
            'slide_id_normalization': 'add_pt_extension',
            'length_mismatch_strategy': 'keep_all_if_mismatch'
        }
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.save_dir / 'agent_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[Agent] Config saved to {self.save_dir / 'agent_config.json'}")
            
    def save_logs(self):
        """Save decision logs and summary statistics"""
        if not self.save_dir:
            return
            
        # Save decisions
        torch.save(self.decisions_log, self.save_dir / 'agent_decisions.pt')
        print(f"[Agent] Decisions saved: {len(self.decisions_log)} slides")
        
        # Compute and save summary
        adaptive_percentiles = self.stats['adaptive_percentiles']
        confidence_scores = self.stats['confidence_scores']
        
        summary = {
            'total_slides_processed': int(self.stats['total_slides_processed']),
            'total_patches_original': int(self.stats['total_patches_original']),
            'total_patches_kept': int(self.stats['total_patches_kept']),
            'overall_filtering_rate': float(1 - (self.stats['total_patches_kept'] / 
                                           max(self.stats['total_patches_original'], 1))),
            'slides_with_no_demographic_data': int(self.stats['slides_with_no_demographic_data']),
            'slides_with_length_mismatch': int(self.stats['slides_with_length_mismatch']),
            'slides_with_coordinate_mismatch': int(self.stats['slides_with_coordinate_mismatch']),
            'coordinate_matched_successfully': int(self.stats['coordinate_matched_successfully']),
            'filtering_mode': 'adaptive' if self.adaptive_filtering else 'fixed',
            'base_filter_percentile': int(self.base_filter_percentile),
            'coordinate_matching_enabled': True
        }
        
        if adaptive_percentiles:
            summary['actual_filtering_rates'] = {
                'mean': float(np.mean(adaptive_percentiles)),
                'std': float(np.std(adaptive_percentiles)),
                'min': float(np.min(adaptive_percentiles)),
                'max': float(np.max(adaptive_percentiles)),
                'median': float(np.median(adaptive_percentiles)),
                'quartiles': [float(q) for q in np.percentile(adaptive_percentiles, [25, 50, 75])]
            }
        
        if confidence_scores:
            summary['confidence_distribution'] = {
                'mean': float(np.mean(confidence_scores)),
                'std': float(np.std(confidence_scores)),
                'quartiles': [float(q) for q in np.percentile(confidence_scores, [25, 50, 75])]
            }
        
        with open(self.save_dir / 'agent_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Agent] Summary saved to {self.save_dir / 'agent_summary.json'}")
        print(f"[Agent] Total patches: {summary['total_patches_original']}, Kept: {summary['total_patches_kept']}, Filtered: {summary['overall_filtering_rate']:.2%}")

