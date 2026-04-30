import numpy as np
import torch
from typing import Dict, List, Optional, Union


class IncrementalMetrics:
    """Incremental segmentation metrics calculator with O(1) memory usage.
    
    Args:
        num_classes (int): Number of classes
        ignore_index (int): Label index to ignore, default 255
        metrics (List[str]): Metrics to calculate, supports ['mIoU', 'mDice']
    """
    
    def __init__(self, 
                 num_classes: int, 
                 ignore_index: int = 255,
                 metrics: List[str] = ['mIoU']):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metrics = metrics
        
        allowed_metrics = ['mIoU', 'mDice']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise ValueError(f'Unsupported metrics: {set(metrics) - set(allowed_metrics)}')
        
        self.reset()
    
    def reset(self):
        self.total_area_intersect = np.zeros((self.num_classes,), dtype=np.float64)
        self.total_area_union = np.zeros((self.num_classes,), dtype=np.float64)  
        self.total_area_pred_label = np.zeros((self.num_classes,), dtype=np.float64)
        self.total_area_label = np.zeros((self.num_classes,), dtype=np.float64)
    
    def add_batch(self, pred_label: Union[torch.Tensor, np.ndarray], 
                  label: Union[torch.Tensor, np.ndarray]):
        """Add batch predictions and labels, update accumulated statistics.
        
        Args:
            pred_label: Predicted labels, shape (H, W) or (B, H, W)
            label: Ground truth labels, shape (H, W) or (B, H, W)
        """
        if isinstance(pred_label, torch.Tensor):
            pred_label = pred_label.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
            
        pred_label = pred_label.astype(np.int64)
        label = label.astype(np.int64)
        
        if pred_label.ndim == 3:  # (B, H, W)
            for i in range(pred_label.shape[0]):
                self._add_single_image(pred_label[i], label[i])
        elif pred_label.ndim == 2:  # (H, W)
            self._add_single_image(pred_label, label)
        else:
            raise ValueError(f'not support tensor dimension: {pred_label.ndim}')
    
    def _add_single_image(self, pred_label: np.ndarray, label: np.ndarray):
        """Add single image prediction and label."""
        mask = (label != self.ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]
        intersect = pred_label[pred_label == label]
        
        area_intersect = np.histogram(intersect, bins=np.arange(self.num_classes + 1))[0]
        area_pred_label = np.histogram(pred_label, bins=np.arange(self.num_classes + 1))[0]
        area_label = np.histogram(label, bins=np.arange(self.num_classes + 1))[0]
        area_union = area_pred_label + area_label - area_intersect
        self.total_area_intersect += area_intersect.astype(np.float64)
        self.total_area_union += area_union.astype(np.float64)
        self.total_area_pred_label += area_pred_label.astype(np.float64)
        self.total_area_label += area_label.astype(np.float64)
    
    def compute(self, logger=None) -> Dict[str, float]:
        """Compute final evaluation metrics.
        
        Args:
            logger: Logger to print detailed per-class metrics
            
        Returns:
            Dict[str, float]: Metrics including aAcc, mIoU, mDice
        """
        eps = 1e-10
        
        # Only compute metrics for drivable_area class (index 1)
        dr_idx = 1
        acc_per_class = self.total_area_intersect / (self.total_area_label + eps)
        overall_acc = np.sum(self.total_area_intersect) / (np.sum(self.total_area_label) + eps)
        
        results = {
            'aAcc': float(acc_per_class[dr_idx]),
            'overall_acc': float(overall_acc)
        }
        
        # Calculate per-class metrics for detailed logging
        per_class_metrics = {}
        
        for metric in self.metrics:
            if metric == 'mIoU':
                iou_per_class = self.total_area_intersect / (self.total_area_union + eps)
                results['mIoU'] = float(iou_per_class[dr_idx])
                per_class_metrics['IoU'] = iou_per_class
                
            elif metric == 'mDice':
                dice_per_class = (2 * self.total_area_intersect) / (
                    self.total_area_pred_label + self.total_area_label + eps)
                results['mDice'] = float(dice_per_class[dr_idx])
                per_class_metrics['Dice'] = dice_per_class
        
        # Log detailed per-class results
        if logger is not None:
            # Build complete table as single string to avoid duplicate logging
            lines = []
            lines.append("Per class results:")
            
            # Define class names for binary segmentation
            class_names = ['others', 'drivable_area']
            
            # Define column widths for strict alignment
            col_widths = [15, 8, 8]  # Class name, IoU/Dice, Acc
            
            # Create table header
            header_cols = ['Class'] + [f'{k}' for k in per_class_metrics.keys()] + ['Acc']
            
            # Format header with strict alignment
            header_line = ""
            for i, (col, width) in enumerate(zip(header_cols, col_widths)):
                header_line += f"{col:>{width}}"
            lines.append(header_line)
            
            # Add separator line
            separator = ""
            for width in col_widths:
                separator += "-" * width
            lines.append(separator)
            
            # Add each class results
            for i in range(self.num_classes):
                class_name = class_names[i] if i < len(class_names) else f"Class{i}"
                row_data = [class_name]
                
                # Add metric values
                for metric_name, metric_values in per_class_metrics.items():
                    row_data.append(f"{metric_values[i]*100:.2f}")
                row_data.append(f"{acc_per_class[i]*100:.2f}")
                
                # Format row with strict alignment
                row_line = ""
                for j, (data, width) in enumerate(zip(row_data, col_widths)):
                    row_line += f"{data:>{width}}"
                lines.append(row_line)
            
            # Add separator before summary
            lines.append(separator)
            
            # Add summary results
            summary_data = ['Global']
            for metric_name in per_class_metrics.keys():
                if metric_name == 'IoU':
                    summary_data.append(f"{results.get('mIoU', 0.0)*100:.2f}")
                elif metric_name == 'Dice':
                    summary_data.append(f"{results.get('mDice', 0.0)*100:.2f}")
            summary_data.append(f"{results['aAcc']*100:.2f}")
            
            # Format summary with strict alignment
            summary_line = ""
            for j, (data, width) in enumerate(zip(summary_data, col_widths)):
                summary_line += f"{data:>{width}}"
            lines.append(summary_line)
            
            # Log entire table as single message
            logger.info("\n" + "\n".join(lines))
        
        return results
    
    def get_per_class_results(self) -> Dict[str, np.ndarray]:
        """Get per-class detailed metrics."""
        eps = 1e-10
        
        results = {
            'per_class_acc': self.total_area_intersect / (self.total_area_label + eps)
        }
        
        for metric in self.metrics:
            if metric == 'mIoU':
                results['per_class_iou'] = self.total_area_intersect / (self.total_area_union + eps)
            elif metric == 'mDice':
                results['per_class_dice'] = (2 * self.total_area_intersect) / (
                    self.total_area_pred_label + self.total_area_label + eps)
                
        return results


def create_evaluator(num_classes: int, 
                    ignore_index: int = 255, 
                    metrics: List[str] = ['mIoU']) -> IncrementalMetrics:
    """Create incremental metrics evaluator."""
    return IncrementalMetrics(num_classes=num_classes, 
                             ignore_index=ignore_index, 
                             metrics=metrics)


def test_metrics():
    """Comprehensive test cases with manually calculated expected results."""
    
    print("=== Testing Incremental Metrics Calculator ===\n")
    
    # Test Case 1: Perfect Prediction
    print("Test Case 1: Perfect Prediction")
    pred1 = np.array([[0, 1, 0], 
                      [1, 0, 1], 
                      [0, 1, 0]])
    gt1 = np.array([[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]])
    
    evaluator1 = create_evaluator(num_classes=2, metrics=['mIoU', 'mDice'])
    evaluator1.add_batch(pred1, gt1)
    results1 = evaluator1.compute()  # Test without logger
    
    print(f"Prediction:\n{pred1}")
    print(f"Ground Truth:\n{gt1}")
    print(f"Expected: aAcc=1.0, mAcc=1.0, mIoU=1.0, mDice=1.0")
    print(f"Actual:   ", end="")
    for k, v in results1.items():
        print(f"{k}={v:.3f}, ", end="")
    print(f"\n{'✓ PASS' if all(abs(v - 1.0) < 1e-6 for v in results1.values()) else '✗ FAIL'}\n")
    
    # Test Case 2: Completely Wrong Prediction  
    print("Test Case 2: Completely Wrong Prediction")
    pred2 = np.array([[1, 0, 1], 
                      [0, 1, 0], 
                      [1, 0, 1]])
    gt2 = np.array([[0, 1, 0], 
                    [1, 0, 1], 
                    [0, 1, 0]])
    
    evaluator2 = create_evaluator(num_classes=2, metrics=['mIoU', 'mDice'])
    evaluator2.add_batch(pred2, gt2)
    results2 = evaluator2.compute()
    
    print(f"Prediction:\n{pred2}")
    print(f"Ground Truth:\n{gt2}")
    print(f"Expected: aAcc=0.0, mAcc=0.0, mIoU=0.0, mDice=0.0")
    print(f"Actual:   ", end="")
    for k, v in results2.items():
        print(f"{k}={v:.3f}, ", end="")
    print(f"\n{'✓ PASS' if all(abs(v - 0.0) < 1e-6 for v in results2.values()) else '✗ FAIL'}\n")
    
    # Test Case 3: Partial Correct Prediction (Manual Calculation)
    print("Test Case 3: Partial Correct Prediction")
    pred3 = np.array([[0, 0], 
                      [1, 1]])  # pred: class 0 has 2 pixels, class 1 has 2 pixels
    gt3 = np.array([[0, 1], 
                    [1, 0]])   # gt: class 0 has 2 pixels, class 1 has 2 pixels
    
    # Manual calculation:
    # Total pixels: 4
    # Correct predictions: (0,0)=✓, (0,1)=✗, (1,0)=✓, (1,1)=✗ -> 2 correct
    # aAcc = 2/4 = 0.5
    
    # Class 0: pred=2, gt=2, intersect=1 -> IoU = 1/(2+2-1) = 1/3 ≈ 0.333
    # Class 1: pred=2, gt=2, intersect=1 -> IoU = 1/(2+2-1) = 1/3 ≈ 0.333  
    # mIoU = (1/3 + 1/3) / 2 = 1/3 ≈ 0.333
    
    # Class 0: Acc = 1/2 = 0.5
    # Class 1: Acc = 1/2 = 0.5
    # mAcc = (0.5 + 0.5) / 2 = 0.5
    
    evaluator3 = create_evaluator(num_classes=2, metrics=['mIoU', 'mDice'])
    evaluator3.add_batch(pred3, gt3)
    results3 = evaluator3.compute()
    
    expected3 = {'aAcc': 0.5, 'mAcc': 0.5, 'mIoU': 1/3, 'mDice': 0.5}
    
    print(f"Prediction:\n{pred3}")
    print(f"Ground Truth:\n{gt3}")
    print(f"Expected: aAcc=0.500, mAcc=0.500, mIoU=0.333, mDice=0.500")
    print(f"Actual:   ", end="")
    for k, v in results3.items():
        print(f"{k}={v:.3f}, ", end="")
    
    # Check if results match expected values (within tolerance)
    tolerance = 1e-6
    all_match = all(abs(results3[k] - expected3[k]) < tolerance for k in expected3.keys())
    print(f"\n{'✓ PASS' if all_match else '✗ FAIL'}\n")
    
    # Test Case 4: With Ignore Index
    print("Test Case 4: With Ignore Index")
    pred4 = np.array([[0, 1, 0], 
                      [1, 0, 1]])
    gt4 = np.array([[0, 1, 255], 
                    [1, 0, 255]])  # Last column should be ignored
    
    # Manual calculation (ignoring index 255):
    # Valid pixels: 4 (ignore last column)
    # Correct: all 4 are correct
    # aAcc = 4/4 = 1.0, mIoU = 1.0
    
    evaluator4 = create_evaluator(num_classes=2, ignore_index=255, metrics=['mIoU'])
    evaluator4.add_batch(pred4, gt4)
    results4 = evaluator4.compute()
    
    print(f"Prediction:\n{pred4}")
    print(f"Ground Truth:\n{gt4} (255 = ignore)")
    print(f"Expected: aAcc=1.000, mAcc=1.000, mIoU=1.000")
    print(f"Actual:   ", end="")
    for k, v in results4.items():
        print(f"{k}={v:.3f}, ", end="")
    
    expected_perfect = all(abs(v - 1.0) < 1e-6 for v in results4.values())
    print(f"\n{'✓ PASS' if expected_perfect else '✗ FAIL'}\n")
    
    # Test Case 5: 3-Class Problem
    print("Test Case 5: 3-Class Problem") 
    pred5 = np.array([[0, 1, 2], 
                      [0, 1, 2]])
    gt5 = np.array([[0, 1, 2], 
                    [1, 2, 0]])
    
    # Manual calculation:
    # Total pixels: 6
    # Correct: (0,0)=✓, (0,1)=✓, (0,2)=✓, (1,0)=✗, (1,1)=✗, (1,2)=✗ -> 3 correct
    # aAcc = 3/6 = 0.5
    
    evaluator5 = create_evaluator(num_classes=3, metrics=['mIoU'])
    evaluator5.add_batch(pred5, gt5)
    results5 = evaluator5.compute()
    per_class5 = evaluator5.get_per_class_results()
    
    print(f"Prediction:\n{pred5}")
    print(f"Ground Truth:\n{gt5}")
    print(f"Results: ", end="")
    for k, v in results5.items():
        print(f"{k}={v:.3f}, ", end="")
    print(f"\nPer-class IoU: {per_class5.get('per_class_iou', 'N/A')}")
    print("✓ 3-class test completed\n")
    
    print("=== All Tests Completed ===")


if __name__ == "__main__":
    test_metrics()
