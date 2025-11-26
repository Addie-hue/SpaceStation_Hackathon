import argparse
import json
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class OptimizedSpaceStationPredictor:
    def __init__(self, model_path, test_path=None, labels_path=None, output_dir="optimized_predictions"):
        self.model = self.load_model(model_path)
        self.test_path, self.labels_path = self.auto_detect_paths(test_path, labels_path)
        self.output_dir = Path(output_dir)
        self.setup_directories()
        self.class_names = self.load_class_names()
        
        print(f"✅ Loaded {len(self.class_names)} classes: {self.class_names}")
        
    def load_model(self, model_path):
        """Load model with auto-detection"""
        model_candidates = [
            model_path,
            'runs/detect/train/weights/best.pt',
            'runs/detect/train2/weights/best.pt', 
            'best.pt',
            'weights/best.pt'
        ]
        
        for candidate in model_candidates:
            if Path(candidate).exists():
                print(f"📁 Loading model from: {candidate}")
                return YOLO(candidate)
        
        raise FileNotFoundError(f"❌ Model not found. Tried: {model_candidates}")
    
    def auto_detect_paths(self, test_path, labels_path):
        """Auto-detect test images and labels paths"""
        test_candidates = [
            test_path,
            '../dataset/test/images',
            'dataset/test/images', 
            'test/images',
            '../test/images'
        ]
        
        # Find test path
        found_test = None
        for candidate in test_candidates:
            if candidate and Path(candidate).exists():
                found_test = Path(candidate)
                break
        
        if not found_test:
            for pattern in ['**/test/images', '**/images', '**/test']:
                matches = list(Path('.').glob(pattern))
                if matches:
                    found_test = matches[0]
                    break
        
        if not found_test:
            raise FileNotFoundError("❌ Could not find test images directory")
        
        # Find labels path
        found_labels = None
        possible_labels = [
            labels_path,
            found_test.parent / "labels",
            found_test.parent.parent / "test" / "labels", 
            found_test.parent.parent / "labels" / "test",
            Path('../dataset/test/labels'),
            Path('dataset/test/labels')
        ]
        
        for candidate in possible_labels:
            if candidate and Path(candidate).exists():
                found_labels = Path(candidate)
                break
        
        print(f"📁 Test images: {found_test}")
        print(f"📁 Test labels: {found_labels}")
        return found_test, found_labels
    
    def load_class_names(self):
        """Load class names from YAML or use defaults"""
        yaml_candidates = ['yolo_params.yaml', '../yolo_params.yaml', 'dataset.yaml', '../dataset.yaml']
        
        for yaml_file in yaml_candidates:
            if Path(yaml_file).exists():
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if 'names' in data:
                            return data['names']
                except Exception as e:
                    print(f"⚠️  Error loading {yaml_file}: {e}")
        
        # Fallback to documented class names
        return ['OxygenTank', 'FirstAidBox', 'FireAidBox', 'SafetySwitchPanel',
                'FireExtinguisher', 'FireAlarm', 'NitrogenTank', 'EmergencyPhone']
        
    def setup_directories(self):
        """Create organized output directory structure"""
        dirs = [
            'images', 'labels', 'metrics/plots', 'metrics/tables',
            'failures/small_objects', 'failures/occluded', 'failures/low_confidence',
            'failures/misclassified', 'failures/missed_detections', 'failures/false_positives',
            'confidence_analysis', 'class_analysis'
        ]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def load_ground_truth(self, image_path):
        """Load ground truth labels for an image"""
        if self.labels_path is None:
            return []
            
        label_path = self.labels_path / image_path.with_suffix('.txt').name
        ground_truth = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        ground_truth.append({
                            'class_id': cls_id,
                            'class_name': self.class_names[cls_id],
                            'x_center': x_center, 'y_center': y_center,
                            'width': width, 'height': height,
                            'area': width * height
                        })
        return ground_truth
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        def center_to_corners(box):
            x1 = box['x_center'] - box['width'] / 2
            y1 = box['y_center'] - box['height'] / 2
            x2 = box['x_center'] + box['width'] / 2
            y2 = box['y_center'] + box['height'] / 2
            return [x1, y1, x2, y2]
        
        box1_corners = center_to_corners(box1)
        box2_corners = center_to_corners(box2)
        
        x1 = max(box1_corners[0], box2_corners[0])
        y1 = max(box1_corners[1], box2_corners[1])
        x2 = min(box1_corners[2], box2_corners[2])
        y2 = min(box1_corners[3], box2_corners[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def match_predictions_to_ground_truth(self, predictions, ground_truth, iou_threshold=0.5):
        """Match predictions to ground truth objects"""
        gt_matched = [False] * len(ground_truth)
        pred_matched = [False] * len(predictions)
        matches = []
        
        sorted_pred_indices = sorted(range(len(predictions)), 
                                   key=lambda i: predictions[i]['confidence'], reverse=True)
        
        for pred_idx in sorted_pred_indices:
            pred = predictions[pred_idx]
            best_iou = iou_threshold
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truth):
                if gt_matched[gt_idx]:
                    continue
                iou = self.calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                gt_matched[best_gt_idx] = True
                pred_matched[pred_idx] = True
                matches.append({
                    'pred_idx': pred_idx, 'gt_idx': best_gt_idx,
                    'iou': best_iou, 'correct_class': predictions[pred_idx]['class_id'] == ground_truth[best_gt_idx]['class_id']
                })
        
        return matches, gt_matched, pred_matched
    
    def find_optimal_confidence(self):
        """Automatically find the optimal confidence threshold"""
        print("🔍 Finding optimal confidence threshold...")
        
        confidence_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
        results = []
        
        # Use a subset of images for faster optimization
        image_paths = list(self.test_path.glob("*.*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        sample_size = min(100, len(image_paths))
        sample_images = image_paths[:sample_size]
        
        print(f"   Testing {len(confidence_levels)} confidence levels on {sample_size} images...")
        
        for conf in confidence_levels:
            metrics = self.evaluate_confidence_level(sample_images, conf)
            metrics['confidence'] = conf
            results.append(metrics)
            print(f"   conf={conf}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
        
        # Find optimal confidence (maximize F1-score)
        optimal = max(results, key=lambda x: x['f1_score'])
        
        print(f"🎯 Optimal confidence: {optimal['confidence']} (F1-score: {optimal['f1_score']:.3f})")
        
        # Plot confidence sweep
        self.plot_confidence_sweep(results)
        
        return optimal['confidence']
    
    def evaluate_confidence_level(self, image_paths, conf_threshold):
        """Evaluate performance at specific confidence level"""
        total_tp, total_fp, total_fn = 0, 0, 0
        total_gt = 0
        
        for img_path in image_paths:
            ground_truth = self.load_ground_truth(img_path)
            total_gt += len(ground_truth)
            
            results = self.model.predict(img_path, conf=conf_threshold, save=False)
            result = results[0]
            
            predictions = []
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                predictions.append({
                    'class_id': cls_id, 'class_name': self.class_names[cls_id],
                    'confidence': confidence, 'x_center': x_center, 'y_center': y_center,
                    'width': width, 'height': height
                })
            
            matches, gt_matched, pred_matched = self.match_predictions_to_ground_truth(
                predictions, ground_truth, iou_threshold=0.5
            )
            
            total_tp += len([m for m in matches if m['correct_class']])
            total_fp += len([m for m in matches if not m['correct_class']])
            total_fp += sum([1 for m in pred_matched if not m])
            total_fn += sum([1 for m in gt_matched if not m])
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision, 'recall': recall, 'f1_score': f1_score,
            'true_positives': total_tp, 'false_positives': total_fp, 'false_negatives': total_fn
        }
    
    def plot_confidence_sweep(self, results):
        """Plot confidence threshold optimization results"""
        confidences = [r['confidence'] for r in results]
        precisions = [r['precision'] for r in results]
        recalls = [r['recall'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(confidences, precisions, 'o-', label='Precision', linewidth=2)
        plt.plot(confidences, recalls, 'o-', label='Recall', linewidth=2)
        plt.plot(confidences, f1_scores, 'o-', label='F1-Score', linewidth=3, color='red')
        
        # Mark optimal point
        optimal_idx = np.argmax(f1_scores)
        plt.axvline(x=confidences[optimal_idx], color='red', linestyle='--', alpha=0.7)
        plt.text(confidences[optimal_idx], max(f1_scores), f'Optimal: {confidences[optimal_idx]}', 
                ha='center', va='bottom', color='red', fontweight='bold')
        
        plt.title('Confidence Threshold Optimization')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis' / 'confidence_optimization.png', dpi=300)
        plt.close()
    
    def analyze_missed_detections(self, image_paths, conf_threshold):
        """Detailed analysis of missed detections"""
        print("🔍 Analyzing missed detection patterns...")
        
        small_object_threshold = 0.01  # 1% of image area
        class_missed = Counter()
        size_analysis = []
        occlusion_analysis = []  # Simplified: objects very close to each other
        
        for img_path in image_paths:
            ground_truth = self.load_ground_truth(img_path)
            results = self.model.predict(img_path, conf=conf_threshold, save=False)
            result = results[0]
            
            predictions = []
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                predictions.append({
                    'class_id': cls_id, 'confidence': confidence,
                    'x_center': x_center, 'y_center': y_center,
                    'width': width, 'height': height
                })
            
            matches, gt_matched, pred_matched = self.match_predictions_to_ground_truth(
                predictions, ground_truth, iou_threshold=0.5
            )
            
            for i, (gt, matched) in enumerate(zip(ground_truth, gt_matched)):
                if not matched:
                    class_missed[gt['class_name']] += 1
                    
                    # Size analysis
                    if gt['area'] < small_object_threshold:
                        size_analysis.append({'class': gt['class_name'], 'area': gt['area'], 'type': 'small'})
                    else:
                        size_analysis.append({'class': gt['class_name'], 'area': gt['area'], 'type': 'normal'})
        
        # Generate missed detection report
        self.generate_missed_detection_report(class_missed, size_analysis)
        
        return class_missed, size_analysis
    
    def generate_missed_detection_report(self, class_missed, size_analysis):
        """Generate detailed missed detection analysis"""
        # Class-wise missed detection plot
        plt.figure(figsize=(12, 6))
        classes, counts = zip(*class_missed.most_common())
        plt.bar(classes, counts, color='lightcoral', alpha=0.7, edgecolor='black')
        plt.title('Missed Detections by Class')
        plt.xlabel('Class Name')
        plt.ylabel('Number of Missed Detections')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_analysis' / 'missed_detections_by_class.png', dpi=300)
        plt.close()
        
        # Size analysis
        small_count = len([s for s in size_analysis if s['type'] == 'small'])
        normal_count = len(size_analysis) - small_count
        
        plt.figure(figsize=(8, 6))
        sizes = ['Small Objects', 'Normal Objects']
        counts = [small_count, normal_count]
        plt.bar(sizes, counts, color=['red', 'blue'], alpha=0.7, edgecolor='black')
        plt.title('Missed Detections by Object Size')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_analysis' / 'missed_detections_by_size.png', dpi=300)
        plt.close()
        
        # Save detailed report
        report = {
            'total_missed': sum(class_missed.values()),
            'class_wise_missed': dict(class_missed),
            'size_analysis': {
                'small_objects_missed': small_count,
                'normal_objects_missed': normal_count,
                'small_object_percentage': (small_count / len(size_analysis) * 100) if size_analysis else 0
            }
        }
        
        with open(self.output_dir / 'metrics' / 'missed_detection_analysis.json', 'w') as f:
            json.dump(report, f, indent=2)
    
    def comprehensive_predict_and_analyze(self, conf_threshold=None):
        """Main optimized prediction and analysis pipeline"""
        print("🚀 SPACE STATION SAFETY DETECTION - OPTIMIZED ANALYSIS")
        
        # Find optimal confidence if not provided
        if conf_threshold is None:
            conf_threshold = self.find_optimal_confidence()
        else:
            print(f"🎯 Using provided confidence: {conf_threshold}")
        
        # Get all test images
        image_paths = list(self.test_path.glob("*.*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        print(f"📊 Processing {len(image_paths)} test images with conf={conf_threshold}...")
        
        all_predictions = []
        all_failures = []
        performance_metrics = {
            'total_images': len(image_paths),
            'total_predictions': 0,
            'total_ground_truth': 0,
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'misclassifications': 0,
            'missed_detections': 0,
            'low_confidence': 0
        }
        
        confusion_data = []
        
        for i, img_path in enumerate(image_paths):
            if i % 50 == 0:
                print(f"   Progress: {i+1}/{len(image_paths)}")
            
            ground_truth = self.load_ground_truth(img_path)
            performance_metrics['total_ground_truth'] += len(ground_truth)
            
            try:
                results = self.model.predict(img_path, conf=conf_threshold, save=False)
                result = results[0]
            except Exception as e:
                continue
            
            # Extract predictions
            predictions = []
            for box in result.boxes:
                cls_id = int(box.cls)
                confidence = float(box.conf)
                x_center, y_center, width, height = box.xywhn[0].tolist()
                
                predictions.append({
                    'image': img_path.name,
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id],
                    'confidence': confidence,
                    'x_center': x_center, 'y_center': y_center,
                    'width': width, 'height': height
                })
            
            performance_metrics['total_predictions'] += len(predictions)
            all_predictions.extend(predictions)
            
            # Analyze failures
            failures = self.analyze_failures(img_path, predictions, ground_truth, result, conf_threshold)
            all_failures.extend(failures)
            
            # Update metrics
            if ground_truth:
                matches, gt_matched, pred_matched = self.match_predictions_to_ground_truth(
                    predictions, ground_truth, iou_threshold=0.5
                )
                
                performance_metrics['true_positives'] += len([m for m in matches if m['correct_class']])
                performance_metrics['false_positives'] += len([m for m in matches if not m['correct_class']])
                performance_metrics['false_positives'] += sum([1 for m in pred_matched if not m])
                performance_metrics['false_negatives'] += sum([1 for m in gt_matched if not m])
                
                for match in matches:
                    gt = ground_truth[match['gt_idx']]
                    pred = predictions[match['pred_idx']]
                    confusion_data.append((gt['class_id'], pred['class_id']))
            
            for failure_type, gt, pred in failures:
                if failure_type in performance_metrics:
                    performance_metrics[failure_type] += 1
            
            # Save outputs
            self.save_visual_prediction(img_path, result)
            self.save_yolo_labels(img_path, predictions)
        
        # Generate comprehensive reports
        self.generate_performance_report(all_predictions, all_failures, performance_metrics, confusion_data, conf_threshold)
        self.generate_confusion_matrix_analysis(confusion_data, all_failures)
        self.analyze_missed_detections(image_paths, conf_threshold)
        
        print(f"✅ Analysis complete! Results saved to {self.output_dir}")
        
        return performance_metrics, conf_threshold
    
    def analyze_failures(self, image_path, predictions, ground_truth, result, conf_threshold):
        """Comprehensive failure analysis"""
        failures = []
        
        if not ground_truth:
            for pred in predictions:
                if pred['confidence'] < conf_threshold + 0.1:  # Low confidence relative to threshold
                    failures.append(('low_confidence', None, pred))
            return failures
        
        matches, gt_matched, pred_matched = self.match_predictions_to_ground_truth(
            predictions, ground_truth, iou_threshold=0.5
        )
        
        # Misclassifications
        for match in matches:
            pred = predictions[match['pred_idx']]
            gt = ground_truth[match['gt_idx']]
            if not match['correct_class']:
                failures.append(('misclassified', gt, pred))
        
        # Missed detections
        for i, (gt, matched) in enumerate(zip(ground_truth, gt_matched)):
            if not matched:
                failures.append(('missed_detections', gt, None))
        
        # False positives
        for i, (pred, matched) in enumerate(zip(predictions, pred_matched)):
            if not matched:
                failures.append(('false_positives', None, pred))
        
        return failures
    
    def save_visual_prediction(self, img_path, result):
        """Save image with predictions"""
        output_path = self.output_dir / 'images' / img_path.name
        plotted_image = result.plot()
        cv2.imwrite(str(output_path), plotted_image)
    
    def save_yolo_labels(self, img_path, predictions):
        """Save predictions in YOLO format"""
        label_path = self.output_dir / 'labels' / img_path.with_suffix('.txt').name
        with open(label_path, 'w') as f:
            for pred in predictions:
                f.write(f"{pred['class_id']} {pred['x_center']} {pred['y_center']} {pred['width']} {pred['height']}\n")
    
    def generate_performance_report(self, all_predictions, all_failures, metrics, confusion_data, conf_threshold):
        """Generate comprehensive performance report"""
        report = {
            'optimal_confidence': conf_threshold,
            'performance_metrics': metrics,
            'failure_breakdown': dict(Counter([f[0] for f in all_failures])),
            'class_distribution': dict(Counter([p['class_name'] for p in all_predictions])),
            'confidence_statistics': {
                'mean': np.mean([p['confidence'] for p in all_predictions]) if all_predictions else 0,
                'std': np.std([p['confidence'] for p in all_predictions]) if all_predictions else 0,
                'min': min([p['confidence'] for p in all_predictions]) if all_predictions else 0,
                'max': max([p['confidence'] for p in all_predictions]) if all_predictions else 0,
            }
        }
        
        # Calculate metrics
        if metrics['total_ground_truth'] > 0:
            precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
            recall = metrics['true_positives'] / metrics['total_ground_truth'] if metrics['total_ground_truth'] > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            report['detection_metrics'] = {
                'precision': precision, 'recall': recall, 'f1_score': f1_score,
                'false_positive_rate': metrics['false_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0,
            }
        
        # Save reports
        with open(self.output_dir / 'metrics' / 'comprehensive_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.create_readable_summary(report, metrics, conf_threshold)
        self.generate_performance_visualizations(report, all_predictions)
    
    def create_readable_summary(self, report, metrics, conf_threshold):
        """Create human-readable summary"""
        summary_path = self.output_dir / 'metrics' / 'performance_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=== OPTIMIZED SPACE STATION SAFETY DETECTION ANALYSIS ===\n\n")
            f.write(f"Optimal Confidence Threshold: {conf_threshold}\n")
            f.write(f"Test Images Processed: {metrics['total_images']}\n")
            f.write(f"Total Ground Truth Objects: {metrics['total_ground_truth']}\n")
            f.write(f"Total Predictions Made: {metrics['total_predictions']}\n\n")
            
            if 'detection_metrics' in report:
                dm = report['detection_metrics']
                f.write("DETECTION METRICS:\n")
                f.write(f"  Precision: {dm['precision']:.3f} ({(dm['precision']*100):.1f}%)\n")
                f.write(f"  Recall: {dm['recall']:.3f} ({(dm['recall']*100):.1f}%)\n")
                f.write(f"  F1-Score: {dm['f1_score']:.3f}\n")
                f.write(f"  True Positives: {metrics['true_positives']}\n")
                f.write(f"  False Positives: {metrics['false_positives']}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
            
            f.write("FAILURE ANALYSIS:\n")
            for failure_type, count in report['failure_breakdown'].items():
                f.write(f"  {failure_type}: {count} cases\n")
            
            f.write(f"\nCONFIDENCE STATISTICS:\n")
            cs = report['confidence_statistics']
            f.write(f"  Mean: {cs['mean']:.3f}\n")
            f.write(f"  Std: {cs['std']:.3f}\n")
            f.write(f"  Range: {cs['min']:.3f} - {cs['max']:.3f}\n")
    
    def generate_performance_visualizations(self, report, all_predictions):
        """Generate performance visualizations"""
        # Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences = [p['confidence'] for p in all_predictions]
        if confidences:
            plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Prediction Confidence Distribution')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / 'metrics' / 'plots' / 'confidence_distribution.png', dpi=300)
            plt.close()
    
    def generate_confusion_matrix_analysis(self, confusion_data, all_failures):
        """Generate confusion matrix analysis"""
        if not confusion_data:
            return
        
        true_classes = [item[0] for item in confusion_data]
        pred_classes = [item[1] for item in confusion_data]
        
        num_classes = len(self.class_names)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        
        for true_cls, pred_cls in confusion_data:
            cm[true_cls][pred_cls] += 1
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, square=True)
        plt.title('Confusion Matrix - True vs Predicted')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics' / 'plots' / 'confusion_matrix.png', dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Optimized Space Station Safety Detection Analysis')
    parser.add_argument('--model', type=str, default='runs/detect/train/weights/best.pt', 
                       help='Path to trained model weights')
    parser.add_argument('--test', type=str, default=None,
                       help='Path to test images directory (auto-detected if not provided)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to test labels directory (auto-detected if not provided)')
    parser.add_argument('--output', type=str, default='optimized_predictions',
                       help='Output directory for predictions')
    parser.add_argument('--conf', type=float, default=None,
                       help='Confidence threshold (auto-optimized if not provided)')
    
    args = parser.parse_args()
    
    try:
        predictor = OptimizedSpaceStationPredictor(args.model, args.test, args.labels, args.output)
        metrics, optimal_conf = predictor.comprehensive_predict_and_analyze(conf_threshold=args.conf)
        
        # Final summary
        print("\n" + "="*60)
        print("🎯 OPTIMIZED PERFORMANCE SUMMARY")
        print("="*60)
        print(f"📊 Optimal Confidence: {optimal_conf}")
        if 'true_positives' in metrics:
            precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives']) if (metrics['true_positives'] + metrics['false_positives']) > 0 else 0
            recall = metrics['true_positives'] / metrics['total_ground_truth'] if metrics['total_ground_truth'] > 0 else 0
            print(f"📈 Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0:.3f}")
            print(f"✅ True Positives: {metrics['true_positives']} | ❌ False Positives: {metrics['false_positives']} | ❌ False Negatives: {metrics['false_negatives']}")
        
        print(f"📁 Comprehensive analysis saved to: {args.output}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Check dataset structure: dataset/test/images/ and dataset/test/labels/")
        print("2. Ensure model file exists")
        print("3. Try: python predict.py --model best.pt")

if __name__ == '__main__':
    main()