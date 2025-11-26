import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from collections import Counter
import pandas as pd

class AdvancedYoloVisualizer:
    def __init__(self, dataset_folder, output_dir="visualization_results"):
        self.dataset_folder = Path(dataset_folder)
        self.output_dir = Path(output_dir)
        self.setup_output_dirs()
        
        # Load class names and colors
        with open('yolo_params.yaml', 'r') as f:
            data = yaml.safe_load(f)
            self.class_names = data['names']
        
        self.class_colors = self.generate_colors(len(self.class_names))
        self.set_mode('train')  # Default to training set
        
    def setup_output_dirs(self):
        """Create organized output directories"""
        dirs = [
            'analysis/class_distribution',
            'analysis/bounding_box_stats',
            'samples/train',
            'samples/val', 
            'samples/test',
            'comparisons'
        ]
        for dir_name in dirs:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def generate_colors(self, num_classes):
        """Generate distinct colors for each class"""
        np.random.seed(42)  # Consistent colors
        colors = []
        for i in range(num_classes):
            # Generate distinct colors
            hue = i / num_classes
            saturation = 0.8 + np.random.random() * 0.2
            value = 0.8 + np.random.random() * 0.2
            
            # Convert HSV to BGR
            hsv = np.uint8([[[hue * 179, saturation * 255, value * 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(map(int, bgr[0, 0])))
        
        return colors
    
    def set_mode(self, mode='train'):
        """Set dataset mode (train/val/test)"""
        self.mode = mode
        if mode == 'train':
            self.images_folder = self.dataset_folder / "train" / "images"
            self.labels_folder = self.dataset_folder / "train" / "labels"
        elif mode == 'val':
            self.images_folder = self.dataset_folder / "val" / "images" 
            self.labels_folder = self.dataset_folder / "val" / "labels"
        else:  # test
            self.images_folder = self.dataset_folder / "test" / "images"
            self.labels_folder = self.dataset_folder / "test" / "labels"
        
        # Get sorted lists of files
        self.image_files = sorted(self.images_folder.glob("*.*"))
        self.image_files = [f for f in self.image_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        self.label_files = sorted(self.labels_folder.glob("*.txt"))
        
        self.num_images = len(self.image_files)
        self.current_index = 0
        
        print(f"📁 Mode: {mode.upper()}")
        print(f"📊 Images: {self.num_images}")
        print(f"🏷️  Labels: {len(self.label_files)}")
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🔬 Running comprehensive dataset analysis...")
        
        all_annotations = []
        bbox_stats = []
        
        # Collect data from all modes
        for mode in ['train', 'val', 'test']:
            self.set_mode(mode)
            
            for img_file, label_file in zip(self.image_files, self.label_files):
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.read().splitlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            all_annotations.append({
                                'mode': mode,
                                'class_id': class_id,
                                'class_name': self.class_names[class_id],
                                'width': width,
                                'height': height,
                                'area': width * height,
                                'image': img_file.name
                            })
                            
                            bbox_stats.append({
                                'class_name': self.class_names[class_id],
                                'width': width,
                                'height': height,
                                'area': width * height,
                                'aspect_ratio': width / height if height > 0 else 0
                            })
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(all_annotations)
        bbox_df = pd.DataFrame(bbox_stats)
        
        if len(df) > 0:
            self.generate_analysis_plots(df, bbox_df)
            self.generate_statistics_report(df, bbox_df)
        
        return df
    
    def generate_analysis_plots(self, df, bbox_df):
        """Generate comprehensive analysis plots"""
        
        # 1. Class Distribution
        plt.figure(figsize=(12, 6))
        class_counts = df['class_name'].value_counts()
        
        plt.subplot(1, 2, 1)
        class_counts.plot(kind='bar', color=[self.class_colors[i] for i in range(len(class_counts))])
        plt.title('Class Distribution Across Dataset')
        plt.xlabel('Class Name')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        mode_class_counts = df.groupby(['mode', 'class_name']).size().unstack(fill_value=0)
        mode_class_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
        plt.title('Class Distribution by Dataset Split')
        plt.xlabel('Dataset Split')
        plt.ylabel('Count')
        plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'analysis' / 'class_distribution' / 'class_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bounding Box Statistics
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(bbox_df['width'], bins=50, alpha=0.7, color='blue')
        plt.title('Bounding Box Width Distribution')
        plt.xlabel('Width (normalized)')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.hist(bbox_df['height'], bins=50, alpha=0.7, color='green')
        plt.title('Bounding Box Height Distribution')
        plt.xlabel('Height (normalized)')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 3)
        plt.hist(bbox_df['area'], bins=50, alpha=0.7, color='red')
        plt.title('Bounding Box Area Distribution')
        plt.xlabel('Area (normalized)')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        plt.hist(bbox_df['aspect_ratio'], bins=50, alpha=0.7, color='purple')
        plt.title('Aspect Ratio Distribution')
        plt.xlabel('Width / Height')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analysis' / 'bounding_box_stats' / 'bbox_statistics.png', dpi=300)
        plt.close()
        
        # 3. Class-wise Bounding Box Sizes
        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(self.class_names):
            class_data = bbox_df[bbox_df['class_name'] == class_name]
            if len(class_data) > 0:
                plt.scatter(class_data['width'], class_data['height'], 
                           color=self.class_colors[i], label=class_name, alpha=0.6)
        
        plt.xlabel('Bounding Box Width')
        plt.ylabel('Bounding Box Height')
        plt.title('Bounding Box Sizes by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'analysis' / 'bounding_box_stats' / 'class_bbox_sizes.png', dpi=300)
        plt.close()
    
    def generate_statistics_report(self, df, bbox_df):
        """Generate detailed statistics report"""
        report = {
            'dataset_overview': {
                'total_annotations': len(df),
                'total_images': len(df['image'].unique()),
                'num_classes': len(self.class_names),
                'annotations_per_image': len(df) / len(df['image'].unique()) if len(df['image'].unique()) > 0 else 0
            },
            'class_statistics': {},
            'bbox_statistics': {
                'mean_width': bbox_df['width'].mean(),
                'mean_height': bbox_df['height'].mean(),
                'mean_area': bbox_df['area'].mean(),
                'mean_aspect_ratio': bbox_df['aspect_ratio'].mean()
            }
        }
        
        # Class-wise statistics
        for class_name in self.class_names:
            class_data = df[df['class_name'] == class_name]
            bbox_class_data = bbox_df[bbox_df['class_name'] == class_name]
            
            report['class_statistics'][class_name] = {
                'count': len(class_data),
                'percentage': (len(class_data) / len(df)) * 100,
                'mean_bbox_area': bbox_class_data['area'].mean() if len(bbox_class_data) > 0 else 0,
                'images_with_class': len(class_data['image'].unique())
            }
        
        # Save JSON report
        import json
        with open(self.output_dir / 'analysis' / 'dataset_statistics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save readable summary
        with open(self.output_dir / 'analysis' / 'dataset_summary.txt', 'w') as f:
            f.write("=== SPACE STATION SAFETY DETECTION - DATASET ANALYSIS ===\n\n")
            f.write("DATASET OVERVIEW:\n")
            f.write(f"Total Annotations: {report['dataset_overview']['total_annotations']}\n")
            f.write(f"Total Images: {report['dataset_overview']['total_images']}\n")
            f.write(f"Classes: {report['dataset_overview']['num_classes']}\n")
            f.write(f"Annotations per Image: {report['dataset_overview']['annotations_per_image']:.2f}\n\n")
            
            f.write("CLASS DISTRIBUTION:\n")
            for class_name, stats in report['class_statistics'].items():
                f.write(f"  {class_name}: {stats['count']} ({stats['percentage']:.1f}%)\n")
            
            f.write(f"\nBOUNDING BOX STATISTICS:\n")
            f.write(f"  Mean Width: {report['bbox_statistics']['mean_width']:.3f}\n")
            f.write(f"  Mean Height: {report['bbox_statistics']['mean_height']:.3f}\n")
            f.write(f"  Mean Area: {report['bbox_statistics']['mean_area']:.3f}\n")
            f.write(f"  Mean Aspect Ratio: {report['bbox_statistics']['mean_aspect_ratio']:.3f}\n")
    
    def visualize_image(self, index):
        """Visualize single image with annotations"""
        if index >= self.num_images:
            print("❌ Index out of range")
            return None
        
        img_file = self.image_files[index]
        label_file = self.labels_folder / img_file.with_suffix('.txt').name
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            print(f"❌ Could not load image: {img_file}")
            return None
        
        original_image = image.copy()
        annotations = []
        
        # Load and draw annotations
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.read().splitlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert normalized coordinates to pixel coordinates
                    img_height, img_width = image.shape[:2]
                    x_center_px = int(x_center * img_width)
                    y_center_px = int(y_center * img_height)
                    width_px = int(width * img_width)
                    height_px = int(height * img_height)
                    
                    # Calculate bounding box corners
                    x1 = x_center_px - width_px // 2
                    y1 = y_center_px - height_px // 2
                    x2 = x_center_px + width_px // 2
                    y2 = y_center_px + height_px // 2
                    
                    # Draw bounding box
                    color = self.class_colors[class_id]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label background
                    label = f"{self.class_names[class_id]}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(image, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    annotations.append({
                        'class_name': self.class_names[class_id],
                        'bbox': [x1, y1, x2, y2],
                        'normalized_bbox': [x_center, y_center, width, height]
                    })
        
        # Add image info
        info_text = f"Image: {img_file.name} | Annotations: {len(annotations)} | Mode: {self.mode}"
        cv2.putText(image, info_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save sample
        sample_path = self.output_dir / 'samples' / self.mode / f"sample_{index:04d}.png"
        cv2.imwrite(str(sample_path), image)
        
        print(f"✅ Saved sample: {sample_path}")
        return image, annotations
    
    def generate_sample_grid(self, num_samples=16):
        """Generate a grid of sample images"""
        samples_per_mode = min(num_samples // 3, self.num_images)
        
        fig, axes = plt.subplots(3, samples_per_mode, figsize=(20, 12))
        if samples_per_mode == 1:
            axes = axes.reshape(3, 1)
        
        modes = ['train', 'val', 'test']
        
        for row, mode in enumerate(modes):
            self.set_mode(mode)
            indices = np.linspace(0, self.num_images - 1, samples_per_mode, dtype=int)
            
            for col, idx in enumerate(indices):
                image, annotations = self.visualize_image(idx)
                if image is not None:
                    # Convert BGR to RGB for matplotlib
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    axes[row, col].imshow(image_rgb)
                    axes[row, col].set_title(f"{mode}\n{len(annotations)} annos")
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'samples' / 'dataset_samples_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Generated sample grid")
    
    def interactive_browser(self):
        """Interactive image browser"""
        print("\n🎮 INTERACTIVE MODE CONTROLS:")
        print("  A - Previous image")
        print("  D - Next image") 
        print("  T - Switch to Train set")
        print("  V - Switch to Validation set")
        print("  S - Switch to Test set")
        print("  C - Save current image")
        print("  Q/ESC - Quit")
        print("  Number + Enter - Jump to specific image")
        
        cv2.namedWindow("YOLO Dataset Visualizer", cv2.WINDOW_NORMAL)
        
        while True:
            image, annotations = self.visualize_image(self.current_index)
            if image is None:
                self.current_index = (self.current_index + 1) % self.num_images
                continue
            
            # Display image
            cv2.imshow("YOLO Dataset Visualizer", image)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            elif key == ord('a'):  # Previous
                self.current_index = (self.current_index - 1) % self.num_images
            elif key == ord('d'):  # Next
                self.current_index = (self.current_index + 1) % self.num_images
            elif key == ord('t'):  # Train set
                self.set_mode('train')
            elif key == ord('v'):  # Validation set
                self.set_mode('val')
            elif key == ord('s'):  # Test set
                self.set_mode('test')
            elif key == ord('c'):  # Save current
                save_path = self.output_dir / 'comparisons' / f"manual_save_{self.mode}_{self.current_index:04d}.png"
                cv2.imwrite(str(save_path), image)
                print(f"💾 Saved: {save_path}")
            elif 48 <= key <= 57:  # Number key
                # Wait for enter to complete number input
                number_str = chr(key)
                while True:
                    next_key = cv2.waitKey(0) & 0xFF
                    if next_key == 13:  # Enter
                        try:
                            target_index = int(number_str)
                            if 0 <= target_index < self.num_images:
                                self.current_index = target_index
                            break
                        except ValueError:
                            break
                    elif 48 <= next_key <= 57:  # Another number
                        number_str += chr(next_key)
                    else:
                        break
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Advanced YOLO Dataset Visualizer')
    parser.add_argument('--dataset', type=str, default='../dataset', 
                       help='Path to dataset folder')
    parser.add_argument('--output', type=str, default='visualization_results',
                       help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['interactive', 'analysis', 'samples', 'all'],
                       help='Visualization mode')
    parser.add_argument('--samples', type=int, default=16,
                       help='Number of samples for grid view')
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"❌ Dataset folder not found: {args.dataset}")
        return
    
    visualizer = AdvancedYoloVisualizer(args.dataset, args.output)
    
    if args.mode == 'interactive':
        visualizer.interactive_browser()
    elif args.mode == 'analysis':
        visualizer.analyze_dataset()
    elif args.mode == 'samples':
        visualizer.generate_sample_grid(args.samples)
    elif args.mode == 'all':
        visualizer.analyze_dataset()
        visualizer.generate_sample_grid(args.samples)
        print("🎉 All visualizations completed!")

if __name__ == "__main__":
    main()