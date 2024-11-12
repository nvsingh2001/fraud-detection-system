"""
Main Execution Script for Fraud Detection System
----------------------------------------------
This script orchestrates the complete fraud detection pipeline by combining
the FraudDetectionSystem and FraudVisualization modules. It handles data processing,
model training, evaluation, and visualization generation.

Usage:
    python fraud_detection_main.py --data-path /path/to/data.csv --output-dir /path/to/output
"""

import os
import argparse
import logging
from datetime import datetime
import json
import yaml

# Import our custom modules
from fraud_detection_system import FraudDetectionSystem
from fraud_detection_viz import FraudVisualization, create_visualization_report

class FraudDetectionPipeline:
    """
    Orchestrates the complete fraud detection workflow.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else self._get_default_config()
        self._setup_logging()
        
    @staticmethod
    def _get_default_config():
        """Provide default configuration if no config file is specified."""
        return {
            'features': ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long'],
            'chunk_size': 10000,
            'output_dir': 'fraud_detection_results',
            'model_filename': 'fraud_detection_model.pkl',
            'log_level': 'INFO'
        }
    
    @staticmethod
    def _load_config(config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'fraud_detection_{timestamp}.log')
        
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FraudDetectionPipeline')
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        dirs = [
            self.config['output_dir'],
            os.path.join(self.config['output_dir'], 'models'),
            os.path.join(self.config['output_dir'], 'visualizations'),
            os.path.join(self.config['output_dir'], 'metrics')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _save_metrics(self, metrics):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(
            self.config['output_dir'],
            'metrics',
            'metrics.json'
        )
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for key, value in metrics.items():
            if hasattr(value, 'tolist'):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        
        with open(metrics_file, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
    
    def run(self, data_path):
        """
        Execute the complete fraud detection pipeline.
        
        Args:
            data_path (str): Path to the input data file
        """
        try:
            self.logger.info("Starting fraud detection pipeline")
            # self._create_output_directories()
            
            # Initialize the fraud detection system
            self.logger.info("Initializing fraud detection system")
            fraud_detector = FraudDetectionSystem(
                features=self.config['features'],
                chunk_size=self.config['chunk_size']
            )

            # Load the trained model
            self.logger.info("Loading Pretrained Model")
            model_path = 'fraud_detection_results/models/fraud_detection_model.pkl'
            fraud_detector.load_model(model_path)
            
            # Process data
            self.logger.info("Processing data")
            X_all, y_all = fraud_detector.process_data(data_path)
            self.logger.info(f"Processed data shape: {X_all.shape}")
            
            # Handle class imbalance
            self.logger.info("Handling class imbalance")
            X_resampled, y_resampled = fraud_detector.handle_imbalance(X_all, y_all)
            self.logger.info(f"Resampled data shape: {X_resampled.shape}")
            
            # # Train model
            # self.logger.info("Training model")
            # fraud_detector.train(X_resampled, y_resampled)
            
            # # Save model
            # model_path = os.path.join(
            #     self.config['output_dir'],
            #     'models',
            #     self.config['model_filename']
            # )
            # fraud_detector.save_model(model_path)
            # self.logger.info(f"Model saved to {model_path}")
            
            # Evaluate model
            self.logger.info("Evaluating model")
            metrics = fraud_detector.evaluate(X_all, y_all)
            self._save_metrics(metrics)

            

            # Generate visualizations
            self.logger.info("Generating visualizations")
            viz_dir = os.path.join(self.config['output_dir'], 'visualizations')
            create_visualization_report(
                fraud_detector,
                metrics,
                output_dir=viz_dir
            )
            self.logger.info("Pipeline completed successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fraud Detection Pipeline'
    )
    parser.add_argument(
        '--data-path',
        required=True,
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for results'
    )
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize and run pipeline
    pipeline = FraudDetectionPipeline(args.config)
    
    # Override output directory if specified in command line
    if args.output_dir:
        pipeline.config['output_dir'] = args.output_dir
    
    # Run the pipeline
    metrics = pipeline.run(args.data_path)
    
    # Print summary metrics
    print("\nFraud Detection Results Summary:")
    print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")

if __name__ == "__main__":
    main()