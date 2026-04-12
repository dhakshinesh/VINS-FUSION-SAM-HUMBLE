#!/usr/bin/env python3
"""
ROS2 Service for Segment Anything Model (SAM)
This service receives images and returns segmentation masks
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vins.srv import SAMSegmentation
import sys
import os

# Add segment-anything to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../segment-anything'))

from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

class SAMService(Node):
    def __init__(self):
        super().__init__('sam_service')
        
        # Get parameters
        self.declare_parameter('sam_model_type', 'vit_t')
        self.declare_parameter('sam_checkpoint_path', '/home/dhakshinesh/segment-anything/checkpoints/mobile_sam.pt')
        self.declare_parameter('use_automatic_mask', True)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model_type = self.get_parameter('sam_model_type').value
        self.checkpoint_path = self.get_parameter('sam_checkpoint_path').value
        self.use_automatic_mask = self.get_parameter('use_automatic_mask').value
        self.device = self.get_parameter('device').value
        
        if not self.checkpoint_path:
            self.get_logger().error("SAM checkpoint path not provided! Please set sam_checkpoint_path parameter")
            sys.exit(1)
            return
        
        # Initialize SAM model
        self.get_logger().info(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")
        self.get_logger().info(f"Using device: {self.device}")
        
        try:
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            
            if self.use_automatic_mask:
                # Use automatic mask generator for full image segmentation
                self.mask_generator = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100,
                )
                self.predictor = None
            else:
                # Use predictor for prompt-based segmentation
                self.predictor = SamPredictor(sam)
                self.mask_generator = None
            
            self.get_logger().info("SAM model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load SAM model: {str(e)}")
            sys.exit(1)
            return
        
        self.bridge = CvBridge()
        
        # Create service
        self.srv = self.create_service(SAMSegmentation, 'sam_segmentation', self.handle_segmentation)
        self.get_logger().info("SAM service ready")
    
    def handle_segmentation(self, request, response):
        """
        Handle segmentation request
        """
        try:
            # Convert ROS image to OpenCV
            try:
                # cv_bridge handles ROS2 sensor_msgs
                gray = self.bridge.imgmsg_to_cv2(request.image, "mono8")
                cv_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                self.get_logger().error(f"Image conversion failed: {e}")
                response.success = False
                return response
            
            # Convert BGR to RGB for SAM
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            if self.use_automatic_mask:
                # Generate automatic masks
                masks = self.mask_generator.generate(rgb_image)
                
                # Combine all masks into a single binary mask
                combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
                
                for mask_data in masks:
                    if isinstance(mask_data['segmentation'], np.ndarray):
                        mask = mask_data['segmentation'].astype(np.uint8) * 255
                    else:
                        # Handle RLE format
                        from pycocotools import mask as mask_utils
                        mask = mask_utils.decode(mask_data['segmentation']).astype(np.uint8) * 255
                    combined_mask = np.maximum(combined_mask, mask)
                
            else:
                # Use predictor with points (if provided)
                if len(request.points) > 0:
                    self.predictor.set_image(rgb_image)
                    points = np.array([[p.x, p.y] for p in request.points])
                    labels = np.ones(len(points))  # All foreground points
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=points,
                        point_labels=labels,
                        multimask_output=False
                    )
                    combined_mask = masks[0].astype(np.uint8) * 255
                else:
                    # No points provided, return empty mask
                    combined_mask = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8)
            
            # Convert mask to ROS image message
            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
            
            response.mask = mask_msg
            response.success = True
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Error in segmentation: {str(e)}")
            response.success = False
            return response

def main(args=None):
    rclpy.init(args=args)
    
    try:
        service = SAMService()
        rclpy.spin(service)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
