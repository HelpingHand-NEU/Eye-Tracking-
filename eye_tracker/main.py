#!/usr/bin/env python3
import os
from .eye_tracking_training import EyeTrackingTraining, EyeTrackingTrainingConfig


def main():
    default_image = "/Users/peterwang/Library/CloudStorage/OneDrive-Personal/文档/Capstone/Table3.jpg"
    image_path = default_image if os.path.exists(default_image) else None
    cfg = EyeTrackingTrainingConfig(
        screen_size=None,
        image_path=image_path,
        training_data_name="peter",
    )
    trainer = EyeTrackingTraining(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
