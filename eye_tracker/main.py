#!/usr/bin/env python3
from .eye_tracking_training import EyeTrackingTraining, EyeTrackingTrainingConfig


def main():
    cfg = EyeTrackingTrainingConfig(
        screen_size=None,
        image_path=None,
        training_data_name="peter",
    )
    trainer = EyeTrackingTraining(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
