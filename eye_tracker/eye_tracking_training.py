from .webcam_training import EyeTrackingTraining, EyeTrackingTrainingConfig


def main():
    cfg = EyeTrackingTrainingConfig(
        screen_size=None,
        image_path=None,
        training_data_name="peter",
        fullscreen=True,
        training_mode="webcam",
    )
    trainer = EyeTrackingTraining(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
