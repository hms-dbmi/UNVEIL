import argparse

from experiment_runner import ExperimentRunner, InferenceRunner


def main():
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(description="Process configs and save paths.")

    # Add the configs_path argument
    parser.add_argument(
        '--configs_path',
        type=str,
        help='Path to the configuration file'
    )

    # Add the save_path argument
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default=None, 
        help='Directory path where the output should be saved (overwrites the configs)',
    )

    # Add mode argument argument
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=["train", "infer"],
        default="train", 
        help='Mode of operation. Can be either "train" or "infer".',
    )

    # Parse the command line arguments
    args = parser.parse_args()

    # Run inference or experiment train
    if args.mode == 'train':
        runner = ExperimentRunner(configs_path=args.configs_path, save_dir=args.save_dir)
    else:
        runner = InferenceRunner(configs_path=args.configs_path, save_dir=args.save_dir)

    # run experiment
    runner.run()


if __name__ == "__main__":
    main()
