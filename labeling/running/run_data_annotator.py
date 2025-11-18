import argparse

from labeling.data_annotator import DataAnnotatorPipeline


parser = argparse.ArgumentParser(description="Label data.")
parser.add_argument("--label", type=str, required=True)
parser.add_argument("--annotating-system-prompt-path", type=str, required=True)
parser.add_argument("--checking-system-prompt-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--output-folder", type=str, required=True)
parser.add_argument("--prompt-round", type=int, required=True)
parser.add_argument("--optimization-flag", type=bool, required=False, default=False)
parser.add_argument("--error-flag", type=bool, required=False, default=False)
parser.add_argument("--error-batch-folder", type=str, required=False)


if __name__ == "__main__":
    args = parser.parse_args()
    
    pipeline = DataAnnotatorPipeline(
        args.label,
        args.annotating_system_prompt_path,
        args.checking_system_prompt_path,
        args.data_path,
        args.output_folder,
        args.prompt_round,
        args.optimization_flag
    )
    if args.error_flag:
        pipeline.annotate_error_data(args.error_batch_folder)
    else:
        # pipeline.annotate()
        # pipeline.check()
        pipeline.main()
