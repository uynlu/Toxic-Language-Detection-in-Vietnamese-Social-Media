import argparse
import os

from labeling.labeling_utils.data_annotator import DataAnnotatorPipeline


parser = argparse.ArgumentParser(description="Label data.")
parser.add_argument("--label-type", type=str, required=True)
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
        label_type=args.label_type,
        annotating_system_prompt_path=args.annotating_system_prompt_path,
        checking_system_prompt_path=args.checking_system_prompt_path,
        data_path=args.data_path,
        output_folder=args.output_folder,
        prompt_round=args.prompt_round,
        optimization_flag=args.optimization_flag
    )
    if args.error_flag:
        if not args.error_batch_folder:
            for folder in os.listdir(os.path.join(args.output_folder, f"round_{args.prompt_round}", "annotated")):
                folder_path = os.path.join(args.output_folder, f"round_{args.prompt_round}", "annotated", folder)

                print(f"Processing folder: {folder_path}")
                error_file = os.path.join(folder_path, "error_data.json")
                if os.path.exists(error_file):
                    pipeline.annotate_error_data(error_batch_folder=folder_path)
        else:
            pipeline.annotate_error_data(error_batch_folder=args.error_batch_folder)
    else:
        # pipeline.annotate()
        # pipeline.check()
        # pipeline.main()
        # pipeline.annotate_temp()
        pipeline.main_parallelly()
        