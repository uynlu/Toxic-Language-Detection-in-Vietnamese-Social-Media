import argparse

from labeling.labeling_utils.errors_analysis import measure_agreement


parser = argparse.ArgumentParser(description="Measure agreement")
parser.add_argument("--result-folder", type=str, required=True)
parser.add_argument("--label-type", type=str, required=True)
parser.add_argument("--agreement-folder", type=str, required=True)
parser.add_argument("--prompt-round", type=int, required=True)
parser.add_argument("--sample", type=int, required=False)


if __name__ == "__main__":
    args = parser.parse_args()

    measure_agreement(
        result_folder=args.result_folder,
        label_type=args.label_type,
        agreement_folder=args.agreement_folder,
        prompt_round=args.prompt_round,
        sample=args.sample
    )
