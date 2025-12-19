import argparse
import os

from dataset.custom_dataset import CustomDataset
from model_running.model_utils.model_executor import ModelExecutor


parser = argparse.ArgumentParser(description="Train model")
parser.add_argument("--pretrained-flag", type=str, required=True)
parser.add_argument("--data-folder", type=str, required=True)
parser.add_argument("--label-type", type=str, required=True)
parser.add_argument("--pretrained-name", type=str, required=False)
parser.add_argument("--cache-dir", type=str, required=False)
parser.add_argument("--max-len", type=int, required=True)
parser.add_argument("--batch-size", type=int, required=True)
parser.add_argument("--num-labels", type=int, required=True)
parser.add_argument("--checkpoint-path", type=str, required=True)
parser.add_argument("--freeze-model", type=bool, required=False)
parser.add_argument("--num-epochs", type=int, required=True)
parser.add_argument("--vocab-folder-path", type=str, required=False)
parser.add_argument("--model-name", type=str, required=False)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.pretrained_flag:
        train_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "train.json"),
            label_type=args.label_type,
            tokenizer_name=args.pretrained_name,
            cache_dir=args.cache_dir,
            max_len=args.max_len
        )
        dev_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "dev.json"),
            label_type=args.label_type,
            tokenizer_name=args.pretrained_name,
            cache_dir=args.cache_dir,
            max_len=args.max_len
        )
        test_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "test.json"),
            label_type=args.label_type,
            tokenizer_name=args.pretrained_name,
            cache_dir=args.cache_dir,
            max_len=args.max_len
        )

        executor = ModelExecutor(
            batch_size=args.batch_size,
            model_name=args.pretrained_name,
            num_labels=args.num_labels,
            cache_dir=args.cache_dir,
            checkpoint_path=args.checkpoint_path,
            freeze_model=args.freeze_model,
            num_epochs=args.num_epochs
        )

        executor.run(train_data, dev_data, test_data)
    else:
        train_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "train.json"),
            label_type=args.label_type,
            max_len=args.max_len,
            vocab_folder_path=args.vocab_folder_path
        )
        dev_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "dev.json"),
            label_type=args.label_type,
            max_len=args.max_len,
            vocab_folder_path=args.vocab_folder_path
        )
        test_data = CustomDataset(
            data_path=os.path.join(args.data_folder, "test.json"),
            label_type=args.label_type,
            max_len=args.max_len,
            vocab_folder_path=args.vocab_folder_path
        )

        executor = ModelExecutor(
            batch_size=args.batch_size,
            model_name=args.model_name,
            num_labels=args.num_labels,
            checkpoint_path=args.checkpoint_path,
            num_epochs=args.num_epochs
        )

        executor.run(train_data, dev_data, test_data)
        