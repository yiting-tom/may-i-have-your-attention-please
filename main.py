import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from data import SST2Dataset
from model import BertForSequenceClassification


BATCH_SIZE = 32
SEED = 9527
DEVICE = None
write = SummaryWriter()


def test_one_epoch(
    dl: DataLoader,
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
) -> float:
    total_train_loss = 0

    model.eval()

    with torch.no_grad():

        for step, input_dict in tqdm(enumerate(dl), total=len(dl)):
            label, input_ids, attn_mask = (
                input_dict["label"],
                input_dict["input_ids"],
                input_dict["attention_mask"],
            )
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)
            label = label.to(DEVICE)

            output = model.forward(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attn_mask,
                output_hidden_states=True,
                labels=label,
                output_attentions=True,
            )
            total_train_loss += output.loss
            output_layer = output.hidden_states[-1]
            output_attentions = output.attentions[-1]
            # print(f"{output_layer[0, ...].sum(dim=-1).tolist() = }")
            print(f"{len(model.bert.encoder.mask_list) = }")
            print(f"{model.bert.encoder.mask_list[-1].shape = }")
            print(f"{input_ids[-1].shape = }")
            print(f"==========================")
            print(f"{model.bert.encoder.mask_list[-1].tolist() = }")
            print(f"{tokenizer.decode(input_ids[-1]) = }")
            # print(f"{output_attentions.shape = }", "\n" * 3)
            # print(f"{output_attentions[0, ...].tolist() = }")
            input()

        return total_train_loss / len(dl)


def train_one_epoch(
    dl: DataLoader,
    model: BertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
    *,
    total_step: int = 0,
) -> float:
    total_train_loss = 0

    for step, input_dict in tqdm(enumerate(dl), total=len(dl)):
        label, input_ids, attn_mask = (
            input_dict["label"],
            input_dict["input_ids"],
            input_dict["attention_mask"],
        )
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        label = label.to(DEVICE)

        model.zero_grad()
        output = model.forward(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attn_mask,
            labels=label,
        )
        total_train_loss += output.loss
        output.loss.backward()
        optimizer.step()
        write.add_scalar("training-loss", output.loss, step + total_step)

    return total_train_loss / len(dl)


def train_n_epochs(
    epochs: int,
    dl: DataLoader,
    model: BertForSequenceClassification,
    optimizer: torch.optim.Optimizer,
) -> float:

    model.train()

    with torch.enable_grad():
        for e in range(epochs):
            print(f"========== epoch {e} ==========")
            loss = train_one_epoch(
                dl, model, optimizer, total_step=e * len(dl)
            )
            print(f"loss: {loss}")


def init_seed():
    global SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def init_device() -> torch.DeviceObjType:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    global DEVICE
    DEVICE = device


def main():
    init_seed()
    init_device()

    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased", do_lower_case=True
    )
    # ds = ColaDataset("data/CoLA/train.tsv", tokenizer)
    ds = SST2Dataset(tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
        output_attentions=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon  - default is 1e-8.
    )
    model = BertForSequenceClassification.from_pretrained("model.ckpt")

    model.to(DEVICE)
    train_n_epochs(10, dl, model, optimizer)
    model.save_pretrained("model.ckpt")
    test_one_epoch(dl, model, tokenizer)


if __name__ == "__main__":
    main()
