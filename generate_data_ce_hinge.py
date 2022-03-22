import pickle
import sys
import torch
from transformers import GPT2TokenizerFast
import pandas as pd
import os
import json


# def generate(l, tokenizer, model, pad_token_dict, num_samples=1000):
#     model.eval()
#     temp_list = ["<|labelpad|>"] * pad_token_dict[l]
#     if len(temp_list) > 0:
#         label_str = " ".join(l.split("_")) + " " + " ".join(temp_list)
#     else:
#         label_str = " ".join(l.split("_"))
#     text = tokenizer.bos_token + " " + label_str + " <|labelsep|> "
#
#     sents = []
#     sample_outputs = model.generate(
#         input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
#         do_sample=True,
#         top_k=50,
#         max_length=200,
#         top_p=0.95,
#         num_return_sequences=num_samples
#     )
#     for i, sample_output in enumerate(sample_outputs):
#         # print("{}: {}".format(i, tokenizer.decode(sample_output)))
#         sents.append(tokenizer.decode(sample_output))
#     return sents

def generate(l, tokenizer, model, pad_token_dict, num_samples=1000):
    model.eval()
    temp_list = ["<|labelpad|>"] * pad_token_dict[l]
    if len(temp_list) > 0:
        label_str = " ".join(l.split("_")) + " " + " ".join(temp_list)
    else:
        label_str = " ".join(l.split("_"))
    text = label_str + " <|labelsep|> "
    encoded_dict = tokenizer.encode_plus(text, return_tensors='pt')
    ids = torch.tensor([[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]).to(device)

    sents = []
    its = num_samples / 250
    if its < 1:
        sample_outputs = model.generate(
            input_ids=ids,
            do_sample=True,
            top_k=50,
            max_length=200,
            top_p=0.95,
            num_return_sequences=num_samples
        )
        for i, sample_output in enumerate(sample_outputs):
            # print("{}: {}".format(i, tokenizer.decode(sample_output)))
            sents.append(tokenizer.decode(sample_output))
    else:
        for it in range(int(its)):
            sample_outputs = model.generate(
                input_ids=ids,
                do_sample=True,
                top_k=50,
                max_length=200,
                top_p=0.95,
                num_return_sequences=250
            )
            for i, sample_output in enumerate(sample_outputs):
                # print("{}: {}".format(i, tokenizer.decode(sample_output)))
                sents.append(tokenizer.decode(sample_output))
    return sents


def post_process(sentences):
    proc_sents = []
    label_sep_token = '<|labelsep|>'
    label_pad_token = '<|labelpad|>'
    pad_token = '<|pad|>'
    bos_token = '<|startoftext|>'
    remove_list = [label_sep_token, label_pad_token, pad_token, bos_token]

    for sent in sentences:
        ind = sent.find(label_sep_token)
        temp_sent = sent[ind + len(label_sep_token):].strip()
        temp_sent = ' '.join([i for i in temp_sent.strip().split() if i not in remove_list])
        proc_sents.append(temp_sent)
    return proc_sents


if __name__ == "__main__":
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    parent_label = sys.argv[3]
    num = int(sys.argv[4])

    device = torch.device('cuda:0')

    with open(os.path.join(data_dir, "parent_to_child.json")) as f:
        parent_to_child = json.load(f)

    fine_tok_path = os.path.join(model_dir, "gpt2/coarse_fine/tokenizer")
    fine_model_path = os.path.join(model_dir, "gpt2/coarse_fine/model/")

    pad_token_dict = pickle.load(open(os.path.join(data_dir, "pad_token_dict.pkl"), "rb"))

    fine_tokenizer = GPT2TokenizerFast.from_pretrained(fine_tok_path, do_lower_case=True)
    fine_model = torch.load(fine_model_path + "coarse_fine.pt", map_location=device)

    all_sents = []
    all_labels = []
    children = parent_to_child[parent_label]
    for ch in children:
        sentences = generate(ch, fine_tokenizer, fine_model, pad_token_dict, num_samples=num)
        sentences = post_process(sentences)
        labels = [ch] * num
        all_sents += sentences
        all_labels += labels

    df = pd.DataFrame.from_dict({"text": all_sents, "label": all_labels})
    pickle.dump(df, open(os.path.join(data_dir, "df_gen_" + parent_label + ".pkl"), "wb"))
