import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AdamW
from gpt2_coarse_finetune import basic_gpt2_tokenize, create_data_loaders, test_generate, format_time
from torch.utils.data import TensorDataset
from torch.nn import CrossEntropyLoss, MarginRankingLoss
import os
import pickle
import sys
import numpy as np
import random
import time
import pandas as pd


def gpt2_hinge_tokenize(tokenizer, sentences, labels, pad_token_dict, child_to_parent, max_length=768):
    input_ids = []
    attention_masks = []
    # For every sentence...
    num_sentences = len(sentences)

    for i, sent in enumerate(sentences):
        hinge_input_ids = []
        hinge_attn_masks = []
        for label in [labels[i], child_to_parent[labels[i]]]:
            processed_label_str = " ".join(label.split("_"))
            temp_list = ["<|labelpad|>"] * pad_token_dict[label]
            if len(temp_list) > 0:
                label_str = processed_label_str + " " + " ".join(temp_list)
            else:
                label_str = processed_label_str
            encoded_dict = tokenizer.encode_plus(
                label_str + " <|labelsep|> " + sent,  # Sentence to encode.
                truncation=True,
                max_length=max_length - 1,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors='pt',  # Return pytorch tensors.
            )
            encoded_dict['input_ids'] = torch.tensor(
                [[tokenizer.bos_token_id] + encoded_dict['input_ids'].data.tolist()[0]]
            )
            encoded_dict['attention_mask'] = torch.tensor(
                [[1] + encoded_dict['attention_mask'].data.tolist()[0]]
            )
            hinge_input_ids.append(encoded_dict['input_ids'])
            hinge_attn_masks.append(encoded_dict['attention_mask'])

        # Add the encoded sentence to the list.
        input_ids.append(torch.cat(hinge_input_ids, dim=0))

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(torch.cat(hinge_attn_masks, dim=0))
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).view(num_sentences, -1, max_length)
    attention_masks = torch.cat(attention_masks, dim=0).view(num_sentences, -1, max_length)
    return input_ids, attention_masks


def compute_doc_prob(logits, b_fine_input_mask, b_fine_labels, doc_start_ind):
    mask = b_fine_input_mask > 0
    maski = mask.unsqueeze(-1).expand_as(logits)
    logits_pad_removed = torch.masked_select(logits, maski).view(-1, logits.size(-1)).unsqueeze(0)
    logits_pad_removed = logits_pad_removed[:, doc_start_ind - 1:-1, :]

    b_fine_labels_pad_removed = torch.masked_select(b_fine_labels, mask).unsqueeze(0)
    b_fine_labels_pad_removed = b_fine_labels_pad_removed[:, doc_start_ind:]
    log_probs = logits_pad_removed.gather(2, b_fine_labels_pad_removed.unsqueeze(dim=-1)).squeeze(dim=-1).squeeze(
        dim=0)
    return log_probs.sum()


def train(model, tokenizer, coarse_train_dataloader, coarse_validation_dataloader, fine_train_dataloader,
          fine_validation_dataloader, doc_start_ind, parent_labels, child_labels, device):
    def calculate_ce_loss(lm_logits, b_labels, b_input_mask, doc_start_ind):
        loss_fct = CrossEntropyLoss()
        batch_size = lm_logits.shape[0]
        logits_collected = []
        labels_collected = []
        for b in range(batch_size):
            logits_ind = lm_logits[b, :, :]  # seq_len x |V|
            labels_ind = b_labels[b, :]  # seq_len
            mask = b_input_mask[b, :] > 0
            maski = mask.unsqueeze(-1).expand_as(logits_ind)
            # unpad_seq_len x |V|
            logits_pad_removed = torch.masked_select(logits_ind, maski).view(-1, logits_ind.size(-1))
            labels_pad_removed = torch.masked_select(labels_ind, mask)  # unpad_seq_len

            shift_logits = logits_pad_removed[doc_start_ind - 1:-1, :].contiguous()
            shift_labels = labels_pad_removed[doc_start_ind:].contiguous()
            # Flatten the tokens
            logits_collected.append(shift_logits.view(-1, shift_logits.size(-1)))
            labels_collected.append(shift_labels.view(-1))

        logits_collected = torch.cat(logits_collected, dim=0)
        labels_collected = torch.cat(labels_collected, dim=0)
        loss = loss_fct(logits_collected, labels_collected)
        return loss

    def calculate_hinge_loss(fine_log_probs, other_log_probs):
        loss_fct = MarginRankingLoss(margin=1.609)
        length = len(other_log_probs)
        temp_tensor = []
        for i in range(length):
            temp_tensor.append(fine_log_probs)
        temp_tensor = torch.cat(temp_tensor, dim=0)
        other_log_probs = torch.cat(other_log_probs, dim=0)
        y_vec = torch.ones(length).to(device)
        loss = loss_fct(temp_tensor, other_log_probs, y_vec)
        return loss

    def calculate_loss(lm_logits, b_labels, b_input_mask, doc_start_ind, fine_log_probs, other_log_probs,
                       lambda_1=0.01, is_fine=True):
        ce_loss = calculate_ce_loss(lm_logits, b_labels, b_input_mask, doc_start_ind)
        if is_fine:
            hinge_loss = calculate_hinge_loss(fine_log_probs, other_log_probs)
            print("CE-loss", ce_loss.item(), "Hinge-loss", hinge_loss.item(), flush=True)
        else:
            hinge_loss = 0
            print("CE-loss", ce_loss.item(), "Hinge-loss", hinge_loss, flush=True)
        return ce_loss + lambda_1 * hinge_loss

    optimizer = AdamW(model.parameters(),
                      lr=5e-4,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    sample_every = 100
    warmup_steps = 1e2
    epochs = 5
    total_steps = (len(coarse_train_dataloader) + len(fine_train_dataloader)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    seed_val = 81
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    total_t0 = time.time()

    for epoch_i in range(0, epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)
        t0 = time.time()
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(coarse_train_dataloader):
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(
                    '  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(coarse_train_dataloader), elapsed),
                    flush=True)
                model.eval()
                lbl = random.choice(parent_labels)
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + " " + " ".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = tokenizer.bos_token + " " + label_str + " <|labelsep|> "
                sample_outputs = model.generate(
                    input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output)), flush=True)
                model.train()

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = calculate_loss(outputs[1], b_labels, b_input_mask, doc_start_ind, None, None, is_fine=False)
            # loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        for step, batch in enumerate(fine_train_dataloader):
            # batch contains -> fine_input_ids mini batch, fine_attention_masks mini batch
            if step % sample_every == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(fine_train_dataloader), elapsed),
                      flush=True)
                model.eval()
                lbl = random.choice(child_labels)
                temp_list = ["<|labelpad|>"] * pad_token_dict[lbl]
                if len(temp_list) > 0:
                    label_str = " ".join(lbl.split("_")) + " " + " ".join(temp_list)
                else:
                    label_str = " ".join(lbl.split("_"))
                text = tokenizer.bos_token + " " + label_str + " <|labelsep|> "
                sample_outputs = model.generate(
                    input_ids=tokenizer.encode(text, return_tensors='pt').to(device),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output)), flush=True)
                model.train()

            b_fine_input_ids_minibatch = batch[0].to(device)
            b_fine_input_mask_minibatch = batch[1].to(device)

            b_size = b_fine_input_ids_minibatch.shape[0]
            assert b_size == 1
            mini_batch_size = b_fine_input_ids_minibatch.shape[1]

            model.zero_grad()

            batch_other_log_probs = []
            prev_mask = None

            for b_ind in range(b_size):
                for mini_batch_ind in range(mini_batch_size):
                    b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    b_fine_labels = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                    outputs = model(b_fine_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_fine_input_mask,
                                    labels=b_fine_labels)
                    log_probs = torch.log_softmax(outputs[1], dim=-1)
                    doc_prob = compute_doc_prob(log_probs, b_fine_input_mask, b_fine_labels, doc_start_ind).unsqueeze(0)
                    if mini_batch_ind == 0:
                        batch_fine_log_probs = doc_prob
                        orig_output = outputs
                        orig_labels = b_fine_labels
                        orig_mask = b_fine_input_mask
                    else:
                        batch_other_log_probs.append(doc_prob)
                    if prev_mask is not None:
                        assert torch.all(b_fine_input_mask.eq(prev_mask))
                    prev_mask = b_fine_input_mask

            loss = calculate_loss(orig_output[1], orig_labels, orig_mask, doc_start_ind, batch_fine_log_probs,
                                  batch_other_log_probs, is_fine=True)
            # loss = criterion(batch_fine_probs.log(), batch_coarse_probs.detach()).sum(dim=-1).mean(dim=-1).mean(dim=-1)
            total_train_loss += loss.item()
            print("Loss:", loss.item(), flush=True)

            loss.backward()
            optimizer.step()
            scheduler.step()

        # **********************************

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / (len(coarse_train_dataloader) + len(fine_train_dataloader))

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epcoh took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in coarse_validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)

            # Accumulate the validation loss.
            loss = calculate_loss(outputs[1], b_labels, b_input_mask, doc_start_ind, None, None, is_fine=False)
            # loss = outputs[0]
            total_eval_loss += loss.item()

        for batch in fine_validation_dataloader:
            # batch contains -> fine_input_ids mini batch, fine_attention_masks mini batch
            b_fine_input_ids_minibatch = batch[0].to(device)
            b_fine_input_mask_minibatch = batch[1].to(device)

            b_size = b_fine_input_ids_minibatch.shape[0]
            assert b_size == 1
            mini_batch_size = b_fine_input_ids_minibatch.shape[1]

            with torch.no_grad():
                batch_other_log_probs = []
                prev_mask = None

                for b_ind in range(b_size):
                    for mini_batch_ind in range(mini_batch_size):
                        b_fine_input_ids = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                        b_fine_labels = b_fine_input_ids_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(device)
                        b_fine_input_mask = b_fine_input_mask_minibatch[b_ind, mini_batch_ind, :].unsqueeze(0).to(
                            device)
                        outputs = model(b_fine_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_fine_input_mask,
                                        labels=b_fine_labels)
                        log_probs = torch.log_softmax(outputs[1], dim=-1)
                        doc_prob = compute_doc_prob(log_probs, b_fine_input_mask, b_fine_labels,
                                                    doc_start_ind).unsqueeze(0)
                        if mini_batch_ind == 0:
                            batch_fine_log_probs = doc_prob
                            orig_output = outputs
                            orig_labels = b_fine_labels
                            orig_mask = b_fine_input_mask
                        else:
                            batch_other_log_probs.append(doc_prob)
                        if prev_mask is not None:
                            assert torch.all(b_fine_input_mask.eq(prev_mask))
                        prev_mask = b_fine_input_mask

            loss = calculate_loss(orig_output[1], orig_labels, orig_mask, doc_start_ind, batch_fine_log_probs,
                                  batch_other_log_probs, is_fine=True)
            total_eval_loss += loss.item()

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / (len(coarse_validation_dataloader) + len(fine_validation_dataloader))

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model


if __name__ == "__main__":
    # basepath = "/Users/dheerajmekala/Work/Coarse2Fine/data/"
    basepath = "/data/dheeraj/coarse2fine/"
    dataset = sys.argv[4] + "/"
    pkl_dump_dir = basepath + dataset

    seed_val = 81
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    algo = sys.argv[5]
    tok_path = pkl_dump_dir + "gpt2/coarse_fine/" + algo + "/tokenizer"
    model_path = pkl_dump_dir + "gpt2/coarse_fine/" + algo + "/model/"
    model_name = "coarse_fine.pt"

    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    use_gpu = int(sys.argv[1])
    # use_gpu = False
    gpu_id = int(sys.argv[2])
    iteration = int(sys.argv[3])
    # iteration = 1

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(pkl_dump_dir + "df_coarse.pkl", "rb"))
    parent_to_child = pickle.load(open(pkl_dump_dir + "parent_to_child.pkl", "rb"))

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>', pad_token='<|pad|>',
                                              additional_special_tokens=['<|labelsep|>', '<|labelpad|>'])

    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    child_to_parent = {}
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            child_to_parent[ch] = p

    parent_labels = []
    child_labels = []
    for p in parent_to_child:
        parent_labels.append(p)
        child_labels += parent_to_child[p]

    all_labels = parent_labels + child_labels

    pad_token_dict = {}
    max_num = -float("inf")
    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        max_num = max(max_num, len(tokens))

    doc_start_ind = 1 + max_num + 1  # this gives the token from which the document starts in the inputids, 1 for the starttoken, max_num for label info, 1 for label_sup

    for l in all_labels:
        tokens = tokenizer.tokenize(" ".join(l.split("_")))
        pad_token_dict[l] = max_num - len(tokens)
    # tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<|startoftext|> sports <|labelsep|> Hello, my dog is cute <|endoftext|>")))

    df_weaksup = None
    for p in parent_to_child:
        for ch in parent_to_child[p]:
            temp_df = pickle.load(
                open(pkl_dump_dir + "exclusive/" + algo + "/" + str(iteration) + "it/" + ch + ".pkl", "rb"))
            temp_df["label"] = [ch] * len(temp_df)
            if df_weaksup is None:
                df_weaksup = temp_df
            else:
                df_weaksup = pd.concat([df_weaksup, temp_df])

    coarse_input_ids, coarse_attention_masks = basic_gpt2_tokenize(tokenizer, df.text.values, df.label.values,
                                                                   pad_token_dict)
    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(coarse_input_ids, coarse_attention_masks)

    # Create a 90-10 train-validation split.
    coarse_train_dataloader, coarse_validation_dataloader = create_data_loaders(dataset, batch_size=4)

    fine_input_ids, fine_attention_masks = gpt2_hinge_tokenize(tokenizer, df_weaksup.text.values,
                                                               df_weaksup.label.values, pad_token_dict, child_to_parent)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(fine_input_ids, fine_attention_masks)

    # Create a 90-10 train-validation split.
    fine_train_dataloader, fine_validation_dataloader = create_data_loaders(dataset, batch_size=1)

    model = train(model,
                  tokenizer,
                  coarse_train_dataloader,
                  coarse_validation_dataloader,
                  fine_train_dataloader,
                  fine_validation_dataloader,
                  doc_start_ind,
                  parent_labels,
                  child_labels,
                  device)
    test_generate(model, tokenizer, all_labels, pad_token_dict, device)

    tokenizer.save_pretrained(tok_path)
    torch.save(model, model_path + model_name)
    pickle.dump(pad_token_dict, open(pkl_dump_dir + "pad_token_dict.pkl", "wb"))
