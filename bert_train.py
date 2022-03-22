from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import pickle
import torch
import sys
import numpy as np
import time
import random
import datetime
import os
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
import json


# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def bert_tokenize(tokenizer, df, label_to_index):
    input_ids = []
    attention_masks = []
    # For every sentence...
    sentences = df.text.values
    labels = copy.deepcopy(df.label.values)
    for i, l in enumerate(list(labels)):
        labels[i] = label_to_index[l]
    labels = np.array(labels, dtype='int32')
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.LongTensor(labels)
    # Print sentence 0, now as a list of IDs.
    return input_ids, attention_masks, labels


def create_data_loaders(dataset):
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )
    return train_dataloader, validation_dataloader


def train(train_dataloader, validation_dataloader, device, num_labels):
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if device == torch.device("cuda"):
        torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

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

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy), flush=True)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

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
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model


def evaluate(model, prediction_dataloader, device):
    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)), flush=True)

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels


def test(df_test_original, label_to_index, index_to_label):
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_test_original, label_to_index)
    # Set the batch size.
    batch_size = 32
    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    predictions, true_labels = evaluate(model, prediction_dataloader, device)
    preds = []
    for i, pred in enumerate(predictions):
        if i == 0:
            pred_probs = pred
        else:
            pred_probs = np.concatenate((pred_probs, pred))

        preds = preds + list(pred.argmax(axis=-1))
    true = []
    for t in true_labels:
        true = true + list(t)

    for i, t in enumerate(true):
        true[i] = index_to_label[t]
        preds[i] = index_to_label[preds[i]]

    print(classification_report(true, preds), flush=True)
    return true, preds, pred_probs


def get_high_quality_inds(true, preds, pred_probs, label_to_index, num, threshold=0.7, percent_threshold=20):
    pred_inds = []
    for p in preds:
        pred_inds.append(label_to_index[p])

    pred_label_to_inds = {}
    for i, p in enumerate(pred_inds):
        try:
            pred_label_to_inds[p].append(i)
        except:
            pred_label_to_inds[p] = [i]

    label_to_probs = {}
    min_ct = float("inf")
    for p in pred_label_to_inds:
        label_to_probs[p] = []
        ct_thresh = 0
        for ind in pred_label_to_inds[p]:
            temp = pred_probs[ind][p]
            if temp >= threshold:
                ct_thresh += 1
            label_to_probs[p].append(temp)
        min_ct = min(min_ct, ct_thresh)
    # min_ct = min(min_ct, int((percent_threshold / (len(label_to_index) * 100.0)) * len(preds)))
    min_ct = num
    print("Collecting", min_ct, "samples as high quality")
    final_inds = {}
    for p in label_to_probs:
        probs = label_to_probs[p]
        inds = np.array(probs).argsort()[-min_ct:][::-1]
        final_inds[p] = []
        for i in inds:
            final_inds[p].append(pred_label_to_inds[p][i])

    temp_true = []
    temp_preds = []
    for p in final_inds:
        for ind in final_inds[p]:
            temp_true.append(true[ind])
            temp_preds.append(preds[ind])

    print("Classification Report of High Quality data")
    print(classification_report(temp_true, temp_preds), flush=True)
    return final_inds


if __name__ == "__main__":
    data_dir = sys.argv[1]
    model_dir = sys.argv[2]
    iteration = sys.argv[3]
    parent_label = sys.argv[4]

    device = torch.device('cuda:0')
    tok_path = os.path.join(model_dir, "bert/" + parent_label + "/tokenizer")
    model_path = os.path.join(model_dir, "bert/" + parent_label + "/model")
    os.makedirs(tok_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(data_dir, "num_dic.json")) as f:
        num_dic = json.load(f)

    df_train = pickle.load(open(os.path.join(data_dir, "df_gen_" + parent_label + ".pkl"), "rb"))
    df_fine = pickle.load(open(os.path.join(data_dir, "df_fine.pkl"), "rb"))
    df_test = df_fine[df_fine["label"].isin(list(set(df_train.label.values)))].reset_index(drop=True)

    with open(os.path.join(data_dir, "parent_to_child.json")) as f:
        parent_to_child = json.load(f)

    for ch in parent_to_child[parent_label]:
        for i in range(1, iteration + 1):
            temp_child_df = pickle.load(open(os.path.join(data_dir, "exclusive/" + str(i) + "it/" + ch + ".pkl"), "rb"))
            if i == 1:
                child_df = temp_child_df
            else:
                child_df = pd.concat([child_df, temp_child_df])
        child_df["label"] = [ch] * len(child_df)
        df_train = pd.concat([df_train, child_df])

    print(df_train.label.value_counts())

    # Tokenize all of the sentences and map the tokens to their word IDs.
    print('Loading BERT tokenizer...', flush=True)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)

    label_set = set(df_train.label.values)
    label_to_index = {}
    index_to_label = {}
    for i, l in enumerate(list(label_set)):
        label_to_index[l] = i
        index_to_label[i] = l

    input_ids, attention_masks, labels = bert_tokenize(tokenizer, df_train, label_to_index)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    # Tell pytorch to run this model on the GPU.

    model = train(train_dataloader, validation_dataloader, device, num_labels=len(label_to_index))
    true, preds, pred_probs = test(df_test, label_to_index, index_to_label)
    high_quality_inds = get_high_quality_inds(true, preds, pred_probs, label_to_index, num_dic[parent_label],
                                              percent_threshold=20 * iteration)

    for p in high_quality_inds:
        inds = high_quality_inds[p]
        temp_df = df_test.loc[inds].reset_index(drop=True)
        os.makedirs(os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it"), exist_ok=True)
        pickle.dump(temp_df, open(
            os.path.join(data_dir, "exclusive/" + str(iteration + 1) + "it/" + index_to_label[p] + ".pkl"), "wb"))

    df_test["pred"] = preds
    pickle.dump(df_test, open(os.path.join(data_dir, "preds_" + parent_label + ".pkl"), "wb"))
