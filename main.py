import utils
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import train

def main():
    # Load files
    neg_text = utils.load_text('data/rt-polaritydata/rt-polarity.neg')
    pos_text = utils.load_text('data/rt-polaritydata/rt-polarity.pos')

    # Concatenate and label data
    texts = np.array(neg_text + pos_text)
    labels = np.array([0]*len(neg_text) + [1]*len(pos_text))

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Tokenize, build vocabulary, encode tokens
    print("Tokenizing...\n")
    tokenized_texts, word2idx, max_len = utils.tokenize(texts)
    input_ids = utils.encode(tokenized_texts, word2idx, max_len)

    # Load pretrained vectors
    embeddings = utils.load_pretrained_vectors(word2idx, "fastText/crawl-300d-2M.vec")
    embeddings = torch.tensor(embeddings)

    # Train Test Split
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    input_ids, labels, test_size=0.1, random_state=42)

    # Load data to PyTorch DataLoader
    train_dataloader, val_dataloader = \
    utils.data_loader(train_inputs, val_inputs, train_labels, val_labels, batch_size=50)

    # Sample configuration:
    filter_sizes = [2, 3, 4]
    num_filters = [2, 2, 2]

    # CNN-rand: Word vectors are randomly initialized.
    utils.set_seed(42)
    cnn_rand, optimizer = utils.initilize_model(vocab_size=len(word2idx),
                                      embed_dim=300,
                                      learning_rate=0.25,
                                      dropout=0.5,
                                      device = device)
    train.train(cnn_rand, optimizer, train_dataloader, val_dataloader, epochs=20)

if __name__ == '__main__':
    main()