from __future__ import division, print_function, unicode_literals

import random

import pandas as pd
import torch
import torch.nn as nn
from models import AttnDecoderRNN, EncoderRNN
from preprocessing import prepareData, preprocess, tensorFromSentence
from torch import optim
from train import train
from utils import MAX_LENGTH, EOS_token, SOS_token, device, hidden_size, learning_rate

import ignite.engine
from ignite.metrics import Rouge

data_path = "examples/nlp/seq2seq/news_summary.csv"

data = pd.read_csv(data_path, encoding="utf-8")
data.head()

Engine = ignite.engine.Engine
Events = ignite.engine.Events

data["headlines"] = data["headlines"].apply(lambda x: preprocess(x))
data["text"] = data["text"].apply(lambda x: preprocess(x))
# print(data['headlines'][20], data['text'][20])

x = data["text"]
y = data["headlines"]
# print(x[50], y[50], sep='\n')


input_lang, output_lang, pairs = prepareData(x, y, False)
# print(random.choice(pairs))


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(5)]  # Maximum size is 175000
criterion = nn.NLLLoss()


def update(engine, training_pair):
    input_tensor = training_pair[0]
    target_tensor = training_pair[1]

    loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    return loss


trainer = Engine(update)
event = Events.ITERATION_COMPLETED


@trainer.on(event)
def log_training(engine):
    batch_loss = engine.state.output
    lr = encoder_optimizer.param_groups[0]["lr"]
    e = engine.state.epoch
    n = engine.state.max_epochs
    i = engine.state.iteration
    print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, lr: {lr}")


trainer.run(training_pairs, max_epochs=5)


evaluation_pairs = [random.choice(pairs) for i in range(5)]


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[: di + 1]


def predict_on_batch(engine, batch):
    x = batch[0]
    y = batch[1]
    y_pred, attentions = evaluate(encoder, decoder, x)
    y_pred = " ".join(y_pred)
    return y_pred.split(), [y.split()]


evaluator = Engine(predict_on_batch)
Rouge(variants=[1]).attach(evaluator, "rouge")
state = evaluator.run(evaluation_pairs)
state.metrics["rouge"]
