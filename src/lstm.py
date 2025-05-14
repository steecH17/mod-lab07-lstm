import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.optimizers import RMSprop
import random


def get_words_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text.split()

words = get_words_from_file('src/input.txt')


unique_words = sorted(list(set(words)))
word_to_num = {word: i for i, word in enumerate(unique_words)}
num_to_word = {i: word for i, word in enumerate(unique_words)}


seq_length = 10
step_size = 1
vocab_size = len(unique_words)


def create_training_data(words, seq_len, step):
    sequences = []
    next_words = []
    for i in range(0, len(words) - seq_len, step):
        sequences.append(words[i: i + seq_len])
        next_words.append(words[i + seq_len])
    return sequences, next_words

sequences, next_words = create_training_data(words, seq_length, step_size)


X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
y = np.zeros((len(sequences), vocab_size), dtype=np.bool)

for i, seq in enumerate(sequences):
    for t, word in enumerate(seq):
        X[i, t, word_to_num[word]] = 1
    y[i, word_to_num[next_words[i]]] = 1


def build_lstm_model(seq_len, vocab_size):
    model = Sequential([
        LSTM(128, input_shape=(seq_len, vocab_size)),
        Dense(vocab_size),
        Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
    return model

text_model = build_lstm_model(seq_length, vocab_size)


def select_next_word(predictions, temp=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temp
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, predictions, 1))


text_model.fit(X, y, batch_size=128, epochs=50)


def make_new_text(output_length, randomness):
    start_pos = random.randint(0, len(words) - seq_length - 1)
    current_seq = words[start_pos: start_pos + seq_length]
    generated_words = current_seq.copy()

    for _ in range(output_length):
        x_input = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(current_seq):
            x_input[0, t, word_to_num[word]] = 1.

        preds = text_model.predict(x_input, verbose=0)[0]
        next_word_idx = select_next_word(preds, randomness)
        next_word = num_to_word[next_word_idx]

        generated_words.append(next_word)
        current_seq = current_seq[1:] + [next_word]

    lines = []
    for i in range(0, len(generated_words), 20):
        line = ' '.join(generated_words[i:i+20])
        lines.append(line)
    return '\n'.join(lines)

result_text = make_new_text(1000, 0.2)
with open('result/gen.txt', 'w', encoding='utf-8') as output_file:
    output_file.write(result_text)

print("Текст успешно сгенерирован и сохранен в gen.txt")