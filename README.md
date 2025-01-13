The scope of this exercise is to implement, train and test a Sequence-toSequence model
using a single LSTM layer Encode and Decoder and to experiment
with different values for EMB_DIM, HID_DIM, BATCH_SIZE, TEACHER_RATIO AND
DROPOUT. The second objective of this exercise consists in training and testing the
best configuration found in the first objective, but with a bidirectional LSTM
Encoder layer.

I trained a total of 4 experiments for the first objective and 2 other
configurations for the second objective of this exercise.

Experiments 1 and 5, with a teacher forcing ratio of 0.5, performed better
than those with a teacher forcing ratio of 1.0, showing that a balanced approach
between teacher forcing and free running during training improves generalization
and prediction quality.

Also, both experiments demonstrated that a well-balanced embedding
dimension and hidden dimension (256 each) yielded good results, possibly
preventing overfitting while capturing enough contextual information.

For experiment 3, initially, there was a sharp increase in validation loss and
perplexity in the 2nd epoch, possibly due to overfitting from the high embedding
dimension or lack of generalization due to full teacher forcing, perhaps leading to
difficulties and errors during inference. Although BLEU score improved slightly over
time, high validation loss and perplexity persisted.

Experiment 5 (with a bidirectional encoder) offered the best performance in
terms of BLEU score. The bidirectional LSTM architecture captures more contextual
information, especially for sentences (sequences) where dependencies between
words (tokens) may span both directions.

Adding a bidirectional layer for the 4th experiment made the model capture
dependencies more effectively, especially given the lack of teacher forcing, by
processing sequences in both forward and backward directions. This bidirectional
setup performed the best in terms of perplexity compared to the other
configurations, but improvements were quite slow due to the 0.0 teacher forcing
ratio.

