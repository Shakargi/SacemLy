from preprocessing import *
from importance_model import *
from cbow_model import *
# with open("nameOfFile", "r") as f:
#     text = f.read()
#
# tokens = clean_and_tockenize(text)
# print(tokens)

text = "The cat sat on the mat and looked at the dog. The dog barked loudly, but the cat did not move. Both animals stayed in the room until dinner time."
words = clean_and_tockenize(text)
vocab = buildVocab(words)
reverseVocab = buildReverseVocab(words)

pairs = generatePairs(words)
data = prepareData(vocab, pairs)

embeddingMatrix, Wout, b = fit(data)

test_context = ["cat", "sat", "on", "the"]
test_vectors = [oneHotEncoder(vocab, w) for w in test_context]
test_input = np.mean(test_vectors, axis=0)

yhat = forward(test_input, embeddingMatrix, Wout, b)
predicted_index = np.argmax(yhat)
predicted_word = reverseVocab[predicted_index]

print("Predicted word:", predicted_word)
