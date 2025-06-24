from preprocessing import *
from importance_model import *
from cbow_model import *
# with open("nameOfFile", "r") as f:
#     text = f.read()
#
# tokens = clean_and_tockenize(text)
# print(tokens)





# text = "The cat sat on the mat and looked at the dog. The dog barked loudly, but the cat did not move. Both animals stayed in the room until dinner time."
# words = clean_and_tockenize(text)
# vocab = buildVocab(words)
# reverseVocab = buildReverseVocab(words)
#
# pairs = generatePairs(words)
# data = prepareData(vocab, pairs)
#
# embeddingMatrix, Wout, b = fit(data)
#
# test_context = ["cat", "sat", "on", "the"]
# test_vectors = [oneHotEncoder(vocab, w) for w in test_context]
# test_input = np.mean(test_vectors, axis=0)
#
# yhat = forward(test_input, embeddingMatrix, Wout, b)
# predicted_index = np.argmax(yhat)
# predicted_word = reverseVocab[predicted_index]
#
# print("Predicted word:", predicted_word)

import numpy as np
import importance_model as im
from preprocessing import clean_and_tockenize


def test_sentencesSplitter():
    text = "Hello world! This is a test. Is it working? Yes, it is."
    sentences = im.sentencesSplitter(text)
    assert len(sentences) == 4, f"Expected 4 sentences, got {len(sentences)}"
    assert sentences[0] == "Hello world!"
    print("sentencesSplitter passed")


def test_buildVocab_and_reverse():
    words = ["hello", "world", "test", "hello"]
    vocab = im.buildVocab(words)
    rev = im.buildReverseVocab(words)
    assert vocab["hello"] == 0 or vocab["hello"] == 1
    assert rev[0] in vocab
    print("buildVocab and buildReverseVocab passed")


def test_generatePairs():
    words = ["the", "quick", "brown", "fox", "jumps"]
    pairs = im.generatePairs(words, windowSize=1)
    expected_context = ["the", "brown"]
    actual_context = pairs[0][0]

    print("Actual context:", actual_context)
    print("Expected context:", expected_context)

    assert set(actual_context) == set(expected_context), f"Expected {expected_context}, got {actual_context}"
    print("generatePairs passed")



def test_oneHotEncoder_and_decoder():
    vocab = {"a": 0, "b": 1, "c": 2}
    vec = im.oneHotEncoder(vocab, "b")
    assert np.argmax(vec) == 1
    word = im.onHotDecoder({0: "a", 1: "b", 2: "c"}, vec)
    assert word == "b"
    print("oneHotEncoder and onHotDecoder passed")


def test_compute_position_score():
    score = im.compute_position_score(2, 5)
    assert 0 <= score <= 1
    print("compute_position_score passed")


def test_compute_length_score():
    score = im.compute_length_score(10, 20)
    assert 0 <= score <= 1
    print("compute_length_score passed")


def test_compute_keyword_density():
    sentenceTokens = ["this", "is", "a", "test", "sentence"]
    keywords = {"test", "sentence"}
    density = im.compute_keyword_density(sentenceTokens, keywords)
    assert density > 0
    print("compute_keyword_density passed")


def test_compute_similarity_to_title():
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    sim = im.compute_similarity_to_title(v1, v2)
    assert np.isclose(sim, 0)
    print("compute_similarity_to_title passed")


def test_Sentences2Vectors_and_extract_features():
    # Setup dummy vocab and embedding
    words = ["this", "is", "a", "test"]
    vocab = im.buildVocab(words)
    embeddingMatrix = np.eye(len(vocab))  # identity matrix for simplicity

    text = "This is a test. This test is simple."
    keywords = {"test", "simple"}
    title = "Test document"

    vectors = im.Sentences2Vectors(im.sentencesSplitter(text), vocab, embeddingMatrix)
    assert len(vectors) == 2

    features = im.extract_features(text, embeddingMatrix, vocab, keywords, title)
    assert len(features) == 2
    for feat in features:
        assert "text" in feat
        assert "vector" in feat
        assert "positionScore" in feat
        assert "lengthScore" in feat
        assert "keywordDensity" in feat
        assert "similarityToTitle" in feat
    print("Sentences2Vectors and extract_features passed")


if __name__ == "__main__":
    test_sentencesSplitter()
    test_buildVocab_and_reverse()
    test_generatePairs()
    test_oneHotEncoder_and_decoder()
    test_compute_position_score()
    test_compute_length_score()
    test_compute_keyword_density()
    test_compute_similarity_to_title()
    test_Sentences2Vectors_and_extract_features()
    print("All tests passed successfully!")
