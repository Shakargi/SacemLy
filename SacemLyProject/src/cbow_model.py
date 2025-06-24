import numpy as np

embedding = 50

def initParams(vocabLength):
    embeddingMatrix = np.random.randn(vocabLength, embedding) - 0.5
    Wout = np.random.randn(embedding, vocabLength) - 0.5
    b = np.zeros(vocabLength)

    return embeddingMatrix, Wout, b

def softmax(array):
    arrayCopy = array.copy().astype(np.float64)
    arrayCopy -= np.max(arrayCopy)
    return np.exp(arrayCopy) / np.sum(np.exp(arrayCopy))

def forward(X, embeddingMatrix, Wout, b):
    embeddingLayer = np.dot(X, embeddingMatrix)
    output = np.dot(embeddingLayer, Wout) + b
    yHat = softmax(output)

    return yHat

def lossFunctionCCE(yHat, y):
    m = len(y)
    loss = -np.sum(y *np.log(yHat + 1e-6)) / m
    return loss


def backprop(yHat, y, X, embeddingMatrix, Wout):
    dZ = yHat - y  # shape: [vocabLength]

    h = np.dot(X, embeddingMatrix)

    dWout = np.outer(h, dZ)
    db = dZ

    dh = np.dot(Wout, dZ)
    dEmbeddingMatrix = np.outer(X, dh)

    return dEmbeddingMatrix, dWout, db

def fit(data, iterations = 3000, alpha = 0.1):
    embeddingMatrix, Wout, b = initParams(data[0][0].shape[0])
    for i in range(iterations):
        for x, y in data:
            yhat = forward(x, embeddingMatrix, Wout, b)

            dembeddingMatrix, dWout, db = backprop(yhat, y, x, embeddingMatrix, Wout)

            embeddingMatrix -= alpha * dembeddingMatrix
            Wout -= alpha * dWout
            b -= alpha * db
        if (i + 1) % 300 == 0:
            print(f"loss iteration {i+1}: {lossFunctionCCE(y, yhat)}")
    return embeddingMatrix, Wout, b






