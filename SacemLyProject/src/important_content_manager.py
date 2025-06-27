import json
import os


def addSentence(topic: str, sentence: str):
    path = "importantSentences.json"

    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    if topic not in data:
        data[topic] = [sentence]
    else:
        if sentence not in data[topic]:
            data[topic].append(sentence)

    with open(path, "w") as file:
        json.dump(data, file, indent=4)


def getSentences(topic: str):
    path = "importantSentences.json"

    # Load existing data
    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    return data.get(topic, [])


def getAllTopics():
    path = "importantSentences.json"

    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    return list(data.items())


def deleteSentence(topic: str, sentence: str):
    path = "importantSentences.json"

    if os.path.exists(path):
        with open(path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    if topic in data and sentence in data[topic]:
        data[topic].remove(sentence)

        if not data[topic]:
            del data[topic]

        with open(path, "w") as file:
            json.dump(data, file, indent=4)


def loadData():
    path = "importantSentences.json"
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    else:
        return {}


def saveData(data):
    path = "importantSentences.json"
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

def displayAll():
    data = loadData()
    for topic, sentences in data.items():
        print(f"Topic: {topic}")
        for s in sentences:
            print(f"  - {s}")


