from preprocessing import clean_and_tockenize

with open("nameOfFile", "r") as f:
    text = f.read()

tokens = clean_and_tockenize(text)
print(tokens)
