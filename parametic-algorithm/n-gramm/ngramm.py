def generate_ngrams(text: str, n: int = 1):
    words = text.split()
    ngrams = []

    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.append(ngram)

    return ngrams

if __name__ == '__main__':
    text = input('Insert Text: ')
    p = generate_ngrams(text)
    print(p)
