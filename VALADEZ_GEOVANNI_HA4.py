import nltk
from nltk import bigrams
from collections import defaultdict

#download brown corpus
nltk.download('brown')
from nltk.corpus import brown

# set corpus to the first 1000 words from the brown corpus
corpus = brown.words()[:1000]

#tokenize words from corpus
tokens = nltk.word_tokenize(' '.join(corpus))
bigram_model = list(bigrams(tokens))

# set bigram count 
bigram_count = defaultdict(int)
unigram_count = defaultdict(int)

for bigram in bigram_model:
    bigram_count[bigram] += 1
    unigram_count[bigram[0]] += 1

bigram_probabilities = {}
for bigram, count in bigram_count.items():
    w1, w2 = bigram
    bigram_probabilities[bigram] = count / unigram_count[w1]

#user input
inpW = input("Enter a word: ")

# find possible next bigram words
possible_next_words = {}
for bigram, probability in bigram_probabilities.items():
    w1, w2 = bigram
    if w1 == inpW:
        possible_next_words[w2] = probability

#sort results 
sorted_next_words = sorted(possible_next_words.items(), key=lambda x: x[1], reverse=True)
#output display
if sorted_next_words:
    print("Possible next words:")
    for word, probability in sorted_next_words:
        print(f"{word}: {probability}")
else:
    print("No possible next words found for input word:", inpW)
