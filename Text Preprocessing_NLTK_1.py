import nltk
# nltk.download('punkt')
nltk.download('stopwords')

from nltk import tokenize
from string import punctuation
from nltk.corpus import stopwords

raw_txt = "Welcome to the world of Deep Learning for NLP! We're in this together, and we'll learn together. NLP is amazing, and Deep Learning makes it even more fun. Let's learn!"

txt_sents = tokenize.sent_tokenize(raw_txt)
# print(raw_txt)
# print(txt_sents)
# print(len(txt_sents))
# print(txt_sents[1])
for i in range(len(txt_sents)):
    print(txt_sents[i])

txt_words_sentence1 = tokenize.word_tokenize(txt_sents[0])
print(len(txt_words_sentence1))
print(txt_words_sentence1)

txt_words = [tokenize.word_tokenize(i) for i in txt_sents]
print(txt_words)

raw_txt_lower = raw_txt.lower()
print(raw_txt_lower)

print(type(raw_txt))
txt_sents_lower = [sent.lower() for sent in txt_sents]
print(txt_sents_lower)

list_punct = list(punctuation)
# print(list_punct)

txt_word_lower = [tokenize.word_tokenize(i) for i in txt_sents_lower]

def drop_punct(x):
    drop = [i for i in x if i not in list_punct]
    return drop

txt_words_nopunct = [drop_punct(i) for i in txt_words]
print(txt_words_nopunct)

list_stop = stopwords.words('english')
# print(list_stop[:20])
list_final = list_punct + list_stop

def drop_punct_stop(x):
    drop = [i for i in x if i not in list_final ]
    return drop

txt_final = [drop_punct_stop(i) for i in txt_word_lower]
print(txt_final)
