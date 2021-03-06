import re
from collections import Counter

def words(text):
    return re.findall(r'\w+',text.lower())

words_count = Counter(words(open('/content/dataset/big.txt').read()))
checked_word = words(open('/content/dataset/wordlist.txt').read())

def P(word, N=sum(words_count.values())):
    return words_count[word]/N

def correction(word):
    if word.lower() in checked_word:
        new_word = word
    else:
        new_word = max(candidates(word), key=P)
        if word[0].lower()==new_word[0]:
            new_word = list(new_word)
            new_word[0] = word[0]
            new_word = ''.join(new_word)
    return new_word

def correction_list(words):
    res = []
    for word in words:
        if word.lower() in checked_word:
            new_word = word
        else:
            new_word = max(candidates(word),key=P)
            if word[0].lower() == new_word[0]:
                new_word = list(new_word)
                new_word[0] = word[0]
                new_word = ' '.join(new_word)
            res.append(new_word)
        return res
    
def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    return set(w for w in words if w in words_count)

def edits1(word):
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

print(correction('Smell'))
