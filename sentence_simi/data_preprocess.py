import pandas as pd
import numpy as np
import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('/home/apeksha/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
from sklearn.model_selection import train_test_split
data = pd.read_csv("/home/apeksha/Desktop/Sentence_Sim_Data/combined_to_use.csv",sep='\t')
data = pd.DataFrame(data)
#print data[:200]
data = data.sample(frac=1).reset_index(drop=True)
print data[:200]
sentence1 = data['sentence1'].tolist()
sentence1=sentence1[:5000]
sentence2 = data['sentence2'].tolist()
sentence2 = sentence2[:5000]
score = data['similarity_score'].tolist()
score = score[:5000]
words = []

for i in range(len(sentence1)):
    sentence1[i] = sentence1[i].split(" ")
    sentence1[i] = [lmtzr.lemmatize(item) for item in sentence1[i] if item.isalpha()]
    words.extend(sentence1[i])

for i in range(len(sentence2)):
    sentence2[i] = sentence2[i].split(" ")
    sentence2[i] = [lmtzr.lemmatize(item) for item in sentence2[i] if item.isalpha()]
    words.extend(sentence2[i])

words = list(set(words))

sentence=sentence1 + sentence2

vocab = model.vocab.keys()
# vectors = []
# for w in sentence:
#     if w in vocab:
#         vectors.append(model[w])
#     else:
#         print("Word {} not in vocab".format(w))
#         vectors.append([0])

#model = Word2Vec(sentence,min_count=1,sg=1,workers = 4)
#vocab=model.wv.vocab.keys()

id_to_word=dict(enumerate(words))
word_to_id = {k:v for v,k in id_to_word.items()}

max_length = max([len(x) for x in sentence])

for i in range(len(sentence1)):
    sentence1[i] = [word_to_id[x] for x in sentence1[i]]+[len(words)]* (max_length-len(sentence1[i]))
for i in range(len(sentence2)):
    sentence2[i] = [word_to_id[x] for x in sentence2[i]] + [len(words)]* (max_length-len(sentence2[i]))



vec_x = np.zeros([len(words)+1,300])
for i,j in id_to_word.items():
    try:
        vec_x[i]=model[j]
    except:
        vec_x[len(words)] = np.zeros(300)

vec_x[len(words)] = np.zeros(300)
#vec_x_ =tf.constant(vec_x,dtype=tf.float32)

# for i in range(len(sentence1)):
#     sentence1[i] = [vec_x[x] for x in sentence1[i]]
#
# for i in range(len(sentence2)):
#     sentence2[i] = [vec_x[x] for x in sentence2[i]]

train_x1 = sentence1
train_x2 = sentence2

val_x1 = train_x1[4800:]
val_x2 = train_x2[4800:]

train_x1 = train_x1[:4800]
train_x2 = train_x2[:4800]

for i in range(len(score)):
    if score[i] > 3:
        score[i] = 1
    else:
        score[i] = 0

val_y = score[4800:]
train_y = score[:4800]


np.savez('train.npz', ques1=train_x1, ques2=train_x2, label=train_y)
np.savez('valid.npz', ques1=val_x1, ques2=val_x2, label=val_y)
#np.savez('/preprocessed_data/test.npz', ques1=test_sent1, ques2=test_sent2, label=label_test)
np.savez('embed.npz',embed=vec_x)