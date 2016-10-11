# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 16:48:00 2016

@author: McSim
"""

import json
from gensim import corpora, models
import numpy as np

def save_answers1(c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs):
    with open("cooking_LDA_pa_task1.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [c_salt, c_sugar, c_water, c_mushrooms, c_chicken, c_eggs]]))

with open("recipes.json") as f:
    recipes = json.load(f)
    
texts = [recipe["ingredients"] for recipe in recipes]
dictionary = corpora.Dictionary(texts)   # составляем словарь
corpus = [dictionary.doc2bow(text) for text in texts]  # составляем корпус документов

np.random.seed(76543)
# здесь мой код для построения модели:
#ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=40, passes=5)

# здесь мой код для сохранения модели:
#ldamodel.save("ldamodel_receipts")

# Загрузка ранее сохранённой модели
ldamodel = models.ldamodel.LdaModel.load("ldamodel_receipts")

# здесь мой код для вывода на печать топовых инградиентов и сохраненияи их списка в переменную:
top_ingr_in_themas = ldamodel.show_topics(num_topics=40, num_words=10, formatted=False)

# здесь мой код для решения следующей задачи: 
# Сколько раз ингредиенты "salt", "sugar", "water", "mushrooms", "chicken", "eggs" 
# встретились среди топов-10 тем? При ответе не нужно учитывать составные ингредиенты, например, "hot water".

ingrs_list = ("salt", "sugar", "water", "mushrooms", "chicken", "eggs")
ingrs_freq_list = []
for i in ingrs_list:
    ingrs_freq_list.append(sum([[top_ingr_in_themas[y][1][x][0] for x in range(len(top_ingr_in_themas[0][1]))].count(i) for y in range(len(top_ingr_in_themas)-1)]))

save_answers1(ingrs_freq_list[0], ingrs_freq_list[1], ingrs_freq_list[2], ingrs_freq_list[3], ingrs_freq_list[4], ingrs_freq_list[5])

### Фильтрация словаря
import copy
dictionary2 = copy.deepcopy(dictionary)
dict_size_before = len(dictionary.dfs)
corpus_size_before = 0
for c in corpus:
    corpus_size_before += len(c)

cuted_dict2 = list({k: v for k, v in dictionary2.dfs.items() if v > 4000}.keys())
dictionary2.filter_tokens(cuted_dict2)
corpus2 = [dictionary2.doc2bow(text) for text in texts]
dict_size_after = len(dictionary2.dfs)
corpus_size_after = 0
for c in corpus2:
    corpus_size_after += len(c)

def save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after):
    with open("cooking_LDA_pa_task2.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [dict_size_before, dict_size_after, corpus_size_before, corpus_size_after]]))
        
#save_answers2(dict_size_before, dict_size_after, corpus_size_before, corpus_size_after)

np.random.seed(76543)
# здесь мой код для построения новой модели:
#ldamodel2 = models.ldamodel.LdaModel(corpus2, id2word=dictionary2, num_topics=40, passes=5)
# здесь мой код для сохранения модели:
#ldamodel.save("ldamodel2_receipts")
# Загрузка ранее сохранённой модели
ldamodel2 = models.ldamodel.LdaModel.load("ldamodel2_receipts")


# здесь мой код вычисления когерентности модели:
tuples = ldamodel.top_topics(corpus)
tuples2 = ldamodel.top_topics(corpus2)
#coherence = sum([tuples[x][1] for x in range(len(tuples))])/len(tuples)
#coherence2 = sum([tuples2[x][1] for x in range(len(tuples2))])/len(tuples2)
coherence = np.mean([c[1] for c in tuples])
coherence2 = np.mean([c[1] for c in tuples2])

def save_answers3(coherence, coherence2):
    with open("cooking_LDA_pa_task3.txt", "w") as fout:
        fout.write(" ".join(["%3f"%el for el in [coherence, coherence2]]))

save_answers3(coherence, coherence2)

np.random.seed(76543)
# здесь мой код для построения третьей модели с иными параметрами:
#ldamodel3 = models.ldamodel.LdaModel(corpus2, id2word=dictionary2, num_topics=40, passes=5, alpha=1)
ldamodel3 = models.ldamodel.LdaModel.load("ldamodel3_receipts")

doc_topics2 = ldamodel2.get_document_topics(corpus2, minimum_probability=0.01)
doc_topics3 = ldamodel3.get_document_topics(corpus2, minimum_probability=0.01)
count_model2 = len(doc_topics2)
count_model3 = len(doc_topics3)

def save_answers4(count_model2, count_model3):
    with open("cooking_LDA_pa_task4.txt", "w") as fout:
        fout.write(" ".join([str(el) for el in [count_model2, count_model3]]))

save_answers4(count_model2, count_model3)

