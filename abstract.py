import math
import random
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import islice
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text, clean_stops):
    symbols = "!\"#$%&()*+-.,/:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        text = text.replace(i, '')
    stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
                  "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
                  "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
                  "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
                  "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him",
                  "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
                  "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
                  "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours"]
    result = word_tokenize(text.lower().rstrip("\r\n"))
    if clean_stops:
        result = [a for a in result if a not in stop_words]
    return result


def get_df(lines):
    DF = {}
    for i in range(len(lines)):
        for w in lines[i]:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
    for i in DF:
        DF[i] = len(DF[i])
    return DF


def get_tf(lines):
    TF = {}
    for i in range(len(lines)):
        TF[i] = Counter(lines[i])
    return TF


def get_tp(lines):
    TP = {}
    for i in range(len(lines)):
        for w in lines[i]:
            try:
                TP[i].add(w)
            except:
                TP[i] = {w}
    return TP


def simple_shared_results(query, lines):
    tp_data = get_tp(lines)
    tp_query = get_tp(query)[0]

    results = {}
    for i in range(len(lines)):
        score = 0
        for token in tp_query:
            if token in tp_data[i]:
                score += 1
        results[i] = score
    sorted_list = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    print("1) SIMPLE: ", sorted_list, "\r\n")
    return list(islice(sorted_list, 10))


def shared_bonus_results(query, lines, df):
    tp_data = get_tp(lines)
    tp_query = get_tp(query)[0]

    results = {}
    for i in range(len(lines)):
        score = 0
        for token in tp_query:
            if token in tp_data[i]:
                score += 1 + 1 / df[token]
        results[i] = score
    sorted_list = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    print("2) BONUS: ", sorted_list, "\r\n")
    return list(islice(sorted_list, 10))


def document_vectorization1(document_tf, words):
    D = np.zeros(len(words))
    for i in range(len(words)):
        D[i] = document_tf[words[i]]
    return D


def document_vectorization2(document_tf, words):
    D = np.zeros(len(words))
    for i in range(len(words)):
        idf = math.log(len(lines) / df[words[i]])
        D[i] = document_tf[words[i]] * idf
    return D


def cosine1_results(query, lines, df):
    query_vector = document_vectorization1(get_tf(query)[0], list(df.keys()))
    tf = get_tf(lines)
    results = {}
    for i in range(len(lines)):
        line_vector = document_vectorization1(tf[i], list(df.keys()))
        results[i] = cosine_similarity([query_vector], [line_vector])[0][0]
    sorted_list = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    print("3) Cosine similarity TF: ", sorted_list, "\r\n")
    return list(islice(sorted_list, 10))


def cosine2_results(query, lines, df):
    query_vector = document_vectorization2(get_tf(query)[0], list(df.keys()))
    tf = get_tf(lines)
    results = {}
    for i in range(len(lines)):
        line_vector = document_vectorization2(tf[i], list(df.keys()))
        results[i] = cosine_similarity([query_vector], [line_vector])[0][0]
    sorted_list = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

    print("4) Cosine similarity TF*IDF: ", sorted_list, "\r\n")
    return list(islice(sorted_list, 10))


"""def documents_vectorization(lines, words):
    D = np.zeros((len(lines), len(words)))
    for i in range(len(lines)):
        for j in range(len(words)):
            D[i][j] = words[j] in lines[i]
        return D"""

if __name__ == '__main__':

    # query = {0: word_tokenize("active learning")}
    query = {0: clean_text(
        "A common bottleneck in deploying supervised learning systems is collecting human-annotated examples. In many domains, annotators form an opinion about the label of an example incrementally --- e.g., each additional word read from a document or each additional minute spent inspecting a video helps inform the annotation. In this paper, we investigate whether we can train learning systems more efficiently by requesting an annotation before inspection is fully complete --- e.g., after reading only 25 words of a document. While doing so may reduce the overall annotation time, it also introduces the risk that the annotator might not be able to provide a label if interrupted too early. We propose an anytime active learning approach that optimizes the annotation time and response rate simultaneously. We conduct user studies on subsets of two document classification datasets and develop simulated annotators that mimic the users. Our simulated experiments show that anytime active learning outperforms several baselines on these two datasets. For example, with an annotation budget of one hour, training a classifier by annotating the first 25 words of each document reduces classification error by 17% over annotating the first 100 words of each document. ", False)}
    file = open("data.txt", "r")
    lines_old = file.readlines()
    lines = lines_old[:]
    file.close()
    for i in range(len(lines)):
        lines[i] = clean_text(lines[i], False)
    df = get_df(lines)

    # SIMPLE SHARED
    """res_1 = simple_shared_results(query, lines)
    for i in res_1:
        print(lines_old[i])
    print("---------------------------------------")"""

    # SHARED + BONUS
    """res_2 = shared_bonus_results(query, lines, df)
    for i in res_2:
        print(lines_old[i])
    print("---------------------------------------")"""

    # COSINE TF
    """res_3 = cosine1_results(query, lines, df)
    for i in res_3:
        print(lines_old[i])
    print("---------------------------------------")"""

    # COSINE TF * IDF
    res_4 = cosine2_results(query, lines, df)
    for i in res_4:
        print(lines_old[i])
    print("---------------------------------------")
