import pandas as pd
import jieba
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import networkx as nx
from snownlp import SnowNLP
from gensim import corpora
from utils.cal_ndcg import ranking_precision_score


def process_raw_text():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')

    stopwords = []
    file = open('stopwords/哈工大停用词表.txt', 'rb')
    lines = file.readlines()
    for l in lines:
        l = l.decode('utf-8')
        l = l.strip('\n')
        l = l.strip(' ')
        stopwords.append(l)

    text = df['text']
    text[2989] = ''  # 奇怪的字符
    text_seg = []
    i = 0
    for t in text:
        seg = jieba.cut(t, cut_all=False)
        text_seg.append(list([word for word in seg if word not in stopwords]))

    punc = '[〜~.…^_!•O∩¥¯︶\\|‿≖✧⊙.o]'
    number = '[0-9]'
    chara = '[a-zA-Z]'
    for n in range(len(text_seg)):
        S = ' '.join(text_seg[n])
        S = re.sub(number, ' ', S)
        S = re.sub(punc, ' ', S)
        S = re.sub(chara, ' ', S)
        text_seg[n] = S.split()

    return text_seg


def get_user_label():
    df = pd.read_csv('../dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')

    label = df[['user', 'label']].values.tolist()
    for i in range(len(label)):
        if label[i][1] == '+':
            label[i][1] = 1
            label[i][0] = label[i][0].strip('USER')
        else:
            label[i][1] = 0
            label[i][0] = label[i][0].strip('USER')

    user = list(set([int(i[0]) for i in label]))
    user.sort()

    # get label
    user_label = dict()
    for u in user:
        user_label[u] = []

    for i in range(9765):
        user_label[int(label[i][0])].append(label[i][1])

    for k, v in user_label.items():
        user_label[k] = int(np.round(np.mean(v)))

    y = []
    for u in user:
        y.append(user_label[u])
    y = np.array(y).reshape(9067, 1)
    return y


def get_user_review():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')

    text_seg = process_raw_text()
    for i in range(len(text_seg)):
        text_seg[i] = ' '.join(text_seg[i])

    vectorizer = TfidfVectorizer(min_df=1, max_features=100)
    reviews_vec = vectorizer.fit_transform(text_seg).toarray()

    label = df[['user', 'label']].values.tolist()
    for i in range(len(label)):
        if label[i][1] == '+':
            label[i][1] = 1
            label[i][0] = label[i][0].strip('USER')
        else:
            label[i][1] = 0
            label[i][0] = label[i][0].strip('USER')

    user = list(set([int(i[0]) for i in label]))
    user.sort()

    user_review = dict()
    for u in user:
        user_review[u] = []

    for i in range(9765):
        user_review[int(label[i][0])].append(reviews_vec[i])
    for k, v in user_review.items():
        user_review[k] = np.mean(v, axis=0)

    X = []
    for u in user:
        X.append(user_review[u])
    X = np.array(X)

    return X


def get_user_feature():
    feature = get_user_review()
    feature = np.array(feature)
    np.save('dzdp_user_feature.npy', feature)


def xgboost_classifier():

    y = get_user_label()
    # X = np.load('dzdp_user_feature.npy')
    # adj_data = np.load('dzdp_USU_sim_0.npy')
    # adj_data_1 = np.load('dzdp_UIU_sim_0.npy')
    # adj_data_2 = np.load('dzdp_URU_sim_0.npy')
    # adj_data_3 = np.load('dzdp_UAU_sim_0.npy')
    # adj_data_4 = np.load('dzdp_UIU_URU_sim_0.npy')
    # adj_data_5 = np.load('dzdp_USU_UIU_sim_0.npy')
    # adj_data_6 = np.load('dzdp_USU_URU_sim_0.npy')
    # adj_data_7 = np.load('dzdp_USU_UIU_URU_sim_0.npy')
    # adj_data = np.array([adj_data, adj_data_1, adj_data_2, adj_data_3, adj_data_4, adj_data_5, adj_data_6, adj_data_7])
    X_train, X_test, y_train, y_test = train_test_split(np.load('dzdp_UIU_URU_sim_0.npy'), y, stratify=y, test_size=0.4,
                                                        random_state=48, shuffle=True)
    
    # X_val, X_test, y_val, y_test = train_test_split(X_test, y_test ,stratify=y_test, test_size=0.5,
    #                                                     random_state=0, shuffle=True)

    # X_train = X_test[0:906]
    X_test = X_test[0:1813]
    y_train= y_test[0:453]
    y_test = y_test[0:1813]
    
    xgbc = LogisticRegression()
    xgbc.fit(X_train, y_train)
    score = xgbc.score(X_test, y_test)
    print(score)
    prob = xgbc.predict_proba(X_test)[:,1]
    pred = xgbc.predict(X_test)
    from sklearn.metrics import f1_score, average_precision_score,recall_score,roc_auc_score,roc_curve,auc
    from utils.cal_ndcg import ndcg_score
    print('recall:', recall_score(y_test, pred))
    print('f1:', f1_score(y_test, pred))
    print('AP:', average_precision_score(y_test, prob))

    single_label = [i[0] for i in y_test]
    sorted_r, single_label = zip(*sorted(zip(prob, single_label), reverse=True))

    print('AUC', roc_auc_score(y_test, prob))
    fpr, tpr, thresholds  =  roc_curve(y_test, prob) 
    roc_auc = auc(fpr,tpr)
    print('roc_auc:',roc_auc)
    print(ranking_precision_score(single_label, sorted_r, 100))
    print(ranking_precision_score(single_label, sorted_r, 500))

    print(ndcg_score(single_label,sorted_r,100))
    print(ndcg_score(single_label, sorted_r, 500))

# xgboost_classifier()


def user_rank_G():
    df = pd.read_csv('ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')
    label = df[['user']].values.tolist()
    for i in range(len(label)):
        label[i][0] = label[i][0].strip('USER')
    
    user = list(set([int(i[0]) for i in label]))
    user.sort()
    
    rank = df[['user', 'star']].values.tolist()
    user_rank = dict()
    for u in user:
        user_rank[u] = []
    for i in range(9765):
        user_rank[int(rank[i][0].strip('USER'))].append(rank[i][1])
    
    rank_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}
    for k, v in user_rank.items():
        for i in range(len(v)):
            user_rank[k][i] = rank_map[v[i]]
    
    G = nx.Graph()
    for u in user:
        G.add_node(u)
    G.add_nodes_from([10000, 10001, 10002])
    for k, v in user_rank.items():
        for r in v:
            G.add_edge(k, int(10000 + r))
    
    adj = nx.adjacency_matrix(G).A
    user_time_adj = adj[0:9067, 9067:]

    return user_time_adj


def user_senti_G():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')
    
    text_seg = process_raw_text()
    for i in range(len(text_seg)):
        text_seg[i] = ' '.join(text_seg[i])
    
    senti = []
    for i in range(len(text_seg)):
        if text_seg[i] == '':
            senti.append(0.5)
        else:
            senti.append(SnowNLP(text_seg[i]).sentiments)
    
    for i in range(len(senti)):
        if senti[i] >= 0.8:
            senti[i] = 2
        if senti[i] < 0.8 and senti[i] >= 0.3:
            senti[i] = 1
        if senti[i] < 0.3:
            senti[i] = 0
    
    all_user = df[['user']].values.tolist()
    for i in range(9765):
        all_user[i][0] = all_user[i][0].strip('USER')
    
    user = list(set([int(i[0]) for i in all_user]))
    user.sort()
    
    user_senti = dict()
    for u in user:
        user_senti[u] = []
    for i in range(9765):
        user_senti[int(all_user[i][0])].append(senti[i])
    
    G = nx.Graph()
    for u in user:
        G.add_node(u)
    G.add_nodes_from([10000, 10001, 10002])
    for k, v in user_senti.items():
        for s in v:
            G.add_edge(k, int(s + 10000))
    
    adj = nx.adjacency_matrix(G).A
    user_senti_adj = adj[0:9067, 9067:]

    return user_senti_adj

# check word frequency of spam/not spam
def word_freq():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')
    label = df[['user', 'label']].values.tolist()
    for i in range(len(label)):
        if label[i][1] == '+':
            label[i][1] = 1
            label[i][0] = label[i][0].strip('USER')
        else:
            label[i][1] = 0
            label[i][0] = label[i][0].strip('USER')

    user = list(set([int(i[0]) for i in label]))
    user.sort()

    text_seg = process_raw_text()

    fake_text = []
    not_fake_text = []
    for i in range(9765):
        if label[i][1] == 1:
            fake_text.append(text_seg[i])
        if label[i][1] == 0:
            not_fake_text.append(text_seg[i])

    dictionary = corpora.Dictionary(fake_text)
    all_fake_text = []
    for fk in fake_text:
        for f in fk:
            all_fake_text.append(f)
    corpus = dictionary.doc2bow(all_fake_text)

    fake_word_dict = []
    for i in range(4707):
        fake_word_dict.append([dictionary[i], corpus[i][1]])
    fake_word_dict.sort(key=lambda x: x[1], reverse=True)

    dictionary1 = corpora.Dictionary(not_fake_text)
    all_not_fake_text = []
    for nfk in not_fake_text:
        for f in nfk:
            all_not_fake_text.append(f)
    corpus1 = dictionary1.doc2bow(all_not_fake_text)

    not_fake_word_dict = []
    for i in range(5291):
        not_fake_word_dict.append([dictionary1[i], corpus1[i][1]])
    not_fake_word_dict.sort(key=lambda x: x[1], reverse=True)

    return fake_word_dict, not_fake_word_dict

# a, b = word_freq()

def user_aspect_G():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')
    label = df[['user', 'label']].values.tolist()
    for i in range(len(label)):
        if label[i][1] == '+':
            label[i][1] = 1
            label[i][0] = label[i][0].strip('USER')
        else:
            label[i][1] = 0
            label[i][0] = label[i][0].strip('USER')

    user = list(set([int(i[0]) for i in label]))
    user.sort()

    text_seg = process_raw_text()
    #    aspect_map = {'服务':0,'服务员':0,'服务态度':0,'热情':0,'态度':0,
    #                  '味道':1,'口味':1,'好吃':1,'好喝':1,'美味':1, '难吃':1,
    #                  '环境':2,'干净':2,'价格':3,'小贵':3,'贵':3,'便宜':3,'实惠':3,'划算':3,'速度':4,'团购':5,'性价比':6}
    aspect_map = {'真的': 0}
    user_aspect = dict()
    for u in user:
        user_aspect[u] = []
    for i in range(9765):
        for word in text_seg[i]:
            if word in aspect_map.keys():
                user_aspect[int(label[i][0])].append(aspect_map[word])
    for k, v in user_aspect.items():
        user_aspect[k] = list(set(v))

    G = nx.Graph()
    for u in user:
        G.add_node(u)
    G.add_node(10000)
    #    G.add_nodes_from(range(10000))
    for k, v in user_aspect.items():
        for a in v:
            G.add_edge(k, 10000)
    #            G.add_edge(k,int(a+10000))

    adj = nx.adjacency_matrix(G).A
    user_aspect_adj = adj[0:9067, 9067:]

    return user_aspect_adj


def user_ip_G():
    df = pd.read_csv('dzdp/ICDM_REVIEWS_TO_RELEASE_encoding%3Dutf-8.csv')
    user_ip = df[['user', 'ip']].values.tolist()
    for i in range(len(user_ip)):
        user_ip[i][0] = user_ip[i][0].strip('USER')
        user_ip[i][0] = int(user_ip[i][0])
        user_ip[i][1] = user_ip[i][1].strip('IP')
        user_ip[i][1] = int(user_ip[i][1])

    user = list(set([int(i[0]) for i in user_ip]))
    user.sort()
    ip = list(set([int(i[1]) for i in user_ip]))
    ip.sort()

    G = nx.Graph()
    for u in user:
        G.add_node(u)
    for i in ip:
        G.add_node(i)
    for ui in user_ip:
        G.add_edge(ui[0], int(10000 + ui[1]))

    adj = nx.adjacency_matrix(G).A
    user_ip_adj = adj[0:9067, 9067:]

    return user_ip_adj
