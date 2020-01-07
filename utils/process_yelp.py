import numpy as np
import networkx as nx
import yaml

yelp_hotel = []
file = open('D:/STUDY/UIC/code/AttributedGCN_final/AGCN/yelp/output_meta_yelpHotelData_NRYRcleaned.txt')
lines = file.readlines()
for l in lines:
    l = l.split(' ')
    yelp_hotel.append(l)

user_product = [[i[2], i[3]] for i in yelp_hotel]

product = [i[3] for i in yelp_hotel]
product_unique = list(set(product))
product_unique.sort(reverse=False)
product_id_dict = dict()
i = 0
for u in product_unique:
    product_id_dict[u] = i
    i += 1

user = [i[2] for i in yelp_hotel]
user_unique = list(set(user))
user_unique.sort(reverse=False)
user_id_dict = dict()
i = 0
for u in user_unique:
    user_id_dict[u] = i
    i += 1

# build product_user_dict, key is product, value is user
product_user_dict = dict()
for product in product_unique:
    product_user_dict[product] = []
for up in user_product:
    if up[1] in product_user_dict.keys():
        product_user_dict[up[1]].append(up[0])

# review id
review_id_list = []
for i in range(len(yelp_hotel)):
    review_id_list.append(i)

# relation of user and review
user_review_dict = dict()
for user in user_unique:
    user_review_dict[user] = []
r_id = 0
for up in user_product:
    user_review_dict[up[0]].append(r_id)
    r_id += 1
user_review_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/yelp_data/user_review_adjlist.yaml"
for k, v in user_review_dict.items():
    user_review_adjlist.append(v)
#with open(path, "w", encoding="utf-8") as f:
#    yaml.dump(user_review_adjlist, f)

user_review_G = nx.Graph()
for r in review_id_list:
    user_review_G.add_node(r, node_type='review')
for u in user_unique:
    user_review_G.add_node(u, node_type='user')
for k, v in user_review_dict.items():
    for r in v:
        user_review_G.add_edge(k, r)
user_review_adj = nx.adjacency_matrix(user_review_G).A
user_review_adj = user_review_adj[5854:, 0:5854]

# relation of product and review
product_review_dict = dict()
for product in product_unique:
    product_review_dict[product] = []
r_id = 0
for up in user_product:
    product_review_dict[up[1]].append(r_id)
    r_id += 1
product_review_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/yelp_data/item_review_adjlist.yaml"
for k, v in product_review_dict.items():
    product_review_adjlist.append(v)
#with open(path, "w", encoding="utf-8") as f:
#    yaml.dump(product_review_adjlist, f)

product_review_G = nx.Graph()
for r in review_id_list:
    product_review_G.add_node(r, node_type='review')
for u in product_unique:
    product_review_G.add_node(u, node_type='product')
for k, v in product_review_dict.items():
    for r in v:
        product_review_G.add_edge(k, r)
product_review_adj = nx.adjacency_matrix(product_review_G).A
product_review_adj = product_review_adj[5854:, 0:5854]

#np.savetxt('D:/STUDY/UIC/code/DGFraud/yelp_data/user_review_adj.txt', np.array(user_review_adj), fmt='%d')
#np.savetxt('D:/STUDY/UIC/code/DGFraud/yelp_data/product_review_adj.txt', np.array(product_review_adj), fmt='%d')

# label
label = [i[4] for i in yelp_hotel]
binary_label = []
for l in label:
    if l == 'N':
        binary_label.append(int(0))
    if l == 'Y':
        binary_label.append(int(1))

binary_label = np.array(binary_label)

# review-user
ru_G = nx.Graph()
review_user_dict = dict()
for r in review_id_list:
    review_user_dict[r] = user_product[r][0]
ru_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/yelp_data/review_user_adjlist.yaml"
for k, v in review_user_dict.items():
    ru_adjlist.append(user_id_dict[v])
#with open(path, "w", encoding="utf-8") as f:
#    yaml.dump(ru_adjlist, f)

# review-item
review_item_dict = dict()
for r in review_id_list:
    review_item_dict[r] = user_product[r][1]
ri_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/yelp_data/review_item_adjlist.yaml"
for k, v in review_item_dict.items():
    ri_adjlist.append(product_id_dict[v])
#with open(path, "w", encoding="utf-8") as f:
#    yaml.dump(ri_adjlist, f)
 
# user-review-item
user_review_item_dict = dict()
for k, v in user_review_dict.items():
    user_review_item_dict[k] = []
for k, v in user_review_dict.items():
    user_review_item_dict[k] = [product_id_dict[review_item_dict[item]] for item in v]
uri_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/data/yelp_data/user_review_item_adjlist.yaml"
for k, v in user_review_item_dict.items():
    uri_adjlist.append(v)
with open(path, "w", encoding="utf-8") as f:
    yaml.dump(uri_adjlist, f)
    
# item=review-user
item_review_user_dict = dict()
for k, v in product_review_dict.items():
    item_review_user_dict[k] = []
for k, v in product_review_dict.items():
    item_review_user_dict[k] = [user_id_dict[review_user_dict[user]] for user in v]
iru_adjlist = []
path = "D:/STUDY/UIC/code/DGFraud/data/yelp_data/item_review_user_adjlist.yaml"
for k, v in item_review_user_dict.items():
    iru_adjlist.append(v)
with open(path, "w", encoding="utf-8") as f:
    yaml.dump(iru_adjlist, f)
