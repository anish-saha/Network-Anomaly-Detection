import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import networkx as nx

network = pd.read_csv("./data/twitter.csv")
fakeIds = pd.read_csv("./data/twitter_fake_ids.csv")

df = pd.DataFrame(columns=["id", "following", "followers", "label"])

ids = set(np.append(network["src"].unique(), network["dst"].unique()))
df["id"] = sorted(ids)

# Sets number of peers following for all users
following_vals = dict(network["src"].value_counts())
def setFollowing(x):
    try:
        return following_vals[x]
    except:
        return 0
df["following"] = df["id"].apply(setFollowing)

# Sets number of followers for all users - note: takes a while
follower_vals = dict(network["dst"].value_counts()) 
def setFollowers(x):
    try:
        return follower_vals[x]
    except:
        return 0
df["followers"] = df["id"].apply(setFollowers)

# Sets labels for fake users (1 if fake, 0 otherwise) - note: takes a while
def setLabels(x):
    try:
        if x in fakeIds["id"].values:
            return 1
        else:
            return 0
    except:
        return 0    
df["label"] = df["id"].apply(setLabels)

df["followers"].value_counts()

df["following"].value_counts()

# FILTERING PROCESS, current decision: 
df = df.loc[(df["followers"] > 0) & (df["following"] > 0)]

len(df)

print("Anomalies in Pruned Network dataset:",len(df.loc[df["label"] == 1].values))
print("Anomalies in FakeIds dataset:",len(fakeIds["id"].values))

df.iloc[:,2:].describe()

pd.plotting.scatter_matrix(df[df.columns[2:]], diagonal='kde')
plt.show()

# Looks like it follows intuitive reasoning; most users have fewer followers than they follow.
plt.scatter(df["following"], df["followers"], s=0.5)
plt.title("Number Following vs. Number Followers")
plt.xlabel("Following")
plt.ylabel("Followers")

# Pruning the dataset makes us lose many anomalies -- following 0 users might be an indicator of a fake user?
# Also reduces dataset size from ~5.4 million to just 75,814 users, which seems weird --> analyze this more?
# Should we even prune users - up for discussion!
df["label"].value_counts()

df.to_csv("./data/filtered_users_labeled.csv")

# Draw the directed graph programatically - Note: takes a REALLY long time
# G = nx.Graph()

# print("Creating list of edges...")
# edgesList = list(zip(network["src"], network["dst"]))
# print("Done.")

# print("Drawing edges...")
# G.add_edges_from(edgesList) # draw all edges
# print("Done.")

# print("Coloring nodes representing fake users...")
# color_map = [ "red" for node in G if node in fakeIds["id"].values ] # color fake users
# print("Done.")

# print("Drawing full graph...")
# nx.draw(G, node_color = color_map,with_labels = True)
# print("Done.")

# plt.savefig("network.png") # save as png
# plt.show() 

df.reset_index(inplace=True)

filtered_ids = df['id'].tolist()
filtered_network = network.loc[(network['src'].isin(filtered_ids)) & (network['dst'].isin(filtered_ids))]
filtered_network.to_csv("./data/twitter_filtered.csv", index=False)
filtered_fakeIds = fakeIds.loc[fakeIds['id'].isin(filtered_ids)]
filtered_fakeIds.to_csv("./data/twitter_labels_filtered.csv", index=False)

filtered_network

filtered_fakeIds
