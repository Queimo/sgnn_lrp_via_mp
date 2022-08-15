#%%

import numpy as np
from load_data import load_data
import torch
from modules import GNN
from train_model import train_model
from subgraph_relevance import subgraph_original, subgraph_mp_transcription, subgraph_mp_forward_hook, get_H_transform
from utils import create_ground_truth, get_feat_order_local_best_guess, get_auac_aupc, get_stats
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pandas as pd
from io import StringIO
import pickle as pkl

#%%

graphs, pos_idx, neg_idx = load_data('BA-2motif')

model_dirs = ['gin-2-ba2motif.torch',
              # 'gin-3-ba2motif.torch',
              # 'gin-4-ba2motif.torch',
              # 'gin-5-ba2motif.torch',
              # 'gin-6-ba2motif.torch',
              'gin-7-ba2motif.torch']

g = graphs[44]
S = [0,1,2,3]
alpha = 0.
verbose = False
num_samples = 50
sample_idx = np.random.choice(len(graphs),num_samples,replace=False)

model_times = []

nn = torch.load('models/'+model_dirs[1])


# #%% md
#
# ### GNNExplainer
# from GNN-LRP git repo
#
# #%%

def sigm(z):
    return torch.tanh(0.5*z)*0.5+0.5

def gnnexplainer(g,nn,H0=None,steps=500,lr=0.5,lambd=0.01,verbose=False):
    z = torch.ones(g.get_adj().shape)*g.get_adj()*2
    num_layer = len(nn.blocks) -1
    for i in range(steps):
        z.requires_grad_(True)

        score = nn.forward(g.get_adj(),H0=H0,masks=[sigm(z)]*num_layer)[g.label] # ,sigm(z)
        emp   = -score
        reg   = lambd*((z)**2).sum() # torch.zeros((1,))

        if i in [j**3 for j in range(100)] and verbose: print('%5d %8.3f %8.3f'%(i,emp.item(),reg.item()))

        (emp+reg).backward()

        with torch.no_grad():
            z = (z - lr*z.grad)
        z.grad = None

    return z.data

# #%% md
#
# gnnexplainer -> Scores for every edges
#
# #%%

R = gnnexplainer(g, nn, verbose=True)

def get_fo_gnnexpl(R, alpha=0, mode='extr'):
    node_order = []
    node_set = set(range(R.shape[0]))

    for i in range(R.shape[0]):
        max_node = None
        max_score = -float('inf')
        for node in node_set:
            subgraph = node_order + [node]
            mask = torch.zeros(R.shape)
            mask[subgraph, :] = alpha
            mask[:, subgraph] = alpha
            mask[subgraph, subgraph] = 1
            score = (mask * R).sum()
            if mode == 'prun':
                score = -score
            if score > max_score:
                max_node = node
                max_score = score
        node_order.append(max_node)
        node_set -= {max_node}

    return node_order

#%% md

### Gradient-based heatmap

#%%

def get_fo_gb(nn, g, mode='extr'):
    if g.node_features is not None:
        H0 = g.node_features
    else:
        H0 = torch.ones([g.get_adj().shape[0],1])
    H0.requires_grad_()
    score = nn.forward(g.get_adj(),H0)[g.label]

    score.backward()
    if mode == 'extr':
        node_order = H0.grad.sum(axis=1).abs().flatten().argsort(descending=True)
    else:
        node_order = H0.grad.sum(axis=1).abs().flatten().argsort(descending=False)

    return node_order

#%% md

### CAM & Grad-CAM

#%%

def CAM(nn,A,H0=None,masks=None):

    if masks is None:
        masks = [1]*(len(nn.blocks)-1)
    H0 = nn.ini(A, H0)

    H = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0))

    A = nn.adj(A)

    for l,mask in zip(nn.blocks[1:],masks):
        H = l.forward(H,A,mask=mask)

    # H = H.sum(dim=0) / 20**.5
    return H

#%%

def get_fo_cam(nn, g, mode='extr'):
    if mode == 'extr':
        return CAM(nn, g.get_adj(), g.node_features)[:,g.label].abs().argsort(descending=True)
    else:
        return CAM(nn, g.get_adj(), g.node_features)[:,g.label].abs().argsort(descending=False)

# #%% md
#
# ## AUC etc. experiments
#
# Comment and uncomment to run different experiments.
#
# #%%

verbose = False
test_samples = 200

# BA-2motif
dataset = 'BA-2motif'
graphs, pos_idx, neg_idx = load_data('BA-2motif')
model_dir = "models/gin-3-ba2motif.torch"; num_layer= 3
nn = torch.load(model_dir)

# # MUTAG
# dataset = 'MUTAG'
# graphs, pos_idx, neg_idx = load_data('MUTAG')
# model_dir = "models/gin-3-mutag.torch"; num_layer= 3
# nn = torch.load(model_dir)

# # Graph-SST2
# dataset = 'Graph-SST2'
# graphs, pos_idx, neg_idx = load_data('Graph-SST2')
# model_dir = "models/gcn-3-sst2graph.torch"; num_layer= 3
# nn = torch.load(model_dir)

message = '{}\nModel depth: {}, model: {}, nb of samples: {}\n'.format(model_dir,num_layer, 'gin', test_samples)
print(message)

messages = []
test_sample_idx = [] # quote out when running for the same samples for aupc/auac
alpha_stats = {}
for alpha in tqdm(np.arange(0.0,1.01,0.05)):
    if dataset == 'BA-2motif':
        stats = {}
        stats['GNNExplainer'] = {'acc': [], 'auc': [], 'auac': [], 'aupc': [], 'acs': [], 'pcs': [], 'label': []}
        stats['Gradient-based'] = {'acc': [], 'auc': [], 'auac': [], 'aupc': [], 'acs': [], 'pcs': [], 'label': []}
        stats['CAM'] = {'acc': [], 'auc': [], 'auac': [], 'aupc': [], 'acs': [], 'pcs': [], 'label': []}
    else:
        stats = {}
        stats['GNNExplainer'] = {'auac_pos': [], 'aupc_pos': [], 'auac_neg': [], 'aupc_neg': []}
        stats['Gradient-based'] = {'auac_pos': [], 'aupc_pos': [], 'auac_neg': [], 'aupc_neg': []}
        stats['CAM'] = {'auac_pos': [], 'aupc_pos': [], 'auac_neg': [], 'aupc_neg': []}
    cnt_pos = test_samples / 2
    cnt_neg = test_samples / 2
    start = time.time()
    random_sample = True if test_sample_idx == [] else False
    i = 0
    while cnt_pos > 0 or cnt_neg > 0:
        if random_sample:
            idx = np.random.randint(len(graphs))
            g = graphs[idx]
            if g.nbnodes < 3: continue
            if g.label == 0:
                if cnt_pos == 0: continue
                else: cnt_pos -= 1
            else:
                if cnt_neg == 0: continue
                else: cnt_neg -= 1
            test_sample_idx.append(idx)
        else:
            if i >= len(test_sample_idx): break
            g = graphs[test_sample_idx[i]]
            i += 1

        gr_tr, all_feats = create_ground_truth(g)

        mode = 'prun'
        # mode = 'extr'

        # H, transforms = get_H_transform(g.get_adj(),nn,gammas=None)
        # fo = get_feat_order_local_best_guess(nn, g, alpha, H, transforms, mode='extr')

        for method in ['GNNExplainer', 'Gradient-based', 'CAM']:
            if method == 'GNNExplainer':
                R = gnnexplainer(g, nn, H0=g.node_features, verbose=False)
                fo = get_fo_gnnexpl(R, alpha, mode)
            elif method == 'Gradient-based':
                fo = get_fo_gb(nn, g, mode)
            elif method == 'CAM':
                fo = get_fo_cam(nn, g, mode)

            nb_nodes = g.get_adj().shape[0]
            best_fo = torch.full((nb_nodes,), -1)
            for ii, fs in enumerate(fo):
                best_fo[fs] = nb_nodes - ii
            fo = best_fo

            if mode == 'extr':
                if dataset == 'BA-2motif':
                    acc, auc = get_stats(gr_tr, fo, all_feats)
                else:
                    auac, acs = get_auac_aupc(nn, g, fo, task=mode, use_softmax=False)
                    aupc, pcs = [], []
            else:
                if dataset == 'BA-2motif':
                    acc, auc = [], []
                else:
                    aupc, pcs = get_auac_aupc(nn, g, fo, task=mode, use_softmax=False)
                    auac, acs = [], []

            if dataset == 'BA-2motif':
                stats[method]['acc'].append(acc)
                stats[method]['auc'].append(auc)
            else:
                if g.label == 0:
                    stats[method]['auac_pos'].append(auac)
                    stats[method]['aupc_pos'].append(aupc)
                else:
                    stats[method]['auac_neg'].append(auac)
                    stats[method]['aupc_neg'].append(aupc)
                # stats[method]['acs'].append(acs)
                # stats[method]['pcs'].append(pcs)
            # stats[method]['label'].append(g.label)
    for method in stats.keys():
        for key in stats[method].keys():
            stats[method][key] = np.mean(stats[method][key])
    alpha_stats[alpha] = stats


#%%

# stat_df = pd.DataFrame(columns=['method','auac_pos','auac_neg','alpha'])
stat_df = pd.DataFrame(columns=['method','aupc_pos','aupc_neg','alpha'])

for alpha, stat in zip(alpha_stats.keys(), alpha_stats.values()):
    # print(alpha,stat)
    # df = pd.DataFrame(stat).T[['auac_pos','auac_neg']]
    df = pd.DataFrame(stat).T[['aupc_pos','aupc_neg']]
    df['alpha'] = alpha
    df['method'] = df.index
    stat_df = stat_df.append(df)
stat_df = stat_df.reset_index(drop=True)
stat_df.to_csv('evaluation_results/gnnexpl_etc_aupc_mutag.csv',index=None)

#%%

stat_df = pd.DataFrame(columns=['method','acc','auc','alpha'])
for alpha, stat in zip(alpha_stats.keys(), alpha_stats.values()):
    # print(alpha,stat)
    df = pd.DataFrame(stat).T[['acc','auc']]
    df['alpha'] = alpha
    df['method'] = df.index
    stat_df = stat_df.append(df)
stat_df = stat_df.reset_index(drop=True)
stat_df.to_csv('evaluation_results/gnnexpl_etc_ba2motif.csv',index=None)

#%% md

## Compare runtime

#%%

dataset = 'BA-2motif'
graphs, pos_idx, neg_idx = load_data('BA-2motif')
model_dir = "models/gin-3-ba2motif.torch"; num_layer= 3
nn = torch.load(model_dir)


#%% md

### L dependency

#%%

num_samples = 10
sample_idx = np.random.choice(len(graphs),num_samples,replace=False)

masks = None
alpha = 0.5
time_stats = []

model_dirs = ['gin-2-ba2motif.torch',
              # 'gin-3-ba2motif.torch',
              # 'gin-4-ba2motif.torch',
              # 'gin-5-ba2motif.torch',
              # 'gin-6-ba2motif.torch',
              'gin-7-ba2motif.torch']

# for i in tqdm(range(1, 26)):
for model in tqdm(model_dirs):
    nn = torch.load("models/"+model)
    subgraph = np.arange(5)
    time_ll = []
    for idx in sample_idx:
        time_l = []
        g = graphs[idx]
        H0 = g.node_features

        # naive GNN-LRP
        time_a = time.time()
        score_naive_gnnlrp = subgraph_original(nn, g, S, alpha=alpha, gamma=None, verbose=verbose)
        time_l.append(time.time()-time_a)

        # sGNN-LRP
        time_a = time.time()
        score_sgnnlrp = subgraph_mp_forward_hook(nn, g, subgraph, alpha,)
        time_l.append(time.time()-time_a)

        # GNNExplainer
        time_a = time.time()
        R = gnnexplainer(g, nn, verbose=False)
        node_order = []
        node_set = set(range(R.shape[0]))

        mask = torch.zeros(R.shape)
        mask[subgraph, :] = alpha
        mask[:, subgraph] = alpha
        mask[subgraph, subgraph] = 1
        score_gnnexpl = (mask * R).sum()

        time_l.append(time.time()-time_a)

        # Gradient
        time_a = time.time()
        if g.node_features is not None:
            H0 = g.node_features
        else:
            H0 = torch.ones([g.get_adj().shape[0],1])
        H0.requires_grad_()
        score = nn.forward(g.get_adj(),H0)[g.label]
        score.backward()
        score_grad = H0.grad.sum(axis=1).abs()[subgraph].sum()

        time_l.append(time.time()-time_a)

        # CAM
        time_a = time.time()
        if masks is None:
            masks = [1]*(len(nn.blocks)-1)
        H0 = nn.ini(g.get_adj(), H0)
        H = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0))
        A = nn.adj(g.get_adj())
        for l,mask in zip(nn.blocks[1:],masks):
            H = l.forward(H,A,mask=mask)
        # H = H.sum(dim=0) / 20**.5
        score_cam = H[subgraph, g.label].abs().sum().data

        time_l.append(time.time()-time_a)
        time_ll.append(time_l)
    time_stats.append([int(model.split('-')[1])] + np.array(time_ll).mean(axis=0).tolist())
time_stats_df = pd.DataFrame(time_stats, columns=['model_depth','naive', 'sGNN-LRP','GNNExplainer','Grad','CAM'])


#%%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(3.5,4))
fig.subplots_adjust(hspace=0.05)  # adjust space between axes
ax1.yaxis.set_label_position("left")

plt.rc('legend', fontsize=12)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.set_xlabel(r'$L$')

line1, = ax1.plot(time_stats_df['model_depth'], time_stats_df['naive'], 'm:')
line2, = ax1.plot(time_stats_df['model_depth'], time_stats_df['sGNN-LRP'], 'r-')
line3, = ax1.plot(time_stats_df['model_depth'], time_stats_df['Grad'], 'b--')
line4, = ax1.plot(time_stats_df['model_depth'], time_stats_df['CAM'], 'y-.')
line5, = ax1.plot(time_stats_df['model_depth'], time_stats_df['GNNExplainer'], 'g-+')
ax1.legend(['naive GNN-LRP', 'sGNN-LRP', 'Gradient-based', '(Grad-)CAM', 'GNNExplainer'])

line2.remove()
line3.remove()
line4.remove()
line5.remove()

ax2.plot(time_stats_df['model_depth'], time_stats_df['GNNExplainer'], 'g-+')
ax2.plot(time_stats_df['model_depth'], time_stats_df['naive'], 'm:')

ax2.set_ylabel('Time (s)')
ax3.plot(time_stats_df['model_depth'], time_stats_df['naive'], 'm:')
ax3.plot(time_stats_df['model_depth'], time_stats_df['sGNN-LRP'], 'r-')
ax3.plot(time_stats_df['model_depth'], time_stats_df['Grad'], 'b--')
ax3.plot(time_stats_df['model_depth'], time_stats_df['CAM'], 'y-.')


ax1.spines.bottom.set_visible(False)
ax3.spines.top.set_visible(False)
ax1.xaxis.tick_top()

ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax2.tick_params(labeltop=False)  # don't put tick labels at the top

ax1.set_ylim(2)
ax2.set_ylim(0.004,0.7)  # outliers only
ax3.set_ylim(-0,0.004)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
plt.savefig('imgs/time_consumption_gnnexpl_etc.eps', dpi=600, format='eps', bbox_inches='tight')
plt.show()

#%% md

### |S| dependency

#%%

H0 = g.node_features
masks = None
alpha = 0.5
time_stats = []

model_dirs = ['gin-3-ba2motif.torch']
model = model_dirs[0]
# for i in tqdm(range(1, 26)):
# for model in tqdm(model_dirs):
for _ in range(10):
    time_l = []

    time_a = time.time()

    # sGNN-LRP
    score_sgnnlrp = subgraph_mp_forward_hook(nn, g, subgraph, alpha,)

    time_l.append(time.time()-time_a)
    time_a = time.time()

for s_size in tqdm(range(1,25)):
    subgraph = np.arange(s_size)
    nn = torch.load("models/"+model)
    time_ll = []
    for _ in range(10):
        time_l = []

        # naive GNN-LRP
        time_a = time.time()
        score_naive_gnnlrp = subgraph_original(nn, g, subgraph, alpha=alpha, gamma=None, verbose=verbose)
        time_l.append(time.time()-time_a)

        # sGNN-LRP
        time_a = time.time()
        score_sgnnlrp = subgraph_mp_forward_hook(nn, g, subgraph, alpha,)
        time_l.append(time.time()-time_a)

        # GNNExplainer
        time_a = time.time()
        R = gnnexplainer(g, nn, verbose=False)
        node_order = []
        node_set = set(range(R.shape[0]))

        mask = torch.zeros(R.shape)
        mask[subgraph, :] = alpha
        mask[:, subgraph] = alpha
        mask[subgraph, subgraph] = 1
        score_gnnexpl = (mask * R).sum()

        time_l.append(time.time()-time_a)

        # Gradient
        time_a = time.time()
        if g.node_features is not None:
            H0 = g.node_features
        else:
            H0 = torch.ones([g.get_adj().shape[0],1])
        H0.requires_grad_()
        score = nn.forward(g.get_adj(),H0)[g.label]
        score.backward()
        score_grad = H0.grad.sum(axis=1).abs()[subgraph].sum()

        time_l.append(time.time()-time_a)

        # CAM
        time_a = time.time()
        if masks is None:
            masks = [1]*(len(nn.blocks)-1)
        H0 = nn.ini(g.get_adj(), H0)
        H = nn.blocks[0].forward(H0,torch.eye(H0.shape[0]).unsqueeze(0))
        A = nn.adj(g.get_adj())
        for l,mask in zip(nn.blocks[1:],masks):
            H = l.forward(H,A,mask=mask)
        # H = H.sum(dim=0) / 20**.5
        score_cam = H[subgraph, g.label].abs().sum().data

        time_l.append(time.time()-time_a)
        time_ll.append(time_l)
    time_stats.append([s_size] + np.array(time_ll).mean(axis=0).tolist())
time_stats_df = pd.DataFrame(time_stats, columns=['s_size','naive','sGNN-LRP','GNNExplainer','Grad','CAM'])


#%%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]}, figsize=(3.5,4))
fig.subplots_adjust(hspace=0.05)  # adjust space between axes
ax1.yaxis.set_label_position("right")
ax2.yaxis.set_label_position("right")
ax3.yaxis.set_label_position("right")

plt.rc('legend', fontsize=12)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax3.set_xlabel(r'$|S|$')
ax3.set_xticks(np.arange(1,20))
ax3.set_xticklabels([str(i) if i % 2 != 0 else '' for i in range(1,20)])

line1, = ax1.plot(time_stats_df['s_size'], time_stats_df['naive'], 'm:')
line2, = ax1.plot(time_stats_df['s_size'], time_stats_df['sGNN-LRP'], 'r-')
line3, = ax1.plot(time_stats_df['s_size'], time_stats_df['Grad'], 'b--')
line4, = ax1.plot(time_stats_df['s_size'], time_stats_df['CAM'], 'y-.')
line5, = ax1.plot(time_stats_df['s_size'], time_stats_df['GNNExplainer'], 'g-+')
# ax1.legend(['naive GNN-LRP', 'sGNN-LRP', 'Gradient-based', '(Grad-)CAM', 'GNNExplainer'])

line2.remove()
line3.remove()
line4.remove()
line5.remove()

ax2.plot(time_stats_df['s_size'], time_stats_df['GNNExplainer'], 'g-+')
ax2.plot(time_stats_df['s_size'], time_stats_df['naive'], 'm:')

# ax2.set_ylabel('Time (s)')
ax3.plot(time_stats_df['s_size'], time_stats_df['naive'], 'm:')
ax3.plot(time_stats_df['s_size'], time_stats_df['sGNN-LRP'], 'r-')
ax3.plot(time_stats_df['s_size'], time_stats_df['Grad'], 'b--')
ax3.plot(time_stats_df['s_size'], time_stats_df['CAM'], 'y-.')


ax1.spines.bottom.set_visible(False)
ax3.spines.top.set_visible(False)
ax1.xaxis.tick_top()

ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax2.tick_params(labeltop=False)  # don't put tick labels at the top

ax1.set_ylim(1.5)
ax2.set_ylim(0.004,0.7)  # outliers only
ax3.set_ylim(-0,0.004)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
plt.show()
# plt.savefig('imgs/time_consumption_gnnexpl_etc_S.eps', dpi=600, format='eps', bbox_inches='tight')


#%% md

## auac aupc on MUTAG and Graph-SST2

#%%

fig = plt.figure(figsize=(16,5))


data_dir = 'evaluation_results/mutag_acti_result.txt'; dataset = 'mutag'
data_dir1 = 'evaluation_results/gnnexpl_etc_auac_mutag.csv'; dataset = 'mutag'
stat_df = pd.read_csv(data_dir1)

with open(data_dir,'r') as f:
    s = f.readlines()[2:]
stats = {}
for i in range(len(s)//2):
    alpha = float(s[i * 2].split(',')[0].split(' ')[-1])
    for sss in s[i * 2 + 1].split('\n')[0].split('\t')[1:]:
        sss = sss.split(':')
        if sss[0].strip() not in stats:
            stats[sss[0].strip()] = []
        stats[sss[0].strip()].append(float(sss[1]))

df = pd.DataFrame()
for key in stats.keys():
    df[key] = stats[key]

df['alpha'] = np.arange(0,1.01,0.05)
df['method'] = 'sGNN-LRP'

stat_df = stat_df.append(df)

plt.subplot(241)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['auac_pos'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['auac_pos'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['auac_pos'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['auac_pos'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['auac_pos'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['auac_pos'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['auac_pos'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['auac_pos'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['auac_pos'].to_numpy()[opt_val_idx], 'g^')

plt.xticks([])
plt.title('MUTAG_positive')
plt.ylabel('AUAC')
plt.legend(['sGNN-LRP', 'Gradient-based', '(Grad-)CAM', 'GNNExplainer'])

plt.subplot(242)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['auac_neg'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['auac_neg'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['auac_neg'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['auac_neg'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['auac_neg'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['auac_neg'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['auac_neg'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['auac_neg'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['auac_neg'].to_numpy()[opt_val_idx], 'g^')

plt.xticks([])
plt.title('MUTAG_negative')

#########################################################################
data_dir = 'evaluation_results/graphsst2_acti_result.txt'; dataset = 'graphsst2'
data_dir1 = 'evaluation_results/gnnexpl_etc_auac_sst2.csv'; dataset = 'graphsst2'
stat_df = pd.read_csv(data_dir1)

with open(data_dir,'r') as f:
    s = f.readlines()[2:]
stats = {}
for i in range(len(s)//2):
    alpha = float(s[i * 2].split(',')[0].split(' ')[-1])
    for sss in s[i * 2 + 1].split('\n')[0].split('\t')[1:]:
        sss = sss.split(':')
        if sss[0].strip() not in stats:
            stats[sss[0].strip()] = []
        stats[sss[0].strip()].append(float(sss[1]))

df = pd.DataFrame()
for key in stats.keys():
    df[key] = stats[key]

df['alpha'] = np.arange(0,1.01,0.05)
df['method'] = 'sGNN-LRP'

stat_df = stat_df.append(df)

plt.subplot(243)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['auac_pos'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['auac_pos'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['auac_pos'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['auac_pos'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['auac_pos'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['auac_pos'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['auac_pos'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['auac_pos'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['auac_pos'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['auac_pos'].to_numpy()[opt_val_idx], 'g^')

plt.xticks([])
plt.title('Graph-SST2_positive')

plt.subplot(244)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['auac_neg'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['auac_neg'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['auac_neg'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['auac_neg'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['auac_neg'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['auac_neg'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['auac_neg'].to_numpy().argsort()[-1]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['auac_neg'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['auac_neg'].to_numpy().argsort()[-1]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['auac_neg'].to_numpy()[opt_val_idx], 'g^')

plt.xticks([])
plt.title('Graph-SST2_negative')

#########################################################################
data_dir = 'evaluation_results/mutag_prun_result.txt'; dataset = 'mutag'
data_dir1 = 'evaluation_results/gnnexpl_etc_aupc_mutag.csv'; dataset = 'mutag'
stat_df = pd.read_csv(data_dir1)

with open(data_dir,'r') as f:
    s = f.readlines()[2:]
stats = {}
for i in range(len(s)//2):
    alpha = float(s[i * 2].split(',')[0].split(' ')[-1])
    for sss in s[i * 2 + 1].split('\n')[0].split('\t')[1:]:
        sss = sss.split(':')
        if sss[0].strip() not in stats:
            stats[sss[0].strip()] = []
        stats[sss[0].strip()].append(float(sss[1]))

df = pd.DataFrame()
for key in stats.keys():
    df[key] = stats[key]

df['alpha'] = np.arange(0,1.01,0.05)
df['method'] = 'sGNN-LRP'

stat_df = stat_df.append(df)

plt.subplot(245)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['aupc_pos'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['aupc_pos'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['aupc_pos'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['aupc_pos'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['aupc_pos'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['aupc_pos'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'].to_numpy()[opt_val_idx], 'g^')

plt.xlabel(r'$\alpha$')
plt.ylabel('AUPC')
plt.title('MUTAG_positive')

plt.subplot(246)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['aupc_neg'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['aupc_neg'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['aupc_neg'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['aupc_neg'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['aupc_neg'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['aupc_neg'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'].to_numpy()[opt_val_idx], 'g^')

plt.xlabel(r'$\alpha$')
plt.title('MUTAG_negative')


#########################################################################
data_dir = 'evaluation_results/graphsst2_prun_result.txt'; dataset = 'graphsst2'
data_dir1 = 'evaluation_results/gnnexpl_etc_aupc_sst2.csv'; dataset = 'graphsst2'
stat_df = pd.read_csv(data_dir1)

with open(data_dir,'r') as f:
    s = f.readlines()[2:]
stats = {}
for i in range(len(s)//2):
    alpha = float(s[i * 2].split(',')[0].split(' ')[-1])
    for sss in s[i * 2 + 1].split('\n')[0].split('\t')[1:]:
        sss = sss.split(':')
        if sss[0].strip() not in stats:
            stats[sss[0].strip()] = []
        stats[sss[0].strip()].append(float(sss[1]))

df = pd.DataFrame()
for key in stats.keys():
    df[key] = stats[key]

df['alpha'] = np.arange(0,1.01,0.05)
df['method'] = 'sGNN-LRP'

stat_df = stat_df.append(df)

plt.subplot(247)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['aupc_pos'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['aupc_pos'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['aupc_pos'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['aupc_pos'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['aupc_pos'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['aupc_pos'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['aupc_pos'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['aupc_pos'].to_numpy()[opt_val_idx], 'g^')

plt.xlabel(r'$\alpha$')
plt.title('Graph-SST2_positive')

plt.subplot(248)
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'],stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'], 'r-')
plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'],stat_df[stat_df['method']=='Gradient-based']['aupc_neg'], 'b--')
plt.plot(stat_df[stat_df['method']=='CAM']['alpha'],stat_df[stat_df['method']=='CAM']['aupc_neg'], 'y-.')
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'],stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'], 'g-+')

opt_val_idx = stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='sGNN-LRP']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='sGNN-LRP']['aupc_neg'].to_numpy()[opt_val_idx], 'r^')
# opt_val_idx = stat_df[stat_df['method']=='Gradient-based']['aupc_neg'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='Gradient-based']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='Gradient-based']['aupc_neg'].to_numpy()[opt_val_idx], 'b^')
# opt_val_idx = stat_df[stat_df['method']=='CAM']['aupc_neg'].to_numpy().argsort()[0]
# plt.plot(stat_df[stat_df['method']=='CAM']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='CAM']['aupc_neg'].to_numpy()[opt_val_idx], 'y^')
opt_val_idx = stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'].to_numpy().argsort()[0]
plt.plot(stat_df[stat_df['method']=='GNNExplainer']['alpha'].to_numpy()[opt_val_idx],stat_df[stat_df['method']=='GNNExplainer']['aupc_neg'].to_numpy()[opt_val_idx], 'g^')

plt.xlabel(r'$\alpha$')
plt.title('Graph-SST2_negative')
plt.savefig('imgs/gnnexpl_etc_aupc_auac.eps', dpi=600, format='eps', bbox_inches='tight')
plt.show()
