

import pandas as pd
import re
import numpy as np
import networkx as nx
import nltk
import warnings
import netrd
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


# -----------------------------------------------------------------------------
# ----- Functions
# -----------------------------------------------------------------------------

def extractEntities(df):
    
    return np.unique(np.concatenate([df['from'].values, df['to'].values]))


def transformToGraph(df):
    """
    Transforms a DataFrame into a weighted graph.

    Args:
        df (pandas.DataFrame): The input DataFrame containing 'from' and 'to' columns.

    Returns:
        networkx.Graph: A weighted graph representation of the input DataFrame.

    """
    # To weighted edgelist
    df = df[['from', 'to']]
    df = df[df['from'] != df['to']]
    df['und_pair'] = df[['from', 'to']].apply(lambda row: ' - '. join(list(set(row))), 1)
    df = df['und_pair'].value_counts().reset_index()
    
    df['from'] = df['index'].apply(lambda x: x.split(' - ')[0])
    df['to'] = df['index'].apply(lambda x: x.split(' - ')[1])
    df['weight'] = df['und_pair']
    df = df[['from', 'to', 'weight']]

    # To weighted graph matrix
    g = nx.from_pandas_edgelist(df,
                                source='from',
                                target='to',
                                edge_attr='weight',
                                create_using=nx.Graph())

    return g 


def transformToAdjVec(g):
    """
    Transforms a graph into an adjacency vector.

    Parameters:
    g (networkx.Graph): The input graph.

    Returns:
    numpy.ndarray: The adjacency vector representation of the graph.
    """

    adj = nx.adjacency_matrix(g).todense()
    adj = adj.astype(float)

    # Account for undirected nature
    adj[np.arange(adj.shape[0])[:,None] > np.arange(adj.shape[1])] = np.nan
    np.fill_diagonal(adj, np.nan)
    
    # ---- flatten to vector
    adj = adj.flatten()
    adj = adj[~np.isnan(adj)]    
    adj = np.array(adj)[0]
    
    adj = nx.adjacency_matrix(g).todense()

    # Order node names alphabetically
    adj = pd.DataFrame(adj.astype(float))
    adj.columns = list(g.nodes)
    adj.index = list(g.nodes)
    ordered_nodes = np.sort(adj.index)
    adj = adj.loc[ordered_nodes, ordered_nodes]
    adj = np.matrix(adj)

    # Account for undirected nature
    adj[np.arange(adj.shape[0])[:,None] > np.arange(adj.shape[1])] = np.nan
    np.fill_diagonal(adj, np.nan)

    # ---- flatten to vector
    vec = adj.flatten()
    vec = vec[~np.isnan(vec)]    
    vec = np.array(vec)[0]
    
    return vec


def pctOverlapSets(setA, setB):

    return round((len(setA & setB) / float(len(setA | setB))), 2)


def selectOverlapingNodes(df, case):
    """
    Selects overlapping nodes from a DataFrame based on a given case.

    Parameters:
    df (DataFrame): The input DataFrame containing the nodes.
    case (str): The case to filter the nodes. If case is 'ASL', no filtering is applied.

    Returns:
    DataFrame: The filtered DataFrame containing only the overlapping nodes.
    """
    if case != 'ASL':
        coder_entities = set(dict_entities_AE[case]['entity'])
             
        df = df[df['from'].apply(lambda x: x in coder_entities)]
        df = df[df['to'].apply(lambda x: x in coder_entities)]
    
    return df


def preprocessInteractionsCoders(case, coder):
    """
    Preprocesses the interactions data for a specific case and coder.

    Args:
        case (str): The path to the case directory.
        coder (str): The name of the coder.

    Returns:
        pandas.DataFrame: The preprocessed interactions data.

    """
    df = pd.read_excel(case + '/edgelist_' + coder + '.xlsx')
    
    if 'time' not in df.columns:
        df['time'] = np.nan
    
    df = df.rename(columns={'source':'from', 'target':'to', 'what':'interaction',
                            'time':'date', 'source location':'paragraph'})
    df = df[['from', 'to', 'interaction', 'date', 'paragraph']]
    df['from'] = df['from'].str.strip()
    df['to'] = df['to'].str.strip()
    df['paragraph'] = df['paragraph'].apply(lambda x: x.split('/')[2][1:] if type(x) == str else np.nan)    
    df['paragraph'] = df['paragraph'].apply(lambda x: int(re.findall('\d+', x)[0]) if type(x) == str else np.nan)
    
    return(df)


def drop_weights(G):
    
    for node, edges in nx.to_dict_of_dicts(G).items():
        for edge, attrs in edges.items():
            attrs.pop('weight', None)


def deltacon(G_1, G_2):
    """
    Calculates the DeltaCon similarity measure between two graphs.

    Parameters:
    - G_1: NetworkX graph
        The first input graph.
    - G_2: NetworkX graph
        The second input graph.

    Returns:
    - deltacon: float
        The DeltaCon similarity measure between G_1 and G_2.
    """
    
    dist_obj = netrd.distance.DeltaCon()
    deltacon = 1 / (1 + dist_obj.dist(G_1, G_2))
    
    return deltacon


def pretiffyTable(df, case):
    """
    Formats and prettifies a given DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to be formatted.
        case (str): The case name.

    Returns:
        pandas.DataFrame: The formatted DataFrame.
    """
    
    # Recode and shuffle the columns and rows
    df = df.applymap(lambda x: '{:.2f}'.format(x))
    # df = round(df, 2)
    df = df.rename(columns={'G3':'GPT-3.5', 'G4':'GPT-4', 'AE':'Coder 1', 'SH':'Coder 2'})
    df.index = df.columns
    df = df.loc[['Coder 1','Coder 2','GPT-3.5','GPT-4'],
                                  ['Coder 1','Coder 2','GPT-3.5','GPT-4']]
    df['Coder'] = df.columns
    df['Case'] = case
    df = df[['Case', 'Coder', 'Coder 1', 'Coder 2', 'GPT-3.5', 'GPT-4']]
    df = df.astype(str).replace('nan', '-')
    
    return df

def sortNodesGraph(g):
    
    g_2 = nx.Graph()
    g_2.add_nodes_from(sorted(g.nodes(data=True)))
    g_2.add_edges_from(g.edges(data=True))
    
    return g_2


def calculateDeltacon(case):
    """
    Calculates the DELTACON values for the given case.

    Parameters:
    - case: The case for which DELTACON values are calculated.

    Returns:
    - df_deltacon: A DataFrame containing the DELTACON values for the given case.
    """

    # Transform to graphs
    g_AE = transformToGraph(dict_interactions_AE[case])
    g_SH = transformToGraph(dict_interactions_SH[case])
    g_G3 = transformToGraph(dict_interactions_G3[case])
    g_G4 = transformToGraph(dict_interactions_G4[case])
    
    # Add missing nodes
    g_G3.add_nodes_from(g_G4)
    g_G3.add_nodes_from(g_AE)
    g_G3.add_nodes_from(g_SH)
    g_G4.add_nodes_from(g_G3)
    g_G4.add_nodes_from(g_AE)
    g_G4.add_nodes_from(g_SH)
    g_SH.add_nodes_from(g_G3)
    g_AE.add_nodes_from(g_G3)
    
    # Sort nodes
    g_AE = sortNodesGraph(g_AE)
    g_SH = sortNodesGraph(g_SH)
    g_G3 = sortNodesGraph(g_G3)
    g_G4 = sortNodesGraph(g_G4)
    
    # Compute DELTACON
    df_deltacon = pd.DataFrame(
        {'G3':[np.nan, deltacon(g_G3, g_G4), deltacon(g_G3, g_AE), deltacon(g_G3, g_SH)],
         'G4':[deltacon(g_G4, g_G3), np.nan, deltacon(g_G4, g_AE), deltacon(g_G4, g_SH)],
         'AE':[deltacon(g_AE, g_G3), deltacon(g_AE, g_G4), np.nan, deltacon(g_AE, g_SH)],
         'SH':[deltacon(g_SH, g_G3), deltacon(g_SH, g_G4), deltacon(g_SH, g_AE), np.nan]})
    
    # Prettify table for the output
    df_deltacon = pretiffyTable(df_deltacon, case)

    return df_deltacon


def calculateTiesCorr(case):
    """
    Calculates the correlation between the adjacency matrices of different graphs.

    Parameters:
    - case: The case identifier.

    Returns:
    - df_corr: The correlation matrix of the adjacency matrices.

    """

    # Transform to graphs
    g_AE = transformToGraph(dict_interactions_AE[case])
    g_SH = transformToGraph(dict_interactions_SH[case])
    g_G3 = transformToGraph(dict_interactions_G3[case])
    g_G4 = transformToGraph(dict_interactions_G4[case])
    
    # Add missing nodes
    g_G3.add_nodes_from(g_G4)
    g_G3.add_nodes_from(g_AE)
    g_G3.add_nodes_from(g_SH)
    g_G4.add_nodes_from(g_G3)
    g_G4.add_nodes_from(g_AE)
    g_G4.add_nodes_from(g_SH)
    g_SH.add_nodes_from(g_G3)
    g_AE.add_nodes_from(g_G3)
    
    # Create dataset with vectorized adajency matrices
    df_vecs = pd.DataFrame()
    df_vecs['G3'] = transformToAdjVec(g_G3)
    df_vecs['G4'] = transformToAdjVec(g_G4)
    df_vecs['AE'] = transformToAdjVec(g_AE)
    df_vecs['SH'] = transformToAdjVec(g_SH)
    
    # Correlation
    df_corr = df_vecs.corr()
    np.fill_diagonal(df_corr.values, np.nan)
    
    # Prettify table for the output
    df_corr = pretiffyTable(df_corr, case)
    
    return df_corr


def calcluateNodesOverlap(case):
    """
    Calculates the overlap between different sets of entities for a given case.

    Parameters:
    - case (str): The case identifier.

    Returns:
    - df_overlap (pandas.DataFrame): A DataFrame containing the overlap percentages between the sets of entities.
    """
   
    set_AE = set(dict_entities_AE[case])
    set_SH = set(dict_entities_SH[case])
    set_G3 = set(dict_entities_G3[case])
    set_G4 = set(dict_entities_G4[case])
    
    df_overlap = pd.DataFrame(
       {'G3':[np.nan, pctOverlapSets(set_G3, set_G4), pctOverlapSets(set_G3, set_AE), pctOverlapSets(set_G3, set_SH)],
        'G4':[pctOverlapSets(set_G4, set_G3), np.nan, pctOverlapSets(set_G4, set_AE), pctOverlapSets(set_G4, set_SH)],
        'AE':[pctOverlapSets(set_AE, set_G3), pctOverlapSets(set_AE, set_G4), np.nan, pctOverlapSets(set_AE, set_SH)],
        'SH':[pctOverlapSets(set_SH, set_G3), pctOverlapSets(set_SH, set_G4), pctOverlapSets(set_SH, set_AE), np.nan]})
    
    # Prettify table for the output
    df_overlap = pretiffyTable(df_overlap, case)
    
    return df_overlap


def countEntitiesInteractions(case):
    """
    Counts the number of entities and interactions for each coder and GPT model for a given case.

    Parameters:
    - case (str): The case identifier.

    Returns:
    - df (pandas.DataFrame): A DataFrame containing the counts of entities and interactions for each coder and GPT model.
    """

    df = pd.DataFrame(
        {'Coder 1':[len(dict_entities_AE[case]), len(dict_interactions_AE[case])],
         'Coder 2':[len(dict_entities_SH[case]), len(dict_interactions_SH[case])],
         'GPT-3.5':[len(dict_entities_G3[case]), len(dict_interactions_G3[case])],
         'GPT-4':[len(dict_entities_G4[case]), len(dict_interactions_G4[case])]})
    df['type'] = ['Entities', 'Interactions']
    df['case'] = case
    df = df[['case', 'type', 'Coder 1', 'Coder 2', 'GPT-3.5', 'GPT-4']]

    return df

# -----------------------------------------------------------------------------
# ----- CONFIG
# -----------------------------------------------------------------------------

WD = '/data'
LIST_CASES = ['Airbus', 'ASL', 'Gurlap', 'Sarclad', 'StdBank']
LIST_MODELS = ['gpt-3.5-turbo', 'gpt-4']

# -----------------------------------------------------------------------------
# ----- LOAD INTERACTIONS AND ENTITIES
# -----------------------------------------------------------------------------

# --- GPT-3.5
dict_interactions_G3 = {
    case:pd.read_pickle(WD + 'data/' + case + '/df_interactions_gpt-3.5-turbo.obj') for case in LIST_CASES
    }

# --- GPT-4
dict_interactions_G4 = {
    case:pd.read_pickle(WD + 'data/' + case + '/df_interactions_gpt-4.obj') for case in LIST_CASES
    }


# --- Coder AE
dict_interactions_AE = {
    case:preprocessInteractionsCoders(WD + 'data/' + case, 'AE') for case in LIST_CASES
    }

# --- Coder SH
dict_interactions_SH = {
    case:preprocessInteractionsCoders(WD + 'data/' + case, 'SH') for case in LIST_CASES
    }

# ------ DICT WITH NAMED ENTITIES
dict_entities_G3 = {case:extractEntities(df) for case, df in dict_interactions_G3.items()}
dict_entities_G4 = {case:extractEntities(df) for case, df in dict_interactions_G4.items()}
dict_entities_AE = {case:extractEntities(df) for case, df in dict_interactions_AE.items()}
dict_entities_SH = {case:extractEntities(df) for case, df in dict_interactions_SH.items()}

# -----------------------------------------------------------------------------
# ----- 1. NUMBER OF NODES AND INTERACTIONS
# -----------------------------------------------------------------------------

list_freq = []
for case in LIST_CASES:
    
    list_freq.append(countEntitiesInteractions(case))

table_freq = pd.concat(list_freq)
table_freq.to_excel(WD + 'output/' + 'table_freq.xlsx', index=False)

# -----------------------------------------------------------------------------
# ----- 2. NODES OVERLAP
# -----------------------------------------------------------------------------

list_overlaps = []
for case in LIST_CASES:
    
    list_overlaps.append(calcluateNodesOverlap(case))
    
table_overlap = pd.concat(list_overlaps)
table_overlap.to_excel(WD + 'output/' + 'table_overlap.xlsx', index=False)


# -----------------------------------------------------------------------------
# ----- 3. CORRELATION OF TIES
# -----------------------------------------------------------------------------

list_corrs = []
for case in LIST_CASES:
    
    list_corrs.append(calculateTiesCorr(case))

table_corr = pd.concat(list_corrs)
table_corr.to_excel(WD + 'output/' + 'table_corr.xlsx', index=False)

# -----------------------------------------------------------------------------
# ----- 4. DELTACON
# -----------------------------------------------------------------------------

list_deltacon = []
for case in LIST_CASES:

    list_deltacon.append(calculateDeltacon(case))

table_deltacon = pd.concat(list_deltacon)
table_deltacon.to_excel(WD + 'output/' + 'table_deltacon.xlsx', index=False)


# -----------------------------------------------------------------------------
# ----- 5. SAVE MATRICIES FOR PLOTS
# -----------------------------------------------------------------------------

case = 'Sarclad'

# Transform to graphs
g_AE = transformToGraph(dict_interactions_AE[case])
g_SH = transformToGraph(dict_interactions_SH[case])
g_G3 = transformToGraph(dict_interactions_G3[case])
g_G4 = transformToGraph(dict_interactions_G4[case])

# Add missing nodes
g_G3.add_nodes_from(g_G4)
g_G3.add_nodes_from(g_AE)
g_G3.add_nodes_from(g_SH)
g_G4.add_nodes_from(g_G3)
g_G4.add_nodes_from(g_AE)
g_G4.add_nodes_from(g_SH)
g_SH.add_nodes_from(g_G3)
g_AE.add_nodes_from(g_G3)

# --- Adj. matricies
adj_G3 = nx.to_pandas_adjacency(g_G3)
adj_G4 = nx.to_pandas_adjacency(g_G4)
adj_AE = nx.to_pandas_adjacency(g_AE)
adj_SH = nx.to_pandas_adjacency(g_SH)

# Order node names alphabetically
ordered_nodes = np.sort(adj_G3.index)
adj_G3 = adj_G3.loc[ordered_nodes, ordered_nodes]
adj_G4 = adj_G4.loc[ordered_nodes, ordered_nodes]
adj_AE = adj_AE.loc[ordered_nodes, ordered_nodes]
adj_SH = adj_SH.loc[ordered_nodes, ordered_nodes]


# Save for plotting
adj_G3.to_csv(WD + 'data/' + 'adj_G3.csv', index=False)
adj_G4.to_csv(WD + 'data/' + 'adj_G4.csv', index=False)
adj_AE.to_csv(WD + 'data/' + 'adj_AE.csv', index=False)
adj_SH.to_csv(WD + 'data/' + 'adj_SH.csv', index=False)
