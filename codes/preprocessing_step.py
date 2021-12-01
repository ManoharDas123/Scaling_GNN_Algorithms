################# Facebook-page-page Graph######################
edges_path = 'facebook_large/musae_facebook_edges.csv'
edges = pd.read_csv(edges_path)
edges.columns = ['source', 'target']

features_path = 'facebook_large/musae_facebook_features.json'
with open(features_path) as json_data:
    features = json.load(json_data)

max_feature = np.max([v for v_list in features.values() for v in v_list])
features_matrix = np.zeros(shape=(len(list(features.keys())), max_feature + 1))

i = 0
for k, vs in tqdm(features.items()):
    for v in vs:
        features_matrix[i, v] = 1
    i += 1

node_features = pd.DataFrame(features_matrix, index=features.keys())
G = nx.from_pandas_edgelist(edges)
#
# print(edges.sample(frac=1).head(5))
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


# ################# Karate Club Graph############################
# G = nx.karate_club_graph()
# print("Number of nodes:", G.number_of_nodes())
# print("Number of edges:", G.number_of_edges())
# adjacency_list = dict(G.adjacency())


##################Distribution Graph ########################
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

#############################Corelation Matrix####################
# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

#----------------------------------------------------------------
nRowsRead = 1000 # specify 'None' if want to read whole file
# musae_facebook_edges.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/musae_facebook_edges.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'musae_facebook_edges.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

plotCorrelationMatrix(df1, 8)
#------------------------------------------------------------------
nRowsRead = 1000 # specify 'None' if want to read whole file
# musae_facebook_features.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/musae_facebook_features.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'musae_facebook_features.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
plotCorrelationMatrix(df2, 8)

#-----------------------------------------------------------------------
nRowsRead = 1000 # specify 'None' if want to read whole file
# musae_facebook_target.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/musae_facebook_target.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'musae_facebook_target.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
plotPerColumnDistribution(df3, 10, 5)

