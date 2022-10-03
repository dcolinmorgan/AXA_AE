import matplotlib.pyplot as plt, numpy as np, os,glob,sys,importlib,pickle,tqdm, networkx as nx,pandas as pd,scipy,seaborn as sns,typing
from scipy import stats
from geopy.geocoders import Nominatim
import geopy.distance
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# df=pd.read_csv('run/AXA_AE/tmp3.txt',sep='\t')#,index_col=None,header=None)
# df.columns=['id','hosp','AE_id','sex','age','idk','reg','reg_name','date','loc','icd9_a','icd9_b','loc2']
# df.to_parquet('run/AXA_AE/AE_AXA_all.paqrquet', compression='GZIP')
# import pyarrow.csv as pv
# import pyarrow.parquet as pq
# filename='run/AXA_AE/tmp3.txt'
# table = pv.read_csv(filename)
# pq.write_table(table, filename.replace('txt', 'parquet'))
import pyreadr

data=pd.DataFrame()
files=glob.glob('/home/edmondyip/AnE_data/data/AE_attendance_csv/*.rds')
for file in files:
    cc = pyreadr.read_r(file)
    data=data.append(cc[None].iloc[:,np.r_[0:2,3:6,8,13,40,50,60,65]])
# data.columns=['id','hosp','sex','age','idk','reg','reg_name','date','loc','icd9_a','icd9_b','loc2']
data.columns=['s' + str(x) for x in range(0,data.shape[1])]

import pyarrow as pa
import pyarrow.parquet as pq
# diag2=data.columns[data.shape[1]-1]
# diag1=data.columns[data.shape[1]-2]

data.rename(columns={data.columns[data.shape[1]-3]: "diag1", data.columns[data.shape[1]-2]: "diag2"},inplace=True)
data.diag1, data.diag2 = np.where(data.diag1.str.contains('None'), [data.diag2, data.diag1], [data.diag1, data.diag2])
# data['s5']=data['s5'].astype('str')
# data['s6']=data['s6'].astype('str')


# list_a = ['asthma']#, 'COPD','asthma','respi','lung','pulm','oxy','air ']
# # dataA=data.loc[data['diag1'].dropna().index]
# dataA=data[~data['diag1'].isna()]
# data2=dataA[dataA['diag1'].str.contains("(" + "|".join(list_a) + ")",case=False)]
# table = pa.Table.from_pandas(data2.astype(str))
# pq.write_table(table, 'run/AXA_AE/AE_AXA_asthma.parquet')

# d2=data2.groupby(by=['s1','s5']).count()
# table = pa.Table.from_pandas(d2.astype(str))
# pq.write_table(table, 'run/AXA_AE/AE_AXA_asthma_count.parquet')

# data3=data[~data['diag1'].str.contains("(" + "|".join(list_a) + ")",case=False)]
data3=data[(data['diag1'].isna())&(data['diag2'].isna())]
d3=data3.groupby(by=['s1','s5']).count()
table = pa.Table.from_pandas(d3.astype(str))
pq.write_table(table, 'run/AXA_AE/AE_AXA_all_NAcount.parquet')

list_a = ['pneumonia', 'COPD','asthma','respi','lung','pulm','oxy','air ']
# dataA=data.loc[data['diag1'].dropna().index]
dataA=data[~data['diag1'].isna()]
data22=dataA[dataA['diag1'].str.contains("(" + "|".join(list_a) + ")",case=False)]

table = pa.Table.from_pandas(data22.astype(str))
pq.write_table(table, 'run/AXA_AE/AE_AXA_all_LUNG.parquet')
table = pa.Table.from_pandas(data3.astype(str))
pq.write_table(table, 'run/AXA_AE/AE_AXA_all_NOTlung.parquet')

# data.to_parquet('run/AXA_AE/AE_AXA_all.parquet', compression='GZIP')

# os.chdir('/home/dcmorgan')
# os.getcwd()
# ##load and prepare data
# # file='run/AXA_AE/AE_data.tsv'#files[5]
# df=pd.read_parquet('run/AXA_AE/AE_AXA_dat_full.parquet')
# list_a = ['pneumonia', 'COPD','asthma','resp','lung','pulm']#,'~Cancer']
# list_b = ['Cancer']
# df.columns=['pat_id','cd9_loc','sess','sex','age','cd9_code','mini_loc','loc1','date','tmp','diag1','diag2','tmp']

# df2=df[df['diag1'].isin(list_a)]
# df2 = df[df['diag1'].str.contains('|'.join(list_a))]

# df.diag1, df.diag2 = np.where(df.diag1.str.contains('None'), [df.diag2, df.diag1], [df.diag1, df.diag2])
# del df['sess'], df['tmp'], df['diag2']
# df=df[~df['diag1'].isna()]

# df2 = df[df['diag1'].str.contains('|'.join(list_a))]

# df2['cd9_loc'].replace({'RH':'Ruttonjee Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'PYN':'Pamela Youde Nethersole Eastern Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'QEH':'Queen Elizabeth Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'CMC':'Caritas Medical Centre'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'KWH':'Kwong Wah Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'TMH':'Tuen Mun Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'PWH':'Prince of Wales Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'NDH':'North District Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'YCH':'Yan Chai Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'UCH':'United Christian Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'QMH':'Queen Mary Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'PWH':'Princess Margaret Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'POH':'Pok Oi Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'TKO':'Tseung Kwan O Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'AHN':'Alice Ho Miu Ling Nethersole Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'SJH':'St. John Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'NLT':'North Lantau Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'TSH':'Tang Shiu Kin Hospital'},regex=True,inplace=True)
# df2['cd9_loc'].replace({'PMH':'Princess Margaret Hospital'},regex=True,inplace=True)

# #organize
# cc=pd.DataFrame()#(columns=['date','pm25','pm10','o3','no2','so2','co','loc'])
# files=glob.glob('aqi-stations-scraper/data/japan-aqi/*')
# for file in files:
#     data=pd.read_csv(file,sep=' |,')
#     data['loc1']=os.path.basename(file).split(',')[0]
#     cc=cc.append(data)

# data2=cc[['date','pm25','pm10','o3','no2','so2','co','loc1']]
# data2['loc1']=data2['loc1'].str.upper().replace({'-':' '},regex=True)
# data2['date']=pd.to_datetime(data2['date'])

# geolocator = Nominatim(user_agent="example app")
# df_loc=pd.DataFrame(columns=['lat','long','name'])
# for ii,i in enumerate(pd.unique(df2['cd9_loc'])):
#     a,b,c=geolocator.geocode(str(i)+", Hong Kong").point
#     df_loc[ii]=[a,b,i]
# df_loc=df_loc.transpose()
# df_loc.columns=['lat','long','name']
# df_loc=df_loc[3:]


# data2['loc1'].replace('centralnaya-str','central',inplace=True)
# data2['loc1'].replace('southern','southern island',inplace=True)
# data2['loc1'].replace('southern-part of chengyang district','chengyang district',inplace=True)

# data_loc=pd.DataFrame(columns=['lat','long','name'])
# for ii,i in enumerate(pd.unique(data2['loc1'])):
#     try:
#         a,b,c=geolocator.geocode(str(i)+", Hong Kong").point
#     except AttributeError:
#         print('no location data')
#     data_loc[ii]=[a,b,i]
# data_loc=data_loc.transpose()
# data_loc.columns=['lat','long','name']
# data_loc=data_loc[3:]

# data_loc=data_loc[~data_loc.duplicated(['lat','long'],keep='first')]
# data_loc.reset_index(inplace=True)

# data_loc=df_loc.append(data_loc)[['lat','long','name']]
# 2
# data_loc.reset_index(inplace=True)


# # geopy DOES use latlon configuration
# data_loc['latlon'] = list(zip(data_loc['lat'], data_loc['long']))
# square = pd.DataFrame(
#     np.zeros((data_loc.shape[0], data_loc.shape[0])),
#     index=data_loc.index, columns=data_loc.index
# )

# # replacing distance.vicenty with distance.distance
# def get_distance(col):
#     end = data_loc.loc[col.name, 'latlon']
#     return data_loc['latlon'].apply(geopy.distance.distance,
#                               args=(end,),
#                               ellipsoid='WGS-84'
#                              )

# distances = square.apply(get_distance, axis=1).T

# data_loc['src']=data_loc['name']
# data_loc['dst']=data_loc['name']

# # np.sum((distances<5)*1)
# D_D=pd.DataFrame((distances<5)*1)
# D_D.index=data_loc['src']
# D_D.columns=data_loc['dst']

# E_E=pd.DataFrame(D_D.stack())#.reset_index(inplace=True)
# # E_E.rename=['source','target']#.reset_index(inplace=True)#.rename(columns={'level_0':'Source','level_1':'Target', 0:'Weight'})
# E_E.reset_index(inplace=True)#
# distance_mat=E_E[E_E[0]>0]

# distance=distances
# distance.index=data_loc['src']
# distance.columns=data_loc['dst']
# distance=pd.DataFrame(distance.stack())
# distance.reset_index(inplace=True)

# #prepare for TF

# distances=distances.astype(str) # df.astype(np.float64)#lues.as_int#('int')#.to_numpy()
# distances=distances.replace('km', '', regex=True)
# distances=distances.astype(np.float64)
# distances.to_numpy()

# plt.figure(figsize=(8, 8))
# plt.matshow(np.corrcoef(distances.T), 0)
# plt.xlabel("region")
# plt.ylabel("region")

# ##split and normalize

# train_size, val_size = 0.5, 0.2


# def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
#     """Splits data into train/val/test sets and normalizes the data.

#     Args:
#         data_array: ndarray of shape `(num_time_steps, num_routes)`
#         train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
#             to include in the train split.
#         val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
#             to include in the validation split.

#     Returns:
#         `train_array`, `val_array`, `test_array`
#     """

#     num_time_steps = data_array.shape[0]
#     num_train, num_val = (
#         int(num_time_steps * train_size),
#         int(num_time_steps * val_size),
#     )
#     train_array = data_array[:num_train]
#     mean, std = train_array.mean(axis=0), train_array.std(axis=0)

#     train_array = (train_array - mean) / std
#     val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
#     test_array = (data_array[(num_train + num_val) :] - mean) / std

#     return train_array, val_array, test_array

# data3=data2[['pm25','pm10','o3','no2','so2','co']].dropna(how='any')#.to_numpy()
# n = 5
# data3=pd.concat([data3] * (n+1), axis=1, ignore_index=True)

# train_array, val_array, test_array = preprocess(data3, train_size, val_size) #data2[['pm25','pm10','o3','no2','so2','co']].dropna(how='any').to_numpy()


# print(f"train set size: {train_array.shape}")
# print(f"validation set size: {val_array.shape}")
# print(f"test set size: {test_array.shape}")

# #TF dataset
# from tensorflow.keras.preprocessing import timeseries_dataset_from_array

# batch_size = 64
# input_sequence_length = 12
# forecast_horizon = 3
# multi_horizon = False


# def create_tf_dataset(
#     data_array: np.ndarray,
#     input_sequence_length: int,
#     forecast_horizon: int,
#     batch_size: int = 128,
#     shuffle=True,
#     multi_horizon=True,
# ):
#     """Creates tensorflow dataset from numpy array.

#     This function creates a dataset where each element is a tuple `(inputs, targets)`.
#     `inputs` is a Tensor
#     of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
#     the `input_sequence_length` past values of the timeseries for each node.
#     `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
#     containing the `forecast_horizon`
#     future values of the timeseries for each node.

#     Args:
#         data_array: np.ndarray with shape `(num_time_steps, num_routes)`
#         input_sequence_length: Length of the input sequence (in number of timesteps).
#         forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
#             `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
#             timeseries `forecast_horizon` steps ahead (only one value).
#         batch_size: Number of timeseries samples in each batch.
#         shuffle: Whether to shuffle output samples, or instead draw them in chronological order.
#         multi_horizon: See `forecast_horizon`.

#     Returns:
#         A tf.data.Dataset instance.
#     """

#     inputs = timeseries_dataset_from_array(
#         np.expand_dims(data_array[:-forecast_horizon], axis=-1),
#         None,
#         sequence_length=input_sequence_length,
#         shuffle=False,
#         batch_size=batch_size,
#     )

#     target_offset = (
#         input_sequence_length
#         if multi_horizon
#         else input_sequence_length + forecast_horizon - 1
#     )
#     target_seq_length = forecast_horizon if multi_horizon else 1
#     targets = timeseries_dataset_from_array(
#         data_array[target_offset:],
#         None,
#         sequence_length=target_seq_length,
#         shuffle=False,
#         batch_size=batch_size,
#     )

#     dataset = tf.data.Dataset.zip((inputs, targets))
#     if shuffle:
#         dataset = dataset.shuffle(100)

#     return dataset.prefetch(16).cache()


# train_dataset, val_dataset = (
#     create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
#     for data_array in [train_array, val_array]
# )

# test_dataset = create_tf_dataset(
#     test_array,
#     input_sequence_length,
#     forecast_horizon,
#     batch_size=test_array.shape[0],
#     shuffle=False,
#     multi_horizon=multi_horizon,
# )

# #functions to make graphs


# def compute_adjacency_matrix(
#     route_distances: np.ndarray, sigma2: float, epsilon: float
# ):
#     """Computes the adjacency matrix from distances matrix.

#     It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
#     compute an adjacency matrix from the distance matrix.
#     The implementation follows that paper.

#     Args:
#         route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
#             distance between roads `i,j`.
#         sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
#         epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
#             if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
#             matrix and `w2=route_distances * route_distances`

#     Returns:
#         A boolean graph adjacency matrix.
#     """
#     num_routes = route_distances.shape[0]
#     route_distances = route_distances / 10000.0
#     w2, w_mask = (
#         route_distances * route_distances,
#         np.ones([num_routes, num_routes]) - np.identity(num_routes),
#     )
#     return (np.exp(-w2 / sigma2) >= epsilon) * w_mask



# class GraphInfo:
#     def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
#         self.edges = edges
#         self.num_nodes = num_nodes


# sigma2 = 0.1
# epsilon = 0.5
# adjacency_matrix = compute_adjacency_matrix(distances, sigma2, epsilon)
# node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
# graph = GraphInfo(
#     edges=(node_indices.tolist(), neighbor_indices.tolist()),
#     num_nodes=adjacency_matrix.shape[0],
# )
# print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")



# ##graph architecture
# class GraphConv(layers.Layer):
#     def __init__(
#         self,
#         in_feat,
#         out_feat,
#         graph_info: GraphInfo,
#         aggregation_type="mean",
#         combination_type="concat",
#         activation: typing.Optional[str] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.graph_info = graph_info
#         self.aggregation_type = aggregation_type
#         self.combination_type = combination_type
#         self.weight = tf.Variable(
#             initial_value=keras.initializers.glorot_uniform()(
#                 shape=(in_feat, out_feat), dtype="float32"
#             ),
#             trainable=True,
#         )
#         self.activation = layers.Activation(activation)

#     def aggregate(self, neighbour_representations: tf.Tensor):
#         aggregation_func = {
#             "sum": tf.math.unsorted_segment_sum,
#             "mean": tf.math.unsorted_segment_mean,
#             "max": tf.math.unsorted_segment_max,
#         }.get(self.aggregation_type)

#         if aggregation_func:
#             return aggregation_func(
#                 neighbour_representations,
#                 self.graph_info.edges[0],
#                 num_segments=self.graph_info.num_nodes,
#             )

#         raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

#     def compute_nodes_representation(self, features: tf.Tensor):
#         """Computes each node's representation.

#         The nodes' representations are obtained by multiplying the features tensor with
#         `self.weight`. Note that
#         `self.weight` has shape `(in_feat, out_feat)`.

#         Args:
#             features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

#         Returns:
#             A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
#         """
#         return tf.matmul(features, self.weight)

#     def compute_aggregated_messages(self, features: tf.Tensor):
#         neighbour_representations = tf.gather(features, self.graph_info.edges[1])
#         aggregated_messages = self.aggregate(neighbour_representations)
#         return tf.matmul(aggregated_messages, self.weight)

#     def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
#         if self.combination_type == "concat":
#             h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
#         elif self.combination_type == "add":
#             h = nodes_representation + aggregated_messages
#         else:
#             raise ValueError(f"Invalid combination type: {self.combination_type}.")

#         return self.activation(h)

#     def call(self, features: tf.Tensor):
#         """Forward pass.

#         Args:
#             features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

#         Returns:
#             A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
#         """
#         nodes_representation = self.compute_nodes_representation(features)
#         aggregated_messages = self.compute_aggregated_messages(features)
#         return self.update(nodes_representation, aggregated_messages)

# ###LSTM + graph convolution

# class LSTMGC(layers.Layer):
#     """Layer comprising a convolution layer followed by LSTM and dense layers."""

#     def __init__(
#         self,
#         in_feat,
#         out_feat,
#         lstm_units: int,
#         input_seq_len: int,
#         output_seq_len: int,
#         graph_info: GraphInfo,
#         graph_conv_params: typing.Optional[dict] = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)

#         # graph conv layer
#         if graph_conv_params is None:
#             graph_conv_params = {
#                 "aggregation_type": "mean",
#                 "combination_type": "concat",
#                 "activation": None,
#             }
#         self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

#         self.lstm = layers.LSTM(lstm_units, activation="relu")
#         self.dense = layers.Dense(output_seq_len)

#         self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

#     def call(self, inputs):
#         """Forward pass.

#         Args:
#             inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

#         Returns:
#             A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
#         """

#         # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
#         inputs = tf.transpose(inputs, [2, 0, 1, 3])

#         gcn_out = self.graph_conv(
#             inputs
#         )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
#         shape = tf.shape(gcn_out)
#         num_nodes, batch_size, input_seq_len, out_feat = (
#             shape[0],
#             shape[1],
#             shape[2],
#             shape[3],
#         )

#         # LSTM takes only 3D tensors as input
#         gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
#         lstm_out = self.lstm(
#             gcn_out
#         )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

#         dense_output = self.dense(
#             lstm_out
#         )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
#         output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
#         return tf.transpose(
#             output, [1, 2, 0]
#         )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)

    
# ##training

# in_feat = 1
# batch_size = 64
# epochs = 20
# input_sequence_length = 12
# forecast_horizon = 3
# multi_horizon = False
# out_feat = 10
# lstm_units = 64
# graph_conv_params = {
#     "aggregation_type": "mean",
#     "combination_type": "concat",
#     "activation": None,
# }

# st_gcn = LSTMGC(
#     in_feat,
#     out_feat,
#     lstm_units,
#     input_sequence_length,
#     forecast_horizon,
#     graph,
#     graph_conv_params,
# )
# inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
# outputs = st_gcn(inputs)

# model = keras.models.Model(inputs, outputs)
# model.compile(
#     optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
#     loss=keras.losses.MeanSquaredError(),
# )
# model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=epochs,
#     callbacks=[keras.callbacks.EarlyStopping(patience=10)],
# )

# ##predict

# x_test, y = next(test_dataset.as_numpy_iterator())
# y_pred = model.predict(x_test)
# plt.figure(figsize=(18, 6))
# plt.plot(y[:, 0, 0])
# plt.plot(y_pred[:, 0, 0])
# plt.legend(["actual", "forecast"])

# naive_mse, model_mse = (
#     np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
#     np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
# )
# print(f"naive MAE: {naive_mse}, model MAE: {model_mse}")