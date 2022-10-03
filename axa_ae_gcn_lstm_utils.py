import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN#, DBSCAN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

import geopy.distance
from geopy.geocoders import Nominatim

# def get_distance(data_loc):
#     end = data_loc.loc[col.name, 'latlon']
#     return data_loc['latlon'].apply(geopy.distance.distance,args=(end,),ellipsoid='WGS-84')

def compute_adjacency_matrix(route_distances: np.ndarray, sigma2: float, epsilon: float):

    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes

def prep_ae(file):
    import pyreadr
    import pyarrow as pa
    import pyarrow.parquet as pq
    import glob
    
    cc = pyreadr.read_r(file)
    data=cc[None].iloc[:,np.r_[0:2,3:6,8,13,40,50,60,65]]
    data.columns=['s' + str(x) for x in range(0,data.shape[1])]

    data.rename(columns={data.columns[data.shape[1]-3]: "diag1", data.columns[data.shape[1]-2]: "diag2"},inplace=True)
    data.diag1, data.diag2 = np.where(data.diag1.str.contains('None'), [data.diag2, data.diag1], [data.diag1, data.diag2])
    list_a = ['pneumonia', 'COPD','asthma','respi','lung','pulm','oxy','air', 'airway']

    dataNL=data[~data['diag1'].str.contains("(" + "|".join(list_a) + ")",case=False, na=False)]
    # d3=d3.append(dataNL.groupby(by=['s1','s6']).count())
    dataL=data[data['diag1'].str.contains("(" + "|".join(list_a) + ")",case=False, na=False)]
    # d4=d4.append(dataL.groupby(by=['s1','s6']).count())
    return dataNL, dataL
                              
def proc_ae(ae_data):
    # pd.unique(ae_data['s1'])
    # replace({'SOUTHERN PART OF ':''},regex=True)
    ae_data['s1'].replace({'RH':'Ruttonjee Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'PYN':'Pamela Youde Nethersole Eastern Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'QEH':'Queen Elizabeth Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'CMC':'Caritas Medical Centre'},regex=True,inplace=True)
    ae_data['s1'].replace({'KWH':'Kwong Wah Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'TMH':'Tuen Mun Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'PWH':'Prince of Wales Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'NDH':'North District Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'YCH':'Yan Chai Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'UCH':'United Christian Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'QMH':'Queen Mary Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'PWH':'Princess Margaret Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'POH':'Pok Oi Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'TKO':'Tseung Kwan O Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'AHN':'Alice Ho Miu Ling Nethersole Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'SJH':'St. John Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'NLT':'North Lantau Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'TSH':'Tang Shiu Kin Hospital'},regex=True,inplace=True)
    ae_data['s1'].replace({'PMH':'Princess Margaret Hospital'},regex=True,inplace=True)
    ae_data.columns=[['cd9_loc','date','diag1']]
    return ae_data

def prep_poll():
    dir_list=os.listdir('poll_data')
    np.sort(dir_list)[-12:]

    df=pd.DataFrame(columns=['DATE','STATION','HOUR','CO','FSP','NO2','NOX','O3','RSP','SO2'])
    dir_list=os.listdir('poll_data/')
    for f in np.sort(dir_list)[-12:]:
        data = pd.read_excel('poll_data/'+f,skiprows=11)
        df = df.append(data,ignore_index=True)
    df.columns=['date','loc','hour','co','fsp','no2','nox','o3','rsp','so2']
    df.to_csv('epd_poll_data.txt',sep='\t')
    df.to_pickle('epd_poll_data.pkl')

def proc_poll():
    poll_data=pd.read_pickle('~/run/AXA_AE_app/data/epd_poll_data.pkl')
    poll_data['date']=pd.to_datetime(poll_data['date'])
    pd.to_datetime(poll_data['date']).dt.isocalendar().day
    poll_data['co']=pd.to_numeric(poll_data['co'], errors='coerce')
    poll_data['fsp']=pd.to_numeric(poll_data['fsp'], errors='coerce')
    poll_data['no2']=pd.to_numeric(poll_data['no2'], errors='coerce')
    poll_data['nox']=pd.to_numeric(poll_data['nox'], errors='coerce')
    poll_data['o3']=pd.to_numeric(poll_data['o3'], errors='coerce')
    poll_data['rsp']=pd.to_numeric(poll_data['rsp'], errors='coerce')
    poll_data['so2']=pd.to_numeric(poll_data['so2'], errors='coerce')

    CC=poll_data.groupby(by=['loc','date']).mean().reset_index()
    return CC

def merge_save(data_loc,CC):
    # data_loc=hosp_loc
    data_loc['latlon'] = list(zip(data_loc['lat'], data_loc['long']))

    square = pd.DataFrame(
        np.zeros((data_loc.shape[0], data_loc.shape[0])),
        index=data_loc.index, columns=data_loc.index
    )
    data_loc.reset_index(inplace=True)


    def get_distance(col):
        end = data_loc.loc[col.name, 'latlon']
        return data_loc['latlon'].apply(geopy.distance.distance,
                                  args=(end,),
                                  ellipsoid='WGS-84'
                                 )
    distances = square.apply(get_distance, axis=1).T


    data_loc['src']=data_loc['name']
    data_loc['dst']=data_loc['name']

    distanceA=distances
    distanceA.index=data_loc['src']
    distanceA.columns=data_loc['dst']

    distances=distances.astype(str) # df.astype(np.float64)#lues.as_int#('int')#.to_numpy()
    distances=distances.replace('km', '', regex=True)
    distances=distances.astype(np.float64)
    distances.shape

    distances=np.round(distances,5)
    distances_plot=np.copy(distances)
    
    dd=[]
    distances.reset_index(inplace=True) ## only run once
    for j,i in enumerate(distances['src'][:18]):
        # print(i,j)
        cc=distances[distances[i]==np.min(distances.iloc[19:,:19],axis=0)[j+1]]['src']
        dd.append(i+'_'+(cc.iloc[0]))
    dd=pd.DataFrame(dd)
    key = pd.DataFrame(dd[0].str.split('_').to_list(), columns=['cd9_loc','poll'])
    CC=CC.merge(key,left_on='loc',right_on='poll')

    ae_data=pd.read_csv('run/AXA_AE_app/data/Lung_incidence.txt',sep='\t')
    ae_data['date']=pd.to_datetime(ae_data['date'])
    ddd=pd.merge(ae_data,CC,on=['date','cd9_loc'])
    table = pa.Table.from_pandas(ddd.astype(str))
    pq.write_table(table, 'run/AXA_AE_app/data/ae_epd_poll_data.parquet')
    
    ae_dataB=pd.read_csv('run/AXA_AE_app/data/nonLung_incidence.txt',sep='\t')
    ae_dataB['date']=pd.to_datetime(ae_dataB['date'])
    eee=pd.merge(ae_dataB,CC,on=['date','cd9_loc'])
    table = pa.Table.from_pandas(eee.astype(str))
    pq.write_table(table, 'run/AXA_AE_app/data/ae_non-epd_poll_data.parquet')
    
    return distances,ae_data,ddd,ae_dataB,eee,distances_plot
def prep_graph_data(data):
    
    
    geolocator = Nominatim(user_agent="example app")
    df_loc=pd.DataFrame(columns=['lat','long','name'])
    for ii,i in enumerate(pd.unique(data['cd9_loc'])):
        a,b,c=geolocator.geocode(str(i)+", Hong Kong").point
        df_loc[ii]=[a,b,i]
    df_loc=df_loc.transpose()
    df_loc.columns=['lat','long','name']
    df_loc=df_loc[3:]

    data_loc=df_loc
    data_loc['latlon'] = list(zip(data_loc['lat'], data_loc['long']))

    square = pd.DataFrame(
        np.zeros((data_loc.shape[0], data_loc.shape[0])),
        index=data_loc.index, columns=data_loc.index
    )
    def get_distance(col):
        end = data_loc.loc[col.name, 'latlon']
        return data_loc['latlon'].apply(geopy.distance.distance,
                                  args=(end,),
                                  ellipsoid='WGS-84'
                                 )
    distances = square.apply(get_distance, axis=1).T

    data_loc['src']=data_loc['name']
    data_loc['dst']=data_loc['name']

    # np.sum((distances<5)*1)
    D_D=pd.DataFrame((distances<5)*1)
    D_D.index=data_loc['src']
    D_D.columns=data_loc['dst']

    E_E=pd.DataFrame(D_D.stack())
    E_E.reset_index(inplace=True)#
    distance_mat=E_E[E_E[0]>0]

    distance=distances
    distance.index=data_loc['src']
    distance.columns=data_loc['dst']
    distance=pd.DataFrame(distance.stack())
    distance.reset_index(inplace=True)

    distances=distances.astype(str)
    distances=distances.replace('km', '', regex=True)
    distances=distances.astype(np.float64)
    distances.shape
    return distances


def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    if data_array.shape[2] ==1:
      inputs = timeseries_dataset_from_array(
          np.expand_dims(data_array[:-forecast_horizon], axis=-1),
          None,
          sequence_length=input_sequence_length,
          shuffle=False,
          batch_size=batch_size,
      )

    else:
      inputs = timeseries_dataset_from_array(
          data_array[:-forecast_horizon,:,1:],
          None,
          sequence_length=input_sequence_length,
          shuffle=False,
          batch_size=batch_size,
      )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:,:,0],
        # data_array[:,:,0],
        # np.expand_dims(data_array[target_offset:,:,0], axis=-1),
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array )#- mean) / std
    val_array = (data_array[num_train : (num_train + num_val)])# - mean) / std
    test_array = (data_array[(num_train + num_val) :])# - mean) / std

    return train_array, val_array, test_array

class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):

        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

class LSTMGC(layers.Layer):

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")#,return_sequences=True)
        self.dense = layers.Dense(output_seq_len)
        # self.dense = layers.Dense(1)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)


def plot_performance():    
    jeff=pd.read_csv('~/run/AXA_AE_app/GNNtest/GNN_mse.txt',sep='\t',index_col=0)
    l=[]
    import ast,seaborn as sns
    for i in np.arange(len(jeff['var'])):
        l.append(len(ast.literal_eval(jeff['var'][i])))

    jeff['var_count']=l
    jeff['var_count']=jeff['var_count']-1
    jeff.rename(columns={'ftime':'train>forecast'},inplace=True)
    jeff['train>forecast'].replace({7:'7>3',14:'14>5',21:'21>7'},inplace=True)
    jeff['shuf'].replace({0:'original_graph',1:'shuffled_graph'},inplace=True)
    jeff['var_count'].replace({1:'AE',5:'pollution',6:'ae+pollution'},inplace=True)
    jeff['density']=np.round(jeff['edges']/jeff['graph_shape'],0)
    jeff['density'].replace({9:10},inplace=True)

    cc=jeff[['shuf','train>forecast','density','mse','naive_mse','var_count']]
    R=cc.melt(id_vars=['var_count','shuf','train>forecast','density'],ignore_index=False).reset_index()
    table = pd.pivot_table(R, values='value', index=['shuf'],columns=['var_count','train>forecast','density']).T#, aggfunc=np.sum)
    table.reset_index(inplace=True)
    # table.rename(columns={'kmeans{k}':'kmeans'},inplace=True)
    # table.rename(columns={'index':'method'},inplace=True)
    # table.rename(columns={'test':'data'},inplace=True)
    # table.rename(columns={'variable':'metric'},inplace=True)

    # table.replace({'group': 0}, 'urban',inplace=True)
    # table.replace({'group': 1}, 'rural',inplace=True)
    table[['original_graph','shuffled_graph']]=np.log(table[['original_graph','shuffled_graph']])
    g=sns.relplot(data=table, x='original_graph', y='shuffled_graph', col='train>forecast',
                  col_order=('7>3','14>5','21>7'),hue='var_count',style='density',kind="scatter",s=100)#,xlim = (.3,.8), ylim = (.3,.8))
    # g.set(ylim=(0.4, .7))
    # g.set(xlim=(0.4, .7))

    def const_line(*args, **kwargs):
        x = (-5,5)#np.arange(0, 50, .5)
        y = (-5,5)#0.2*x
        plt.plot(y, x,ls='--', linewidth=1, color='grey')#, C='k')

    g.map(const_line)

    g.savefig('GNNtest/performance.png')
