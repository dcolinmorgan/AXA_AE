import pandas as pd, numpy as np,os,typing,matplotlib.pyplot as plt,glob,joblib 
from joblib import Parallel
import pyreadr,pyarrow as pa,pyarrow.parquet as pq

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tqdm.keras import TqdmCallback

import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.linear_model import ElasticNet

from axa_ae_gcn_lstm_utils import prep_ae, proc_ae, prep_poll, proc_poll, merge_save, prep_graph_data #, get_distance
from axa_ae_gcn_lstm_utils import compute_adjacency_matrix, GraphInfo, preprocess, GraphConv, LSTMGC, create_tf_dataset

tf.__version__

# prep_ae()

"""##Health Data"""
ae_data=pd.read_parquet('~/run/AXA_AE_app/data/AE_AXA_Lung_incidence.parquet')
ae_data=ae_data[['s0']].reset_index()
ae_dataB=pd.read_parquet('~/run/AXA_AE_app/data/AE_AXA_NonLung_incidence.parquet')
ae_dataB=ae_dataB[['s0']].reset_index()

ae_data=proc_ae(ae_data)
ae_data.to_csv('~/run/AXA_AE_app/data/Lung_incidence.txt',sep='\t')
ae_dataB=proc_ae(ae_dataB)
ae_dataB.to_csv('~/run/AXA_AE_app/data/nonLung_incidence.txt',sep='\t')

geolocator = Nominatim(user_agent="example app")
hosp_loc=pd.DataFrame(columns=['lat','long','name'])
for ii,i in enumerate(np.unique(ae_data['cd9_loc'].values)):
    a,b,c=geolocator.geocode(str(i)+", Hong Kong").point
    hosp_loc[ii]=[a,b,i]
    

hosp_loc=hosp_loc.transpose()
hosp_loc.columns=['lat','long','name']
hosp_loc=hosp_loc[3:]

"""##Poll Data"""
# !wget -P run/AXA_AE_app/poll_data/ https://cd.epic.epd.gov.hk/EPICDI/air/yearly/data/{1990..2021}_EN.xlsx

# get_poll()
CC=proc_poll()

geolocator = Nominatim(user_agent="example app")
poll_loc=pd.DataFrame(columns=['lat','long','name'])

for ii,i in enumerate(np.unique(CC['loc'].values)):
    a,b,c=geolocator.geocode(str(i)+", Hong Kong").point
    poll_loc[ii]=[a,b,i]

    
poll_loc=poll_loc.transpose()

poll_loc.columns=['lat','long','name']
poll_loc=poll_loc[3:]

hosp_loc=hosp_loc.append(poll_loc)
hosp_loc=hosp_loc[hosp_loc['name']!='SOUTHERN']

distances,ae_data,ddd,ae_dataB,eee,distances_plot=merge_save(hosp_loc,CC)
plt.matshow(distances_plot)
plt.close() 
"""## Graph data"""
distances=prep_graph_data(ddd)

covar_list=[
['diag1','diag1','fsp','rsp','o3','no2','so2'],
['diag1','fsp','rsp','o3','no2','so2'], #'nox','co'
['diag1','diag1'],
# ['fsp','rsp','o3','no2','so2']
]
shuf_list, pred_horizon_list, var_list, dist_list, mse_list, naive_mse_list, EL_mse_list,graph_list, graph_edge_list, train_shape_list, val_shape_list, test_shape_list= ([] for i in range(12))

fore_time=[[21,7],[14,5],[7,3]]
for dist_mat_shuf in [0,1]:
    for dist_mat_tr in [.99998,5,10,15]:
        for covar in covar_list:
            for ftime in fore_time:
            
                if dist_mat_shuf==1:
                    distances=np.random.permutation(distances)
                if dist_mat_tr>1:
                    adjacency_matrix =np.where(distances > dist_mat_tr, 0, 1)
                elif dist_mat_tr<1:
                    sigma2 = 0.1
                    # dist_mat_tr = .99998
                    adjacency_matrix = compute_adjacency_matrix(distances, sigma2, dist_mat_tr)


                node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
                graph = GraphInfo(
                    edges=(node_indices.tolist(), neighbor_indices.tolist()),
                    num_nodes=adjacency_matrix.shape[0],
                )
                print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

                plt.figure(figsize=(8, 8))
                plt.matshow((adjacency_matrix))
                plt.close() 
                """## LSTM data"""
                
                # ae_data=pd.read_csv('~/run/AXA_AE_app/Lung_incidence.txt',sep='\t')
                # ae_data['date']=pd.to_datetime(ae_data['date'])
                # ddd=pd.merge(ae_data,CC,on=['date','cd9_loc'])
                
                measures_array=ddd.pivot_table(values=covar, index='date', columns=['cd9_loc'], aggfunc='mean')#.fillna(method='ffill') ## COand NOX missing locations for now

                measures_array=measures_array.fillna(method='ffill').dropna().values ## ffill more NaN
                
                measures_array=measures_array.reshape([measures_array.shape[0],len(covar),18])
                measures_array=np.transpose(measures_array, [0, 2, 1])

                plt.figure(figsize=(18, 6))
                plt.plot(measures_array[:,1])
                plt.legend(covar)
                plt.close() 
                """## LSTM setup"""

                train_size, val_size = 0.5, 0.2

                train_array, val_array, test_array = preprocess(measures_array, train_size, val_size)

                print(f"train set size: {train_array.shape}")
                print(f"validation set size: {val_array.shape}")
                print(f"test set size: {test_array.shape}")

                np.isfinite(train_array).all()
                # np.isfinite(b_0).all()


                batch_size = 64
                input_sequence_length = ftime[0]
                forecast_horizon = ftime[1]
                multi_horizon = True

                train_dataset, val_dataset = (
                    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
                    for data_array in [train_array, val_array]
                )

                test_dataset = create_tf_dataset(
                    test_array,
                    input_sequence_length,
                    forecast_horizon,
                    batch_size=test_array.shape[0],
                    shuffle=False,
                    multi_horizon=multi_horizon,
                )

                train_dataset

                """## Model training"""

                keras.backend.clear_session()

                in_feat = val_dataset.element_spec[0].shape[3]
                batch_size = 64
                epochs = 50
                multi_horizon = False
                out_feat = 14
                lstm_units = 256
                graph_conv_params = {
                    "aggregation_type": "mean",
                    "combination_type": "concat",
                    "activation": None,
                }

                st_gcn = LSTMGC(
                    in_feat,
                    out_feat,
                    lstm_units,
                    input_sequence_length,
                    forecast_horizon,
                    graph,
                    graph_conv_params,
                )
                inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
                outputs = st_gcn(inputs)

                model = keras.models.Model(inputs, outputs)
                model.compile(
                    optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
                    loss=keras.losses.MeanAbsoluteError(),
                )

                model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    callbacks=[keras.callbacks.EarlyStopping(patience=3)],
                    verbose=0,
                    # callbacks=[TqdmCallback(verbose=0)]
                )


#                 x_test, y = next(test_dataset.as_numpy_iterator())
#                 y_pred = model.predict(x_test)
                
#                 naive_mse, model_mse = (
#                     # np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
#                     # np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),

#                     np.square(np.nanmean(x_test[:, -1, :,0] - y[:, 0, :])),
#                     np.square(np.nanmean(y_pred[:, 0, :] - y[:, 0, :])),
#                 )
#                 print(f"naive MSE: {naive_mse}, model MSE: {model_mse}")

#                 plt.figure(figsize=(18, 6))
#                 plt.plot(y[:, 0, 0])
#                 plt.plot(y_pred[:, 0, 0])


                train_array2=np.transpose(train_array, [2,0,1])
                cc=train_array2.reshape(train_array.shape[0]*train_array.shape[1],train_array.shape[2])
                x_test, y = next(test_dataset.as_numpy_iterator())
                y_pred = model.predict(x_test)
                EN=ElasticNet().fit(cc[:-135,1:],cc[:-135,0])
                predA=EN.predict(cc[-135:,1:])

                plt.figure(figsize=(18, 6))
                plt.plot(y[:, 0, 0])
                plt.plot(y_pred[:, 0, 0])
                plt.plot(x_test[:, -1, 0,0])
                plt.plot(predA)

                plt.legend(["actual",'forecast', "naive","ElastNet"])

                naive_mse, model_mse, EL_mse = (
                    np.square(np.nanmean(x_test[:, -1, :,0] - y[:, 0, :])),
                    np.square(np.nanmean(y_pred[:, 0, :] - y[:, 0, :])),
                    np.square(np.nanmean(predA - y[:, 0,0])),
                )
                pathlib.Path('run/AXA_AE_app/GNNtest'+str(date)+'/figs/').mkdir(parents=True, exist_ok=True)
                
                # print(f"model MSE: {model_mse},naive MSE: {naive_mse}, ElNet MSE:{EL_mse}") 

                plt.title('horiz:'+str(ftime)+'_var:'+str(len(covar))+'_shuf:'+str(dist_mat_shuf)+'_adjcut'+str(dist_mat_tr)+'_mse:'+str(model_mse)+'_ELN_mse:'+str(EL_mse))
                plt.legend(["actual", "forecast"])
                plt.savefig('run/AXA_AE_app/GNNtest/figs/horiz'+str(ftime[0])+'_var'+str(len(covar))+'_shuf'+str(dist_mat_shuf)+'_adjcut'+str(dist_mat_tr)+'_mse'+str(np.round(model_mse,4))+'.png')
                plt.close() 
                
                shuf_list.append(dist_mat_shuf)
                pred_horizon_list.append(ftime[0])
                var_list.append(covar)
                dist_list.append(dist_mat_tr)
                mse_list.append(model_mse)
                naive_mse_list.append(naive_mse)
                EL_mse_list.append(EL_mse)
                graph_list.append(graph.num_nodes)
                graph_edge_list.append(len(graph.edges[0]))
                train_shape_list.append(train_array.shape)
                val_shape_list.append(val_array.shape)
                test_shape_list.append(test_array.shape)
                

df = pd.DataFrame(list(zip(shuf_list, pred_horizon_list, var_list,dist_list, mse_list, naive_mse_list, graph_list, graph_edge_list, train_shape_list, val_shape_list, test_shape_list)), 
           columns =['shuf','ftime','var','distTR','mse','naive_mse','graph_shape','edges','train_shape','val_shape','test_shape'])

df.to_csv('run/AXA_AE_app/GNNtest/GNN_mse.txt',sep='\t')