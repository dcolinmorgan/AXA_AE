import pandas as pd, numpy as np,os,typing,matplotlib.pyplot as plt,glob,joblib ,sys,getopt
from joblib import Parallel
import pyreadr,pyarrow as pa,pyarrow.parquet as pq
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.metrics import MeanAbsoluteError,RootMeanSquaredError,BinaryCrossentropy,mean_absolute_error
# from tqdm.keras import TqdmCallback
from sklearn.linear_model import ElasticNet,LinearRegression,BayesianRidge,LassoLars

import pathlib
# from tensorflow.keras.metrics import 
import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler,normalize
# from keras.utils import normalize
# from tensorflow.keras.metrics import MeanAbsoluteError,RootMeanSquaredError,BinaryCrossentropy,mean_absolute_error
# from keras.utils import normalize
from axa_ae_gcn_lstm_utils import prep_ae, proc_ae, prep_poll, proc_poll, merge_save, prep_graph_data, plot_performance,shuf_data
from axa_ae_gcn_lstm_utils import compute_adj_mat, GraphInfo, preprocess, GCNet, LSTM_GCN, mk_tfsf

tf.__version__
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

"""
Description:
Run GCN_LSTM benchmark.
Usage:
-h, --help: help
-i, --indir: directory where intersections are location
-o, --outdir
-t, --fore_time: forecast time (days) , default 10,7,5,3
-g, --graph_shuf: shuffle graph 0=No, default=0,1
-c, --graph_cut: distance threshold for graph inclusion default=5,10,15
-s, --shufd: shuffle data, 0=None, 1=ae, 2=poll, 3=all, default=all
-f, --figs: print figs, default 0=No

Example:
source activate tf-mac ## use miniforge not conda to install tensorflow-dev, -mac, etc to utilize M1 gpu
python Documents/axa_ae_gcn_lstm/axa_ae_gcn_lstm.py -i Documents/axa_ae_gcn_lstm/ -t 7 -c 10 -f 1 # -g 0 -s 0
python Documents/axa_ae_gcn_lstm/axa_ae_gcn_lstm.py -i Documents/axa_ae_gcn_lstm/ -f 1 ## full benchmark
python Documents/axa_ae_gcn_lstm/axa_ae_gcn_lstm.py -i Documents/axa_ae_gcn_lstm/ -t 7 -c 10 -f 1 ## full benchmark

"""

def main(argv):
    #Create variables
    indir,outdir,fore_time,dist_mat_shuf,graph_cutoff,shufd= ([None for i in range(6)])
    try:
        opts, args = getopt.getopt(argv, 'hi:o:t:g:c:s:f:', ['help', 'indir=','outdir=','fore_time=','dist_mat_shuf=','graph_cutoff=','shufd=','figs='])
    except getopt.GetoptError as err:
        print(str(err))  # will print something like "option -a not recognized"
        print(__doc__)
        return 2

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(__doc__)
            return 0
        elif opt in ('-i', '--indir'):
            indir = arg
        elif opt in ('-o', '--outdir'):
            outdir = arg
        elif opt in ('-t', '--fore_time'):
            FT = arg
        elif opt in ('-g','--graph_shuf'):
            GS = arg
        elif opt in ('-c', '--graph_cutoff'):
            GC = arg
        elif opt in ('-s','--shufd'):
            SD = arg
        elif opt in ('-f','--figs'):
            figs = arg

        else:
            print('Unknown option', opt)
            return 1
    if indir is None:
        indir= ''
    if outdir is None:
        import time
        timestr = time.strftime("%Y%m%d-%H%M%S")
        outdir=indir+'GNNtest_'+timestr
    if not 'GS' in locals():
        GS =[0,1]
    else: 
        GS= [GS]
    if not 'SD' in locals():
        SD = [0,1]#,2,3,4]
    else:
        SD=[int(SD)]
    if not 'FT' in locals(): #=='many':
        FT=10
        FT=[FT,FT-3]#,FT-5,FT-7]
        FT=[[int(i)*3,int(i)] for i in FT]
    else:
        FT=[[int(FT)*3,int(FT)]]
    if not 'GC' in locals(): #=='many':
        GC=5
        GC=[int(GC),int(GC)*2,int(GC)*3]
    else:
        GC=[int(GC)]
    if not 'figs' in locals():
        figs=None
    else:
        figs=int(figs)

    # prep_ae() # python AXA_AE.py ## iterates on '/home/edmondyip/AnE_data/data/AE_attendance_csv/*.rds'
    pathlib.Path(outdir+'/figs/').mkdir(parents=True, exist_ok=True)
    # pathlib.Path(outdir+'/raw/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(outdir+'/predR/figs/').mkdir(parents=True, exist_ok=True)
    pathlib.Path(outdir+'/predR/data/').mkdir(parents=True, exist_ok=True)


    """##Health Data"""
    ae_data=pd.read_parquet(indir+'data/AE_AXA_Lung_incidence.parquet')
    ae_data=ae_data[['s0']].reset_index()
    ae_dataB=pd.read_parquet(indir+'data/AE_AXA_NonLung_incidence.parquet')
    ae_dataB=ae_dataB[['s0']].reset_index()

    ae_data['prev']=ae_data['s0']/ae_dataB['s0']
    ae_data['total']=ae_dataB['s0']
    
    ae_data=proc_ae(ae_data)
    ae_data.to_csv(outdir+'/Lung_prevalence.txt',sep='\t')

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
    CC=proc_poll(indir)

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
    CC=CC[CC.date.dt.isocalendar().year<2021]
    distances,ae_data,eee,distances_plot=merge_save(hosp_loc,CC,outdir)
    if figs==1:
        if not (outdir+'/figs/dist_graph.png'):
            plt.matshow(distances_plot)
            plt.savefig(outdir+'/figs/dist_graph.png')
            plt.close() 
    """## Graph data"""
    distances=prep_graph_data(eee)

    covar_list=[
    ['diag1','total','fsp','nox','rsp','o3','no2','so2'],
    ['diag1','fsp','nox','rsp','o3','no2','so2'], #'nox','co'
    ['diag1','total'],
    ['diag1','total','nox']
    ]
    shuf_list, pred_horizon_list, var_list, dist_list, graph_list, graph_edge_list, train_shape_list, val_shape_list, test_shape_list,shufd_list= ([] for i in range(10))#, mse_list, naive_mse_list, EL_mse_list,mape_list, naive_mape_list, EL_mape_list, mae_list, naive_mae_list, EL_mae_list= ([] for i in range(19))
    METS=pd.DataFrame();METS2=pd.DataFrame();METS3=pd.DataFrame();
    for dist_mat_shuf in tqdm(GS,leave=True): # in [0,1]:
        for graph_cutoff in GC: # in [.99998,5,10,15]:
            for covar in tqdm(covar_list,leave=False):
                for foretime in FT:
                    for shufd in SD:# in tqdm([0,1,2,3],desc= f"loop for i= {i}",leave=False):
                        print([dist_mat_shuf,graph_cutoff,shufd])
                        if dist_mat_shuf==1:
                            tmpA=distances.copy()
                            distances=np.random.permutation(distances)
                            distances=pd.DataFrame(distances)
                            distances.index=tmpA.index
                            distances.columns=tmpA.index
                        if graph_cutoff>1:
                            adjacency_matrix =np.where(distances > graph_cutoff, 0, 1)
                        elif graph_cutoff<1:
                            sigma2 = 0.1
                            # graph_cutoff = .99998
                            adjacency_matrix = compute_adj_mat(distances, sigma2, graph_cutoff)

                        ddd=eee.copy()

                        node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
                        graph = GraphInfo(
                            edges=(node_indices.tolist(), neighbor_indices.tolist()),
                            num_nodes=adjacency_matrix.shape[0],
                        )
                        if (figs==1) and (dist_mat_shuf==0):
                            if not os.path.isfile(outdir+'/figs/adj_mat'+str(graph_cutoff)+'.png'):
                                plt.figure(figsize=(8, 8))
                                plt.matshow((adjacency_matrix))
                                plt.savefig(outdir+'/figs/adj_mat'+str(graph_cutoff)+'.png')
                                plt.close() 
                        """## LSTM data"""

                        ddd[['diag1','total','co','nox','fsp','no2','o3','rsp','so2']] = ddd.groupby('cd9_loc')[['diag1','total','co','nox','fsp','no2','o3','rsp','so2']].rolling(3).mean().droplevel(0)
                        
                        ddd.drop(columns=['Unnamed: 0','loc','poll','co'],inplace=True)

                        
                        if shufd!=0:
                            dd=[np.random.permutation(ddd[var]) for var in ddd.columns[3:]] ## dont shuffle y or date
                            ddd.iloc[:,3:]=pd.DataFrame(dd).T
                        ddd['prev'] = ddd['diag1']/ddd['total']
                            # ddd=ddd #shuf_data(ddd,shufd)
#                         if shufd==1:
#                             ddd['total']=np.random.permutation(ddd['total']) ##shuffle within sample (location) and time
#                         elif shufd==2: 
#                             # ddd['co']=np.random.permutation(ddd['co'])
#                             ddd['fsp']=np.random.permutation(ddd['fsp'])
#                             ddd['no2']=np.random.permutation(ddd['no2'])
#                             ddd['nox']=np.random.permutation(ddd['nox'])
#                             ddd['o3']=np.random.permutation(ddd['o3'])
#                             ddd['rsp']=np.random.permutation(ddd['rsp'])
#                             ddd['so2']=np.random.permutation(ddd['so2'])
#                         elif shufd==3: 

#                             ddd['total']=np.random.permutation(ddd['total'])
#                             # ddd['co']=np.random.permutation(ddd['co'])
#                             ddd['fsp']=np.random.permutation(ddd['fsp'])
#                             ddd['no2']=np.random.permutation(ddd['no2'])
#                             ddd['nox']=np.random.permutation(ddd['nox'])
#                             ddd['o3']=np.random.permutation(ddd['o3'])
#                             ddd['rsp']=np.random.permutation(ddd['rsp'])
#                             ddd['so2']=np.random.permutation(ddd['so2'])

                        measures_array=ddd.pivot_table(values=covar, index=['date'], columns=['cd9_loc'], aggfunc='mean',sort=False) #.fillna(method='ffill') ## COand NOX missing locations for now
                        del ddd
                        measures_array=measures_array.fillna(method='bfill',axis=1).dropna().values ## ffill more NaN
                        measures_array=measures_array.reshape([measures_array.shape[0],len(covar),distances.shape[0]])
                        measures_array=np.transpose(measures_array, [0, 2, 1])

                        """## LSTM setup"""

                        train_size, val_size = 0.5, 0.3

                        train_array, val_array, test_array = preprocess(measures_array, train_size, val_size)
                        del measures_array

                        batch_size = 64
                        input_sequence_length = foretime[0]
                        forecast_horizon = foretime[1]
                        multi_horizon = True

                        train_dataset, val_dataset , test_dataset= (
                            mk_tfsf(data_array, input_sequence_length, forecast_horizon, batch_size=data_array.shape[0], shuffle=False,  multi_horizon= multi_horizon)
                            for data_array in [train_array, val_array,test_array]
                        )

                        """## Model training"""

                        keras.backend.clear_session()

                        in_feat = val_dataset.element_spec[0].shape[3]
                        batch_size = batch_size
                        epochs = 500
                        multi_horizon = True
                        out_feat = distances.shape[0]
                        lstm_units = 256
                        graph_conv_params = {
                            "aggregation_type": "mean",
                            "combination_type": "concat",
                            "activation": None,
                        }

                        st_gcn = LSTM_GCN(
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
                            optimizer=keras.optimizers.RMSprop(learning_rate=0.002),
                            loss=keras.losses.MeanAbsoluteError(),
                        )

                        model.fit(       #### fit shuffles inside loop so repeat train and predict and plot together
                            train_dataset,
                            validation_data=val_dataset,
                            epochs=epochs,
                            callbacks=[keras.callbacks.EarlyStopping(patience=50)],
                            verbose=0,
                            # callbacks=[TqdmCallback(verbose=0)] # callbacks=[pbar]
                        )                        

                        x_test, y = next(test_dataset.as_numpy_iterator())

                        y_pred = model.predict(x_test)
                        
                        # metrics=[MeanAbsoluteError]#,MeanAbsoluteError,BinaryCrossentropy]                  
                        
                        
                        test_array=np.transpose(test_array, [0, 2, 1])
                        # train_array=np.transpose(train_array, [0, 2, 1])
                        # metric_df=pd.DataFrame(['naive_'+str(i),'model_'+str(i),'EL_'+str(i)] for i in metrics)
                        # for m in methods:
                            # method=str(m).split('.')[-1].split("'")[0]

                        # m=MeanAbsoluteError#eval(method)()
                        naive=np.mean(mean_absolute_error(y_true=y[:,0,:],y_pred=test_array[-y.shape[0]:,0,:]))#.result().numpy()
                        # metric_df['naive']=m.result().numpy()

                        # m=MeanAbsoluteError#eval(method)()
                        gcn=np.mean(mean_absolute_error(y_true=y[:,0,:], y_pred=y_pred[:,0,:]))#.result().numpy()
                        # metric_df['GCN']=m.result().numpy()
                        test_array=np.transpose(test_array, [0, 2, 1])
                        
                        # Lin_Train=train_array[:,:,1:].reshape(train_array.shape[0]*train_array.shape[1],train_array.shape[2]-1)
                        # Lin_Val=train_array[:,:,0].reshape(train_array.shape[0]*train_array.shape[1])
                        # Lin_Test=test_array[:,:,1:].reshape(test_array.shape[0]*test_array.shape[1],test_array.shape[2]-1)
                        # Lin_True=test_array[:,:,0]
                        Lin_Train=train_array[:y.shape[0],:,1:].reshape(y.shape[0]*train_array.shape[1],train_array.shape[2]-1)
                        Lin_Val=train_array[:y.shape[0],:,0].reshape(y.shape[0]*train_array.shape[1])
                        Lin_Test=test_array[:y.shape[0],:,1:].reshape(y.shape[0]*test_array.shape[1],test_array.shape[2]-1)
                        Lin_True=test_array[:y.shape[0],:,0]


                        methods=[LinearRegression,ElasticNet,BayesianRidge,LassoLars]                  
                        methods_df=pd.DataFrame([str(i)] for i in methods)
                        for method in methods:
                            met=str(method).split('.')[-1].split("'")[0]
                            mm=method().fit(Lin_Train,Lin_Val)
                            Lin_pred=mm.predict(Lin_Test)
                            if 'Lin_predA' not in locals():
                                Lin_predA=pd.DataFrame(Lin_pred)
                            else:
                                Lin_predA=pd.concat([Lin_predA,pd.DataFrame(Lin_pred)],axis=1)
                            Lin_pred=Lin_pred.reshape(y.shape[0],test_array.shape[1])
                            cc=np.mean(mean_absolute_error(y_true=Lin_True, y_pred=Lin_pred)) ##.result().numpy()
                            methods_df[met]=cc
                            # print(cc)
                            METS=pd.DataFrame(methods_df.drop(columns=0).iloc[0,:]).T
                        METS=METS.append(methods_df) ### add other methods to output (despite diff dims)
                        METS['naive']=naive;METS['gcn']=gcn;
                        METS=METS.drop(columns=0).drop_duplicates(keep='first')
                        METS['d']=graph_cutoff
                        METS['t']=foretime[0]
                        METS['c']=graph_cutoff
                        METS['v']=len(covar)
                        METS['g']=dist_mat_shuf
                        METS['s']=shufd
                        METS['n']=graph.num_nodes
                        METS['e']=len(graph.edges[0])

                        Lin_predA.columns=methods_df.columns[1:]
                        
                        Lin_predA['y']=y[:,0,:].reshape(y_pred[:,0,:].shape[0]*y_pred[:,0,:].shape[1])
                        Lin_predA['naive']=test_array[-y.shape[0]:,:,0].reshape(y_pred[:,0,:].shape[0]*y_pred[:,0,:].shape[1])
                        Lin_predA['gcn']=y_pred[:,0,:].reshape(y_pred[:,0,:].shape[0]*y_pred[:,0,:].shape[1])
                        pathlib.Path(outdir+'/predR/figs/').mkdir(parents=True, exist_ok=True)
                        pathlib.Path(outdir+'/predR/data/').mkdir(parents=True, exist_ok=True)
                                
                        
                        outtt=outdir+'/predR/data/horiz'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_s'+str(shufd)+'_d'+str(graph_cutoff)+'.txt'
                        # ccc=np.transpose(x_test, [0,3,2,1])
                        Lin_predA.to_csv(outtt)
                        METS.to_csv(outdir+'/GNN_perf.txt',sep='\t',mode='a')
                        
                        tmp=Lin_predA['LinearRegression']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        tmp=Lin_predA['ElasticNet']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        tmp=Lin_predA['BayesianRidge']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        # tmp=Lin_predA['LassoLars']
                        # plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        tmp=Lin_predA['naive']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        tmp=Lin_predA['gcn']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        tmp=Lin_predA['y']
                        plt.plot(np.mean(tmp.to_numpy().reshape(y.shape[0],y.shape[2]),axis=1))
                        plt.legend(['ElasticNet','BayesianRidge','naive','gcn','y'])
                        outtt=outdir+'/predR/figs/horiz'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_s'+str(shufd)+'_d'+str(graph_cutoff)+'.png'

                        plt.savefig(outtt)
                        del methods_df, Lin_predA,METS
                        # np.savez(outtt,y=y,ypred=y_pred,predA=predA,predB=predB)#[:y.shape[0],:])

                        # shuf_list.append(dist_mat_shuf)
                        # pred_horizon_list.append(foretime[0])
                        # var_list.append(covar)
                        # dist_list.append(graph_cutoff)

                        # graph_list.append(graph.num_nodes)
                        # graph_edge_list.append(len(graph.edges[0]))
                        # train_shape_list.append(train_array.shape)
                        # val_shape_list.append(val_array.shape)
                        # test_shape_list.append(test_array.shape)
                        # shufd_list.append(shufd)
                        
                        


#     df = pd.DataFrame(list(zip(shuf_list, pred_horizon_list, var_list,dist_list,  graph_list, graph_edge_list, train_shape_list, val_shape_list, test_shape_list,shufd_list)),
#        columns =['shuf','foretime','var','distTR','graph_shape','edges','train_shape','val_shape','test_shape','data_shuffle'])
    
#     # df=pd.merge(df,METS)
#     df=pd.concat([df, METS.iloc[:,3:].drop_duplicates().reset_index().drop(columns='index')], axis=1)#,ignore_index=True)
    
    # df.to_csv(outdir+'/GNN_perf.txt',sep='\t')

    plot_performance(outdir+'/GNN_perf.txt',outdir)

    print('All done!')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))