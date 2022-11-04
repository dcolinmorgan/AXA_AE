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

import pathlib
# from tensorflow.keras.metrics import 
import geopy.distance
from geopy.geocoders import Nominatim
from sklearn.preprocessing import MinMaxScaler,normalize
# from keras.utils import normalize
from sklearn.linear_model import ElasticNet,LinearRegression
# from keras.utils import normalize
from axa_ae_gcn_lstm_utils import prep_ae, proc_ae, prep_poll, proc_poll, merge_save, prep_graph_data, plot_performance
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
        SD = [0,1,2,3]
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
    CC=CC[CC.date.dt.isocalendar().year<2020]
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
                        ddd['prev'] = ddd['diag1']/ddd['total']
                        
                        # if shufd==0:
                        #     ddd=ddd
                        if shufd==1:
                            ddd['total']=np.random.permutation(ddd['total']) ##shuffle within sample (location) and time
                        elif shufd==2: 
                            ddd['co']=np.random.permutation(ddd['co'])
                            ddd['fsp']=np.random.permutation(ddd['fsp'])
                            ddd['no2']=np.random.permutation(ddd['no2'])
                            ddd['nox']=np.random.permutation(ddd['nox'])
                            ddd['o3']=np.random.permutation(ddd['o3'])
                            ddd['rsp']=np.random.permutation(ddd['rsp'])
                            ddd['so2']=np.random.permutation(ddd['so2'])
                        elif shufd==3: 

                            ddd['total']=np.random.permutation(ddd['total'])
                            ddd['co']=np.random.permutation(ddd['co'])
                            ddd['fsp']=np.random.permutation(ddd['fsp'])
                            ddd['no2']=np.random.permutation(ddd['no2'])
                            ddd['nox']=np.random.permutation(ddd['nox'])
                            ddd['o3']=np.random.permutation(ddd['o3'])
                            ddd['rsp']=np.random.permutation(ddd['rsp'])
                            ddd['so2']=np.random.permutation(ddd['so2'])
                        # ddd.to_csv('test.txt',sep='\t')

                        measures_array=ddd.pivot_table(values=covar, index=['date'], columns=['cd9_loc'], aggfunc='mean',sort=False) #.fillna(method='ffill') ## COand NOX missing locations for now
                        del ddd

                        measures_array=measures_array.fillna(method='bfill',axis=1).dropna().values ## ffill more NaN

                        
                        measures_array=measures_array.reshape([measures_array.shape[0],len(covar),distances.shape[0]])
                        measures_array=np.transpose(measures_array, [0, 2, 1])

                        """## LSTM setup"""

                        train_size, val_size = 0.5, 0.3

                        train_array, val_array, test_array = preprocess(measures_array, train_size, val_size)
                        del measures_array
                        
                        
                        # from keras.utils import normalize
                        # train_array[:,:,1:]=normalize(train_array[:,:,1:],axis=2)#,order=0)
                        # val_array[:,:,1:]=normalize(val_array[:,:,1:],axis=2)#,order=0)
                        # test_array[:,:,1:]=normalize(test_array[:,:,1:],axis=2)#,order=0)
                        
                        batch_size = 64
                        input_sequence_length = foretime[0]
                        forecast_horizon = foretime[1]
                        multi_horizon = True

                        train_dataset, val_dataset , test_dataset= (
                            mk_tfsf(data_array, input_sequence_length, forecast_horizon, batch_size=data_array.shape[0], shuffle=False,  multi_horizon= multi_horizon)
                            for data_array in [train_array, val_array,test_array]
                        )

                        # test_dataset = mk_tfsf(
                        #     test_array,
                        #     input_sequence_length,
                        #     forecast_horizon,
                        #     batch_size=test_array.shape[0],
                        #     shuffle=False,
                        #     multi_horizon=multi_horizon,
                        # )

                        # train_dataset

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
                            optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
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
                        
                        # train_array=normalize(train_array,axis=1)
#                         train_array2=np.transpose(train_array, [0,2,1])
#                         zzz=train_array2[:,1:,:].reshape(train_array2.shape[0]*train_array2.shape[2],(train_array2.shape[1]-1))
#                         www=train_array2[:,0,:].reshape(train_array2.shape[0]*train_array2.shape[2],train_array2.shape[1]-len(covar)-1)
#                         EN=ElasticNet(random_state=0).fit(zzz,www)
                        
#                         # test_array=normalize(test_array,axis=1)
#                         test_array2=np.transpose(test_array, [0,2,1])
#                         zzzz=test_array2[:,1:,:].reshape(test_array2.shape[0]*test_array2.shape[2],(test_array2.shape[1]-1))
#                         wwww=test_array2[:,0,:].reshape(test_array2.shape[0]*test_array2.shape[2],test_array2.shape[1]-len(covar)-1)
#                         predA=LR.predict(zzzz)
                        ccc=train_array[:,:,1:].reshape(train_array.shape[0]*train_array.shape[1],train_array.shape[2]-1)
                        ddd=train_array[:,:,0].reshape(train_array.shape[0]*train_array.shape[1])
                        LR=LinearRegression().fit(ccc,ddd)
            
                        cccc=test_array[:,:,1:].reshape(test_array.shape[0]*test_array.shape[1],test_array.shape[2]-1)
                        dddd=test_array[:,:,0].reshape(test_array.shape[0]*test_array.shape[1])
                        predA=LR.predict(cccc)
                        predB=predA.reshape(test_array.shape[0],test_array.shape[1])
        
                        # predA=predA.reshape(test_array2.shape[0],test_array2.shape[2])
                        # wwww=wwww.reshape(test_array2.shape[0],test_array2.shape[2])

                        outtt=outdir+'/predR/data/horiz'+str(foretime[0])+'_var'+str(len(covar))+'_gshuf'+str(dist_mat_shuf)+'_dshuf'+str(shufd)+'_adjcut'+str(graph_cutoff)
                        # ccc=np.transpose(x_test, [0,3,2,1])
                        np.savez(outtt,y=y,ypred=y_pred,predA=predA,predB=predB)#[:y.shape[0],:])

                        metrics=[MeanAbsoluteError]#,MeanAbsoluteError,BinaryCrossentropy]                  
                        metric_df=pd.DataFrame(['naive_'+str(i),'model_'+str(i),'EL_'+str(i)] for i in metrics)
                        for m in metrics:
                            method=str(m).split('.')[-1].split("'")[0]

                            m=eval(method)()
                            # print([y.shape,x_test.shape,y_pred.shape,predA.shape])
                            # m.update_state(y_true=y[:,0,:], y_pred=test_array2[:y.shape[0],0,:])#.result().numpy()
                            m.update_state(y_true=y[:,0,:],y_pred=test_array[-y.shape[0]:,:,0])#.result().numpy()
                            metric_df['naive_'+method]=m.result().numpy()
                            m=eval(method)()
                            m.update_state(y_true=y[:,0,:], y_pred=y_pred[:,0,:])#.result().numpy()
                            metric_df['model_'+method]=m.result().numpy()
                            m=eval(method)()
                            m.update_state(y_true=dddd, y_pred=predA)#.result().numpy()
                            dd=dddd.reshape(int(dddd.shape[0]/distances.shape[0]),distances.shape[0])
                            m.update_state(y_true=dd[:y_pred[:,0,0].shape[0],:],y_pred=predB[:y_pred[:,0,0].shape[0],:])#.result().numpy()
                            ### LinReg -- enumerate districts
                            metric_df['EL_'+method]=m.result().numpy()
                            
                            METS=METS.append(metric_df)
                        del metric_df
                        if figs==1:
                            pathlib.Path(outdir+'/predR/').mkdir(parents=True, exist_ok=True)
                            ppp=0;rrr=0;PC=[]
                            metric_df=pd.DataFrame(['ypred_'+str(i),'EL_'+str(i),'naive_'+str(i)] for i in metrics)

                            for region,RR in enumerate((distances.index)):
                                
                                ccc=train_array[:,region,1:]
                                ddd=train_array[:,region,0]

                                LR=LinearRegression().fit(ccc,ddd)
                                cccc=test_array[:,region,1:]
                                dddd=test_array[:,region,0]

                                predC=LR.predict(cccc)
                                PC.append(predC)                                
                                
                                plt.figure(figsize=(18, 6))
                                plt.plot(y[:, 0, region])
                                plt.plot(y_pred[:, 0, region])
                                plt.plot(predB[:y_pred[:,0,0].shape[0],region])
                                plt.plot(predC[:y_pred[:,0,0].shape[0]])
                                plt.plot(test_array[:y.shape[0],region,0])

                                A=mean_absolute_error(y[:, 0, region],y_pred[:, 0, region]).numpy()
                                B=mean_absolute_error(y[:, 0, region],predB[:y_pred[:,0,0].shape[0],region]).numpy()
                                C=mean_absolute_error(y[:, 0, region],test_array[:y.shape[0],0,0]).numpy()
                                plt.title('MAE_pred'+str(A)+"_LRpred"+str(B)+"_naive"+str(C))

                                plt.legend(["actual",'forecast',"full_LinReg","single_LinReg", "naive"])                            
                                # if A.numpy() < B.numpy() and A.numpy() < C.numpy() and np.std(y) <np.mean(y)/2:
                                    # ppp = ppp + 1
                                # rrr=rrr+1
                                metric_df['ypred_'+str(i)]=A
                                metric_df['El_'+str(i)]=B
                                metric_df['naive_'+str(i)]=C
                                METS2=METS2.append(metric_df)
                                
                                RR=RR.replace(' ', '')
                                pathlib.Path(outdir+'/predR/'+RR+'/figs/').mkdir(parents=True, exist_ok=True)
                                pathlib.Path(outdir+'/predR/'+RR+'/data/').mkdir(parents=True, exist_ok=True)
                                
                                plt.title('_horiz:'+str(foretime)+'_var:'+str(len(covar))+'_gshuf:'+str(dist_mat_shuf)+'_dshuf:'+str(shufd)+'_adjcut'+str(graph_cutoff)+'mae'+str(A))
                                plt.savefig(outdir+'/predR/'+RR+'/figs/h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)+'mae'+str(A)+'.png')
                                plt.close()
                                
                                
                                outtt=outdir+'/predR/'+RR+'/data/h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)+'mae'+str(A)
                                # ccc=np.transpose(x_test, [0,3,2,1])
                                np.savez(outtt,y=y,ypred=y_pred,LR=predA,LR1=predB,naive=predC)#[:y.shape[0],:])

                            # print(str(ppp)+'/'+str(rrr)+'='+str(ppp/rrr))
                                
                                
                                # pathlib.Path(outdir+'/predR/').mkdir(parents=True, exist_ok=True)\
                                metric_B=pd.DataFrame(['ypred_'+str(i),'EL_'+str(i),'EL1_'+str(i),'naive_'+str(i)] for i in metrics)

                                for meth in ['mean','median','sum','min','max']:
                                    a=eval('np.'+meth)(y[:, 0, :],axis=1)
                                    b=eval('np.'+meth)(y_pred[:, 0, :],axis=1)
                                    c=eval('np.'+meth)(predB[:y_pred[:,0,0].shape[0],:],axis=1)
                                    e=eval('np.'+meth)(pd.DataFrame(PC),axis=0)[:y_pred[:,0,0].shape[0]]
                                    d=eval('np.'+meth)(test_array[:y.shape[0],:,0],axis=1)

                                    # for region in range(distances.shape[0]):
                                    plt.figure(figsize=(18, 6))
                                    plt.plot(a)
                                    plt.plot(b)#*np.mean(y)+10)
                                    plt.plot(c)
                                    plt.plot(e)
                                    plt.plot(d)

                                    A=mean_absolute_error(a,b).numpy()
                                    B=mean_absolute_error(a,c).numpy()
                                    C=mean_absolute_error(a,d).numpy()
                                    D=mean_absolute_error(a,e).numpy()
                                    plt.title("_LRpred"+str(B)+"_sLRpred"+str(D)+"_naive"+str(C)+'MAE_pred'+str(A))

                                    plt.legend(["actual",'forecast',"LinReg","single_LR", "naive1"])
                                    # plt.savefig(outdir+'/pred/'+meth+'_b4_2020_25drop.png')
                                    pathlib.Path(outdir+'/predR/'+RR+'/figs/'+meth+'/').mkdir(parents=True, exist_ok=True)
                                    pathlib.Path(outdir+'/predR/'+RR+'/data/'+meth+'/').mkdir(parents=True, exist_ok=True)

                                    plt.savefig(outdir+'/predR/'+RR+'/figs/'+meth+'/_h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)+'mae'+str(A)+'.png')
                                    
                                    metric_B['ypred_'+str(i)+str(meth)]=A
                                    metric_B['EL_'+str(i)+str(meth)]=B
                                    metric_B['EL1_'+str(i)+str(meth)]=C
                                    metric_B['naive_'+str(i)+str(meth)]=D
                                    METS3=METS3.append(metric_B)
                                    
                                    pathlib.Path(outdir+'/predR/'+RR+'/data/'+meth+'/').mkdir(parents=True, exist_ok=True)
                                    outtt=outdir+'/predR/'+RR+'/data/'+meth+'/h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)+'mae'+str(A)
                                    # ccc=np.transpose(x_test, [0,3,2,1])
                                    np.savez(outtt,y=a,ypred=b,naive=d,LR=c,LR1=e)#[:y.shape[0],:])

                                del metric_B
                                outtt=outdir+'/predR/'+RR+'/data/'+meth+'_'+'h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)
                                pathlib.Path(outtt).mkdir(parents=True, exist_ok=True)
                                METS3.to_csv(outtt+'/mae_'+str(A)+'_'+RR+'_'+meth+'_perf.txt',sep='\t')
                                
                            outtt=outdir+'/predR/'+RR+'/data/mae_h'+str(foretime[0])+'_v'+str(len(covar))+'_g'+str(dist_mat_shuf)+'_d'+str(shufd)+'_a'+str(graph_cutoff)
                            pathlib.Path(outtt).mkdir(parents=True, exist_ok=True)
                            METS2.to_csv(outtt+'/mae_'+RR+'perf.txt',sep='\t')
                        shuf_list.append(dist_mat_shuf)
                        pred_horizon_list.append(foretime[0])
                        var_list.append(covar)
                        dist_list.append(graph_cutoff)

                        graph_list.append(graph.num_nodes)
                        graph_edge_list.append(len(graph.edges[0]))
                        train_shape_list.append(train_array.shape)
                        val_shape_list.append(val_array.shape)
                        test_shape_list.append(test_array.shape)
                        shufd_list.append(shufd)


    df = pd.DataFrame(list(zip(shuf_list, pred_horizon_list, var_list,dist_list,  graph_list, graph_edge_list, train_shape_list, val_shape_list, test_shape_list,shufd_list)), #, mse_list, naive_mse_list,EL_mse_list,mape_list, naive_mape_list, EL_mape_list, mae_list, naive_mae_list, EL_mae_list)), 
       columns =['shuf','foretime','var','distTR','graph_shape','edges','train_shape','val_shape','test_shape','data_shuffle'])#,'mse','naive_mse','EL_mse','mape','naive_mape','EL_mape','mae','naive_mae','EL_mae'])
    
    # df=pd.merge(df,METS)
    df=pd.concat([df, METS.iloc[:,3:].drop_duplicates().reset_index().drop(columns='index')], axis=1)#,ignore_index=True)
    
    df.to_csv(outdir+'/GNN_perf.txt',sep='\t')

    plot_performance(outdir+'/GNN_perf.txt',outdir)

    print('All done!')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))