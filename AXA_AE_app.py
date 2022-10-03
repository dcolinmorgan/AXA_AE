from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import gzip,numpy as np, pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

px.set_mapbox_access_token('pk.eyJ1IjoiZGNvbGlubW9yZ2FuIiwiYSI6ImNsM2kwc3p2YjBhOGUzam1zOXdtenV0d2wifQ.R-SgXef7l-FI_zO7qYuQDQ')
app = Dash(__name__)

# f = gzip.GzipFile('/home/dcolinmorgan/mysite/coord_fulldiag_UVI.npy.gz', "r")
f = gzip.GzipFile('/home/dcolinmorgan/mysite/coord_fulldiag_UVI_min.npy.gz', "r")

data5=np.load(f,allow_pickle=True)
# data5=pd.read_csv('/content/drive/MyDrive/hku/AXA/loc_coord_diag.txt',sep='\t')
data5=pd.DataFrame(data5,columns=['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'lat',
       'long', 'name', 'year', 'week', 'diag1', 'UVI'])

# del data5['loc1_x'],data5['loc1_y'],data5['kmeans{k}'],data5['date']#,data5['name']
# data5['year']=pd.to_datetime(data5['year'], format='%Y')
data5['weekA']=data5['week']+(52*(data5['year']-np.nanmin(data5['year'])))
CC=pd.DataFrame(data5.groupby(['pm25','pm10','o3','no2','so2','co','lat','long','name','year','week','UVI','weekA']).agg('diag1').count()).reset_index()
data6=data5.melt(['lat','long','year','week','name','diag1','weekA'])
data6=data6[data6.value>0]
data6['diag2']=data6['diag1'].str.split('(').str[1].str.split(')').str[0]
data6['diag2']=data6['diag2'].astype(str)
data7=data6.groupby(['lat','long','year','name','weekA']).count()
data7.reset_index(inplace=True)
data7['variable']='LUNG'
del data7['week'], data7['diag1'],data7['diag2']
DD=data7.append(data6[['lat','long','name','weekA','year','variable','value']])
DD=DD[~DD.duplicated(keep='first')]
EE=data7.merge(data6[['lat','long','name','year','weekA','variable','value']],on=['lat','long','name','year','weekA'])
EE[~EE.duplicated(keep='first')]
del EE['variable_x']
EE.rename(columns={'value_x':'Lung','variable_y':'varaible','value_y':'value'},inplace=True)
# print(EE)

app.layout = html.Div([
###buttons/interactivity
    html.Div([

        dcc.Slider(
            data6['year'].min(),
            data6['year'].max(),
            step=None,
            id='slider',
            value=data6['year'].min(),
            marks={str(year): str(year) for year in DD['year'].unique()}
        ),#, style={'width': '49%', 'padding': '0px 20px 20px 20px'})
    # ])

        dcc.Dropdown(
              DD['variable'].unique(),
              'pm10',
              id='dropdown'
          )
    ]),
###layout form
    html.Div([
            dcc.Graph(id='graph0'),
            dcc.Graph(
                id='graphA'
                # hoverData={'points': [{'customdata': 'pm25'}]}
            )],style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
            dcc.Graph(id='graphB'),
            dcc.Graph(id='graphC')],
         style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}
        )


])
### interactive data query
def get_data(data6,year_value,variable_value):
    return data6[(data6['year']==year_value)&(data6['variable']==variable_value)]

### individual plot layouts with same input from query function
@app.callback(
    Output("graph0", "figure"),
    Input("slider", "value"),
    Input("dropdown", "value"))
def plottt_geo(year_value,variable_value):
    df=get_data(data6,year_value,variable_value)
    df=df.groupby(['lat','long','year','week','name','variable']).count().reset_index()
    fig=ff.create_hexbin_mapbox(
      data_frame=df, lat="lat", lon="long",
      nx_hexagon=10, opacity=0.9, labels={"color": variable_value},
      color="diag1", agg_func=np.sum, color_continuous_scale="Icefire")
    # fig = px.density_mapbox(data6, lat='lat', lon='long', z='value', radius=10,
    #      center=dict(lat=0, lon=180), zoom=0,
    #      mapbox_style="stamen-terrain")
    return fig #.show()

@app.callback(
    Output("graphA", "figure"),
    Input("slider", "value"),
    Input("dropdown", "value"))
def plottt_line(year_value,variable_value):
    df=get_data(DD,year_value,variable_value)

    fig = px.line(data_frame=df, x='weekA', y='value',color='name')
    return fig #.show()

@app.callback(
    Output("graphB", "figure"),
    Input("slider", "value"),
    Input("dropdown", "value"))
def plottt_diag(year_value,variable_value):
    df=get_data(data6,year_value,variable_value)
    # fig = px.line(data_frame=df, x='week', y='value',color='name')
    fig = px.bar(data_frame=df, x='diag2', y='value',color='name')
    return fig #.show()

@app.callback(
    Output("graphC", "figure"),
    Input("slider", "value"),
    Input("dropdown", "value"))
def plottt_scatter(year_value,variable_value):
    # df=get_data(CC,year_value,variable_value)
    # fig = px.scatter(data_frame=df, x='Lung', y='value',color='name')
    fig = px.scatter(data_frame=CC[CC['year']==year_value], y='diag1', x=variable_value,color='name')
    return fig #.show()

# if __name__ == '__main__':
# app.run_server()
