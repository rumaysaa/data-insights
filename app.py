from flask import Flask,render_template,request,redirect,session
from flask_session import Session
import numpy as np,json,csv
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,roc_curve, auc
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,mean_absolute_percentage_error
from modules.mul_logistic import Mlor
from modules.simple_logistic import Slor
from modules.knnScratch_module import KnnS
from modules.simple_linear import SLinearReg
from modules.ranForest_module import randomForest
from modules.decTree_module import decision_tree
from modules.nBayes_module import naiveBayes
from modules.svm_module import Svm
from modules.mul_linear import Mlir
import plotly.figure_factory as ff


app = Flask(__name__)
app.secret_key = 'BAD_SECRET_KEY'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

#--------get--------------

@app.route('/',methods=['GET'])
def index():
    session.clear()
    return render_template('index.html')

@app.route('/error',methods=['GET'])
def error():
    return render_template('error.html')

#---------post-------------

@app.route('/file',methods=[ 'POST','GET'])
def get_file():
    f = request.files['file']
    if(len(f.read()) == 0):
        return render_template('index.html', msg="Upload valid file.\n File should not be empty")
    f.seek(0)
    df = pd.read_csv(f)
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    prep_f = pre_process(df)
    f=prep_f.to_csv(index=False)
    csv_dicts = [{k: v for k, v in row.items()} for row in csv.DictReader(f.splitlines(), skipinitialspace=True)]
    session['csvfile'] = json.dumps(csv_dicts)
    model = request.form.get('selectModel')
    prep_f.to_csv(os.path.join('files','csvfile.csv'),index=False)
    new_df=pd.read_csv('files/csvfile.csv')
    if(len(df.columns) < 2):
        return render_template('index.html', msg="Upload valid file.\n File should contain atleast 2 columns")
    corht = corr_heatmap(df)
    colnames = new_df.columns.to_list()
    gd=None
    if(model =='Simple Linear Regression S'):
        gd = ['Batch', 'Stochastic', 'Mini Batch']
    return render_template('selection.html', options=colnames, modl= model,gd=gd,corr_html=corht)

@app.route('/analysis',methods=['POST','GET'])
def analysis():
    req = request.form
    print(req)
    model = req.get("modelname")
    y = req['y']
    df=pd.read_csv('files/csvfile.csv')
    gd=batchsize=None
    if('selectGD' in req):
        gd = req['selectGD']
        if(gd == 'Mini Batch'):
            batchsize = int(req['batchsize'])
    dropx=[]
    for i in req:
        if(i[0]=='x' and (req[i] in df)):
            dropx.append(req[i])
    df = df.drop(dropx,axis=1)
    df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
    df.to_csv(os.path.join('files','csvfile.csv'),index=False)
    df = pd.read_csv('files/csvfile.csv')
    correlated_columns = corr(df,y)
    cols=df.columns.to_numpy()
    if(len(correlated_columns) == 0):
        xi = df.drop([y], axis=1).to_numpy()
        x_cols = np.delete(cols, np.where(cols == y))
    else:
        xi = df[correlated_columns].to_numpy()
        x_cols = correlated_columns
    #print("xi",xi)
    yi = df[y].to_numpy()
    prediction = None
    scattplot_html=None
    accuracy=report=plotmet_html=sig_html=errors=rocplot_html=GD_plot=lr=epoch=n_neighbors=scattplot_html= None
    if(model == 'Multiple Logistic Regression'):
        model_ml = Mlor()
        model_ml.fit(xi,yi)
        accuracy,report,plotmet_html,rocplot_html = mul_log(model_ml)
        if(x_cols[0] in req):
           prediction = predict(model_ml,x_cols,req)
    if(model == 'Multiple Linear Regression'):
        model_mli = Mlir()
        model_mli.fit(xi,yi)
        errors,scattplot_html = mul_lin(model_mli)
        if(x_cols[0] in req):
            prediction = predict(model_mli,x_cols,req)
    if(model == 'Simple Logistic Regression S'):
        model_sl = Slor()
        model_sl.fit(xi,yi)
        accuracy,report,plotmet_html,sig_html = sim_log(model_sl)
        if(x_cols[0] in req):
            prediction = predict_singl(model_sl,x_cols,req)
    if(model == 'KNN S'):
        n_neighbors = int(req['k'])
        model_knn = KnnS(n_neighbors)
        model_knn.fit(xi,yi)
        accuracy,report,plotmet_html,scattplot_html = knnS(model_knn)
        if(x_cols[0] in req):
            prediction = predict(model_knn,x_cols,req)  
    if(model == 'Simple Linear Regression S'):
        lr = float(req['LR'])
        epoch = int(req['epoch'])
        model_lir = SLinearReg(epoch,lr,gd,batchsize)
        model_lir.fit(xi,yi.reshape(-1,1))
        errors,scattplot_html,GD_plot = sim_lin(model_lir)
        #print(correlated_columns[0] in req)
        if(x_cols[0] in req):
            prediction = predict_singl(model_lir,x_cols,req)
    if(model == 'Random Forest classifier'):
        rf_clf = randomForest()
        rf_clf.fit(xi,yi)
        accuracy,report,plotmet_html,rocplot_html = randomforest(rf_clf)
        if(x_cols[0] in req):
            prediction = predict(rf_clf,x_cols,req)    
    if(model == 'Decision tree classifier'):
        dt_clf = decision_tree()
        dt_clf.fit(xi,yi)
        accuracy,report,plotmet_html,rocplot_html = decisiontree(dt_clf)
        if(x_cols[0] in req):
            prediction = predict(dt_clf,x_cols,req) 

    if(model == 'Naive bayes classifier'):
        nb_clf = naiveBayes()
        nb_clf.fit(xi,yi)
        accuracy,report,plotmet_html,rocplot_html = naivebayes(nb_clf)
        if(x_cols[0] in req):
            prediction = predict(nb_clf,x_cols,req)

    if(model == 'Support Vector Machine'):
        sv_clf = Svm()
        sv_clf.fit(xi,yi)
        accuracy,report,plotmet_html,rocplot_html = svm(sv_clf)
        if(x_cols[0] in req):
            prediction = predict(sv_clf,x_cols,req)
    col1,col2 = np.array_split(x_cols,2)
    return render_template('analysis.html',plotmet_html=plotmet_html, acc=accuracy,tables=report,lr=lr,epoch=epoch,cols1=col1,cols2=col2,model=model,y=y,gd=gd,batchsize=batchsize,pred=prediction,scattplot=scattplot_html,err=errors,roc=rocplot_html,GD_plot=GD_plot,sig_html=sig_html,k=n_neighbors)


#--------functions-----------

def corr(df,y):
    correlations = df.corr()[y]
    correlation_threshold = 0.1
    correlated_columns = [col for col, corr in correlations.items() if abs(corr) > correlation_threshold]
    correlated_columns.remove(y)
    return correlated_columns

def pre_process(df):
    label_encoder = LabelEncoder()
    for column in df.columns:
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])
    df = df.ffill()
    df = df.interpolate()
    return df

def corr_heatmap(df):
    correlation_matrix = df.corr()
    custom_colorscale = [[0, 'lightblue'], [0.5, 'lightred'], [1, 'lightgreen']]
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                x=correlation_matrix.columns,
                                y=correlation_matrix.index,
                                colorscale='magenta',
                                colorbar=dict(title='Correlation')))
    # Add text annotations with the correlation values
    for i, row in enumerate(correlation_matrix.index):
        for j, col in enumerate(correlation_matrix.columns):
            fig.add_annotation(text=f"{correlation_matrix.iloc[i, j]:.2f}",
                           x=col, y=row,
                           xref='x1', yref='y1',
                           showarrow=False)
    return fig.to_html(full_html=False)

def calculate_error_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [["Mean-squared-error",mse], ["Root-mean-squared-error",rmse], ["Mean-absolute-error",mae], ["R-squared",r2]]

def calculate_accuracy(y_test,y_pred):
    return accuracy_score(y_test, y_pred)*100

def classifi_report_to_html(y_test, y_pred):
    report = classification_report(y_test, y_pred,output_dict=True)
    df = pd.DataFrame(report).transpose()
    html = df.to_html()   
    html = html.replace('class="dataframe"', 'class="table table-hover "')
    html = html.replace('table border="1"','table border="0"')
    html = html.replace('tr style="text-align: right;"','tr style="text-align: left;"')
    return [html]

def plot_conf_metrics(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    unique_labels = sorted(set(y_test) | set(y_pred))
    fig1 = ff.create_annotated_heatmap(z=cm,x=unique_labels, y=unique_labels, colorscale='Viridis')
    plotmet_html = fig1.to_html(full_html=False)
    return plotmet_html

def plot_roc_curve(y_test, y_probs):
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    # Create ROC curve plot
    fig = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
    fig.update_layout(showlegend=True)
    # Convert the Plotly figure to HTML
    return fig.to_html(full_html=False)

def scatterplot(x_test,y_test,y_pred):
    fig = px.scatter(x_test, x=0, y=1,color=y_pred, color_continuous_scale='RdBu',symbol=y_test, labels={'symbol': 'label', 'color': 'score of <br>first class'})
    fig.update_traces(marker_size=12, marker_line_width=1.5)
    fig.update_layout(legend_orientation='h')
    return fig.to_html(full_html=False)

def predict(model,correlated_columns,req):
        new_x=[]
        for i in correlated_columns:
            if(i in req):
                new_x.append(float(req[i]))
        print(new_x)
        new_x = np.array(new_x).reshape(1,-1)
        print(new_x)
        prediction = model.predict(new_x)
        return prediction[0]

def predict_singl(model,correlated_columns,req):
        for i in correlated_columns:
            new_x= float(req[i])
        new_x = np.array(new_x)
        prediction = model.predict(new_x)
        print(prediction)
        return prediction

#--------model functions--------

def mul_log(obj):
    y_pred = obj.predict(obj.x_test)
    y_test =obj.y_test
    rocplot_html = None
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    y_probs = obj.predict_proba(obj.x_test)
    if (y_pred.max()<2 and y_test<2):
        rocplot_html = plot_roc_curve(y_test,y_probs)
    return accuracy,report,plotmet_html,rocplot_html

def mul_lin(obj):
    y_test = (obj.y_test).flatten()
    x_train,x_test = (obj.x_train).flatten(),(obj.x_test).flatten()
    y_train = (obj.y_train).flatten()
    y_pred = (obj.predict(obj.x_test)).flatten()
    x_pred = (obj.predict(obj.x_train)).flatten()
    errors = [calculate_error_metrics(y_test,y_pred)]
    scat_fig = go.Figure([
        go.Scatter(x=x_train, y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=x_test, y=y_test, 
                   name='test', mode='markers'),
        #go.Scatter(x=x_train,y=y_pred,name='prediction')
    ])
    return errors,scat_fig.to_html(full_html=False)


def sim_log(obj):
    y_pred = (obj.predict(obj.x_test)).astype(int).flatten()
    print(obj.m,obj.c)
    y_test = (obj.y_test).astype(int)
    rocplot_html = None
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=(obj.x_train).flatten(), y=y_pred, name='Sigmoid'))
    fig.update_layout(title='Sigmoid Function',
                  xaxis_title='z',
                  yaxis_title='Sigmoid(z)')
    sig_html = fig.to_html(full_html=False)
    if(y_pred.max()<2  and y_test<2):
        rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,sig_html

def sim_lin(obj):
    y_test = (obj.y_test).flatten()
    x_train,x_test = (obj.x_train).flatten(),(obj.x_test).flatten()
    y_train = (obj.y_train).flatten()
    x_pred = (obj.predict(obj.x_test)).flatten()
    errors = None#[calculate_error_metrics(y_test,y_pred)]
    scat_fig = go.Figure([
        go.Scatter(x=x_train, y=y_train, 
                   name='train', mode='markers'),
        go.Scatter(x=x_test, y=y_test, 
                   name='test', mode='markers'),
        #go.Scatter(x=x_test,y=y_pred,name='prediction')
    ])
    cost_values = np.array(obj.cost)
    iterations = list(range(obj.epochs))
    # Plot the cost function using the provided values
    trace_cost = go.Scatter(x=iterations, y=cost_values, mode='lines', name='Cost Function')
    # Create a layout for the plot
    layout = go.Layout(title='Cost vs Iteration', xaxis=dict(title='Iteration'), yaxis=dict(title='Cost'))
    GD_fig = go.Figure(data=[trace_cost], layout=layout)
    return errors,scat_fig.to_html(full_html=False),GD_fig.to_html(full_html=False)

def knnS(obj):
    y_pred = obj.predict(obj.x_test)
    y_test =obj.y_test
    rocplot_html = None
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    scattplot_html = scatterplot(obj.x_test,y_test,y_pred)
    if (y_pred.max()<2  and y_test<2):
        rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,scattplot_html,rocplot_html

def randomforest(obj):
    y_pred = obj.predict(obj.x_test)
    print(obj.x_test,y_pred)
    y_test = obj.y_test 
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    rocplot_html = sig_html=None
    if (y_pred.max()<2  and y_test<2):
            rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,rocplot_html

def decisiontree(obj):
    y_pred = obj.predict(obj.x_test)
    y_test = obj.y_test 
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    rocplot_html = None
    if (y_pred.max()<2  and y_test<2):
        rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,rocplot_html

def naivebayes(obj):  
    y_pred = obj.predict(obj.x_test)
    y_test = obj.y_test 
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    rocplot_html = None
    if (y_pred.max()<2  and y_test<2):
        rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,rocplot_html

def svm(obj):
    y_pred = obj.predict(obj.x_test)
    y_test = obj.y_test 
    accuracy = calculate_accuracy(y_test, y_pred)
    report = classifi_report_to_html(y_test, y_pred)
    plotmet_html = plot_conf_metrics(y_test,y_pred)
    rocplot_html = None
    if (y_pred.max()<2  and y_test<2):
            rocplot_html = plot_roc_curve(y_test,y_pred)
    return accuracy,report,plotmet_html,rocplot_html

if __name__ == "__main__":
    app.run(host='localhost', port=3000,debug=True)



