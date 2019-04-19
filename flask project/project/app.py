from flask import Flask
import pandas as pd
from flask import Flask, render_template, json, request
app = Flask(__name__)

_myFile='CBC.xls'

@app.route("/")
def main():
    return render_template("dashboard.html")

@app.route("/custList",methods = ['POST', 'GET'])
def custList():
    if request.method == 'POST':
      custList = request.form
      print(custList)
      for key, value in custList.items():
        if(key=='category'):
            _cat=value
            print(value)
            print(_cat)
        if(key=='myFile'):
            _myFile=value
    a=predictCustList(_cat)   
    print(a)
    return '<div class="col-sm"><p class="font-weight-bold" align="center" style="font-size:30px" >TARGET CUSTOMER LIST</p></div><div class="container"><div style ="background-color: #def4f7;" class="card border-primary mb-3" style="max-width:25rem;border-radius: 30px; font-size:30px" ><form method = "POST" action ="http://127.0.0.1:5000/#"><br><divclass="col-sm"><h1 class="display-1" align="center" ><p lign="center">'+a.to_html(escape=False)+'</p></h1></div></form> </div></div>'


 
@app.route('/viewCustDetails',methods = ['POST', 'GET'])
def viewCustDetails():

    if request.method == 'POST':
      viewCustDetails = request.form
      for key, value in viewCustDetails.items():
        if(key=='cid'):
            _cid=value
        if(key=='category'):
            _cat=value
        if(key=='myFile'):
            _myFile=value
    print(_cat)          
    a=viewCustomer(_cid)
    print(a)

    b=predictCust(_cat,_cid)
   
    return '<head><style>table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%}td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}tr:nth-child(even) {background-color: #dddddd;}</style></head><body><div class="container"><div style ="background-color: #def4f7;" class="card border-primary mb-3" style="max-width:25rem;border-radius: 30px; font-size:30px" ><form method = "POST" action ="http://127.0.0.1:5000/"><br><divclass="col-sm"><h1 class="display-1" align="center" ><p lign="center"><h4>'+b+'</h4><br><table><tr><th>Key</th><th>Value</th></tr><tr><td>ID</td><td>'+str(a['ID#'])+'</td></tr><tr><td>Gender</td><td>'+str(a['Gender'])+'</td></tr><tr><td>total_expenditure</td><td>'+str(a['M'])+'</td></tr><tr><td>months_since_last_purchase</td><td>'+str(a['R'])+'</td></tr><tr><td>number_of_purchases</td><td>'+str(a['F'])+'</td></tr><tr><td>months_since_first_purchase</td><td>'+str(a['FirstPurch'])+'</td></tr><tr><td>ChildrensBooks_purchased</td><td>'+str(a['ChildBks'])+'</td></tr><tr><td>YouthBooks_purchased</td><td>'+str(a['YouthBks'])+'</td></tr><tr><td>CookBooks_purchased</td><td>'+str(a['CookBks'])+'</td></tr><tr><td>DoityourselfBooks_purchased</td><td>'+str(a['DoItYBks'])+'</td></tr><tr><td>Dict_Encycl_Atlases_purchased</td><td>'+str(a['RefBks'])+'</td></tr><tr><td>ArtBooks_purchased</td><td>'+str(a['ArtBks'])+'</td></tr><tr><td>GeographyBooks_purchased</td><td>'+str(a['GeogBks'])+'</td></tr><tr><td>Secrets_Italian_Cooking</td><td>'+str(a['ItalCook'])+'</td></tr><tr><td>num_items_purchased_of_product_48</td><td>'+str(a['ItalAtlas'])+'</td></tr><tr><td>num_items_purchased_of_product_58</td><td>'+str(a['ItalArt'])+'</td></tr><tr><td>bought_art_history_of_florence</td><td>'+str(a['Florence'])+'</td></tr><tr> <td>History Related Purchase</td><td>'+str(a['Related Purchase'])+'</td></table></p></h1></div></form></div></div></body>'

 
@app.route('/viewNewPrediction',methods = ['POST', 'GET'])
def viewNewPrediction():
    if request.method == 'POST':
      viewNewPrediction = request.form
      for key, value in viewNewPrediction.items():
        if(key=='Monetary'):
            _Monetary=value
        if(key=='Recency'):
            _Recency=value
        if(key=='Frequency'):
            _fq=value
        if(key=='gender'):
            _gender=value
        if(key=='FirstPurch'):
            _fp=value
        if(key=='Related Purchase'):
            _RP=value
        if(key=='myFile'):
            _myFile=value
    b=predictCust1(_gender,_Monetary,_Recency,_fq,_fp,_RP)
   
    return '<head><style>table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%}td, th {border: 1px solid #dddddd;text-align: left;padding: 8px;}tr:nth-child(even) {background-color: #dddddd;}</style></head><body><div class="container"><div style ="background-color: #def4f7;" class="card border-primary mb-3" style="max-width:25rem;border-radius: 30px; font-size:30px" ><form method = "POST" action ="http://127.0.0.1:5000/"><br><divclass="col-sm"><h1 class="display-1" align="center" ><p lign="center"><h2>'+b+'</h2></p></h1></div></form></div></div></body>'

    
      
    


from sklearn.cross_validation import train_test_split

print(_myFile)
df =  pd.read_excel(_myFile,sheetname="DATA")
df.dropna()
df.dtypes    ##finding column types in table

correlations=df[['Gender','M','R','F','FirstPurch','Related Purchase','Florence']].corr()    

temp_df=df[['Gender','M','R','F','FirstPurch','Related Purchase','Florence']]
temp_df.dtypes
y=temp_df.iloc[:,6]  ## 5th column
x=temp_df.iloc[:,0:6]   ##0 to 4th column
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2) ## splitting data into train(0.8) and test data(0.2)
xtrain,xval,ytrain,yval=train_test_split(xtrain, ytrain, test_size=0.4375) ## spitting train data into 2:train and validation(0.4375from train)
from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(xtrain,ytrain) # default value for n_neighbors is 5
print(type(xtest))

def predictCust(categoryname,cid):
    custrow=viewCustomer(cid)
    d = {'Gender':[custrow['Gender']],'M': [custrow['M']], 'R': [custrow['R']], 'F':[custrow['F']], 'FirstPurch':[custrow['FirstPurch']], 'Related Purchase':[custrow[categoryname]] }
    listdf = pd.DataFrame(data=d)
    print(listdf)

    probs = model.predict_proba(listdf)
    print(probs)
    print(probs[:,1])
       
    if(probs[:,1][0]>=0.300772):
            return 'Probability score is:'+str(probs[:,1][0])+'<br>Customer Will Buy the Book' 
    elif(probs[:,1][0]<0.300772):
            return 'Probability score is:'+str(probs[:,1][0])+'<br>Customer Will not Buy the Book'
    else:
            return 'Customer is not in the data base'

def predictCust1(gender,M,R,F,FP,RP):
    d = {'Gender':[gender],'M': [M], 'R': [R], 'F':[F], 'FirstPurch':[FP], 'Related Purchase':[RP] }
    listdf = pd.DataFrame(data=d)
    print(listdf)
    probs = model.predict_proba(listdf)
    print(probs)
    print(probs[:,1])
       
    if(probs[:,1][0]>=0.300772):
            return 'Probability score is:'+str(probs[:,1][0])+'<br>Customer Will Buy the Book' 
    else:
            return 'Probability score is:'+str(probs[:,1][0])+'<br>Customer Will not Buy the Book'
    

def viewCustomer(cid):
      
    for index,row in df.iterrows():
        if(str (row['ID#']) in (cid) ):
              return row
        
            
    
    

def predictCustList(categoryname):
    listdf=df[['Gender','M','R','F','FirstPurch',categoryname]]
    print(listdf)

    probs = model.predict_proba(listdf)
    print(probs)
    print(probs[:,1])
    
    dfList=pd.DataFrame(columns=['Seq#','ID#','Gender','M','R','F','FirstPurch',categoryname,'Yes_Florence','No_Florence'])
    i=-1
    for row in probs[:,1]:
        i=i+1
        if(row>=0.300772):
            dfList=dfList.append(df.iloc[i,:])
    return dfList

#predict(xtest)

if __name__ == "__main__":
    app.run()