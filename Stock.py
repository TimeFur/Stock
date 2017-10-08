import os,sys

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

from datetime import *
import re
import requests
from bs4 import BeautifulSoup
import copy

from collections import OrderedDict
from grs import Stock
from grs import TWSENo

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import sklearn as sk

'''===========================
        Log Decorate
==========================='''
def showlog(func):
    def d_f(*argv):
        result=func(*argv)
        for i in result:
            print i
    
        return result
    return d_f

class StockObj():
    def __init__ (self, get_month):
        print "Stock"

        self.stockdata=OrderedDict()
        self.today=datetime.now().strftime("%Y/%m/%d")
        self.get_month=get_month

        self.OutputData=OrderedDict()
        self.InputData=OrderedDict()

        self.machine=OrderedDict()


    def showStocknum(self):
        stock_all=[]

        twse_no = TWSENo()
        for i in twse_no.all_stock_no:
            tmp_dict = {}
            tmp_dict["id"] = i
            tmp_dict["name"] = twse_no.searchbyno(i)[i]
            stock_all.append(tmp_dict)
            #print i,twse_no.searchbyno(i)[i]

        return stock_all

    def stock_url_parse(self, stock_id, month):

        url = "http://www.twse.com.tw/exchangeReport/STOCK_DAY"        
        str_month = lambda d : ('0' + str(d.month)) if d.month < 10 else str(d.month)
        dateinfo = datetime.now()
        result_list = []
        
        for i in range(month + 1):
            #set to the month first data
            dateinfo = dateinfo - timedelta(days = dateinfo.day - 1)        
            date = str(dateinfo.year) + str_month(dateinfo) + '01'

            #get the json
            payload = {'response' : 'json',
                       'date' : date,
                       'stockNo' : stock_id}
            r = requests.get(url, params = payload)
            try:
                #get the raw data
                json_result = r.json()
                for rawdata in json_result['data']:
                    result_list.append(rawdata)
                #print json_result['data']

                #move to the previous month
                dateinfo = dateinfo - timedelta(days = 1)
            except:
                pass
        #print result_list
        return result_list
    
    def stockGet(self, s_num, month):

        stocklist = self.stock_url_parse(s_num, month)
        time_list=[]
        stock_list=[]

        self.stockdata[s_num]=OrderedDict()

        for i in stocklist:

            time_obj=re.sub(r"\d+/",str(int(i[0][:i[0].find('/')])+1911)+"/",i[0],1)
            time=datetime.strptime(time_obj,"%Y/%m/%d")
            if i[6] != '--':
                if i[6].find(',') > 0:
                    i[6] = i[6].replace(',', '')
                self.stockdata[s_num][time_obj]=float(i[6])

                time_list.append(time)
                stock_list.append(i[6])

        #plt.plot(time_list,stock_list)
        #plt.show()

        #for stock_num in self.stockdata:
        #        for  d in self.stockdata[stock_num]:
        #                print d,self.stockdata[stock_num][d]


    def buyStock(self, s_num, date, count):    #("2330","2016/12/12",3)
        if self.stockdata.has_key(s_num):
            pay=float(self.stockdata[s_num][date])*count*1000
            print "You have to pay:",pay
        else:
            self.stockGet(s_num,3)
            pay=float(self.stockdata[s_num][date])*count*1000
            print "You have to pay:",pay

    def cal_BuyorNotbuy(self, s_num, earn, countday):

        if self.OutputData.has_key(s_num) == False:
            self.OutputData[s_num]=OrderedDict()

        #--------Check Data retrieve & parameter reasonable
        if self.stockdata.has_key(s_num) == False:
            self.stockGet(s_num, self.get_month)

        list_key = self.stockdata[s_num].keys()

        #--------Eval
        for i, date in enumerate(list_key):
            Earn_Flag = False
            if i + countday == len(list_key):
                break

            for d_index in range(i + 1, i + countday + 1):
                diff = float(self.stockdata[s_num][list_key[d_index]]) - float(self.stockdata[s_num][date])
                if diff >= earn:
                    #print "Earn money at %s, value=%s" % (list_key[d_index], self.stockdata[s_num][list_key[d_index]])
                    Earn_Flag=True

            if Earn_Flag == True:
                self.OutputData[s_num][date]=1
            else:
                self.OutputData[s_num][date]=0

        return self.OutputData

    def cal_KDBox(self,s_num,calday):

        K=50.0
        D=50.0

        if self.InputData.has_key(s_num) == False:
            self.InputData[s_num]=OrderedDict()

        #============Check date
        if self.stockdata.has_key(s_num)==False:
            self.stockGet(s_num,self.get_month)

        list_key=self.stockdata[s_num].keys()
        if(len(list_key)<calday):
            #yield "Calday is too long, list len=%d" % (len(list_key))
            return False

        #============Evalaute KD
        count_day=0
        RSV=0
        ex_RSV=0

        while(len(list_key) > calday+count_day):
            max_stock=0
            min_stock=0xFFFF
            avg_v=0.0
            for i in range(calday):
                v=float(self.stockdata[s_num][list_key[i+count_day]])
                avg_v = avg_v + v
                if v>max_stock:
                    max_stock=v
                if v<min_stock:
                    min_stock=v

            avg_v = avg_v/calday

            if(max_stock!=min_stock):
                RSV=((self.stockdata[s_num][list_key[calday+count_day]]-min_stock)/(max_stock-min_stock))*100.0
            else:
                RSV=ex_RSV
            ex_RSV=RSV
            now_K=K*(2.0/3.0)+RSV*(1.0/3.0)
            now_D=D*(2.0/3.0)+now_K*(1.0/3.0)

            K=now_K
            D=now_D

            #yield "Date:%s CurStock=%f, RSV=%s, K=%f D=%f (min=%f, max=%f)" % (list_key[calday+count_day],self.stockdata[s_num][list_key[calday+count_day]], RSV, now_K, now_D, min_stock, max_stock),

            if self.InputData[s_num].has_key(list_key[calday+count_day]) == False:
                self.InputData[s_num][list_key[calday+count_day]] = OrderedDict()

            self.InputData[s_num][list_key[calday+count_day]]["CurStock"] = self.stockdata[s_num][list_key[calday+count_day]]
            self.InputData[s_num][list_key[calday+count_day]]["RSV"] = RSV
            self.InputData[s_num][list_key[calday+count_day]]["K"] = now_K
            self.InputData[s_num][list_key[calday+count_day]]["D"] = now_D
            self.InputData[s_num][list_key[calday+count_day]]["Avg"] = avg_v

            count_day=count_day+1

        return self.InputData

    def cal_RSIBox(self, Stock_id, calday):
        if self.InputData.has_key(Stock_id) == False:
            self.InputData[Stock_id]=OrderedDict()

        #============Check date
        if self.stockdata.has_key(Stock_id) == False:
            self.stockGet(Stock_id,self.get_month)

        list_key = self.stockdata[Stock_id].keys()
        if(len(list_key)<calday):
            print "Calday is too long, list len=%d" % (len(list_key))
            return False

        #=================
        count=0
        while(len(list_key)>calday+count):
            pos_sum = 0
            pos_per = 0
            neg_sum = 0
            neg_per = 0
            for i in range(calday):
                val1=self.stockdata[Stock_id][list_key[i+count]]
                val2=self.stockdata[Stock_id][list_key[i+count+1]]

                if val2 >= val1:
                    pos_sum += val2 - val1
                else:
                    neg_sum += val1 - val2

            pos_per = float(pos_sum) / float(calday)
            neg_per = float(neg_sum) / float(calday)

            if self.InputData[Stock_id].has_key(list_key[calday+count]) == False:
                self.InputData[Stock_id][list_key[calday+count]] = OrderedDict()
            if pos_per!=0 or neg_per!=0:
                RSI = pos_per/(pos_per + neg_per)
            else:
                RSI=0

            self.InputData[Stock_id][list_key[calday+count]]["Pos_per"]=pos_per
            self.InputData[Stock_id][list_key[calday+count]]["Neg_per"]=neg_per
            self.InputData[Stock_id][list_key[calday+count]]["RSI"]=RSI

            count+=1
        return self.InputData

class ClassifyObj():

        def __init__ (self, Input, Output):
                print "Classifier"

                self.InputData=Input
                self.OutputData=Output

                self.machine = OrderedDict()

                self.c_names = ["Nearest Neighbor",
                         "Linear SVM", "RBF SVM", "Decision SVM",
                         "Decision Tree",
                         "Random Forest",
                         "AdaBoost",
                         "Naive Bayes",
                         "Linear Discriminant Annlysis",
                         "Quadratic Discriminant Analysis"]
                self.classifiers = [KNeighborsClassifier( 3 ),
                                   SVC( kernel ="linear", C = 0.025 ), SVC(gamma = 2, C = 1),
                                   DecisionTreeClassifier( max_depth = 5),
                                   RandomForestClassifier( max_depth = 5, n_estimators = 10, max_features = 1),
                                   AdaBoostClassifier(),
                                   GaussianNB(),
                                   LinearDiscriminantAnalysis(),
                                   QuadraticDiscriminantAnalysis()]

        def training_each_classifiers(self, Stock_id):
                done_flag = 0
                for n, mlp in zip(self.c_names, self.classifiers):
                        print "==================%s===============" % (n)
                        try:
                                fitting_result = self.Train(Stock_id, mlp, n)
                                if fitting_result == 0:
                                        done_flag = 0
                                        print "No need to train"
                                        break
                                elif fitting_result == 1:
                                        done_flag = 1
                        except:
                                print "==========Error========%s" % (n)
                return done_flag

        def Train(self, Stock_id, mlp = None, mlp_name = None):
                list_input_data=[]
                list_output_data=[]

                for day in self.OutputData[Stock_id]:
                    if self.InputData[Stock_id].has_key(day) == True:

                        #Get Input Data
                        tmp_input=[]
                        for k in self.InputData[Stock_id][day]:
                            tmp_input.append(self.InputData[Stock_id][day][k])
                        list_input_data.append(tmp_input)

                        #Get Output Data
                        list_output_data.append(self.OutputData[Stock_id][day])

                        #print day, self.InputData[Stock_id][day], self.OutputData[Stock_id][day]

                sample_length=len(list_input_data)
                train_len = sample_length*(2.0/3.0)
                train_len = int(train_len)

                Train_X = np.array(list_input_data[:train_len])
                Train_Y = np.array(list_output_data[:train_len])

                Test_X = np.array(list_input_data[train_len:])
                Test_Y = np.array(list_output_data[train_len:])

                #print "Training sample number=%d, Total sample = %d" % (train_len, sample_length)
                if len(list_output_data) == list_output_data.count(0) or len(list_output_data) == list_output_data.count(0):
                    print "list_output_data is zero"
                    return 0
                #=============================================Learning & Training
                if mlp == None:
                    mlp = MLPClassifier( hidden_layer_sizes=(20,20), max_iter=100, alpha=1e-4,
                                                                        solver='lbfgs', verbose=10, tol=1e-6, random_state=1,
                                                                        learning_rate_init=.1)
                if mlp_name == None:
                    mlp_name = "Neural Network"

                mlp.fit(Train_X,Train_Y)

                #print "MLP Loss=",mlp.loss_

                #print "Buy count=",list_output_data.count(1)
                #print "Not Buy count=",list_output_data.count(0)

                TrainScore = mlp.score(Train_X,Train_Y)
                TestScore = mlp.score(Test_X,Test_Y)
                print "Score Training=",TrainScore
                print "Score Test=",TestScore

                self.machine[Stock_id]=OrderedDict()
                self.machine[Stock_id][mlp_name] = OrderedDict()

                self.machine[Stock_id][mlp_name]["machine"]=mlp
                self.machine[Stock_id][mlp_name]["TrainScore"]=TrainScore
                self.machine[Stock_id][mlp_name]["TestScore"]=TestScore

                return 1

        def predict_each_classifiers(self, Stock_id, day):
                Buy_list = []
                buy_flag = 0
                for n, mlp in zip(self.c_names, self.classifiers):
                        print "==================%s===============" % (n)
                        try:
                                predict_result = self.predict(Stock_id, day, n)
                                if predict_result[0] == 1:
                                        buy_flag = 1
                                        Buy_list.append({"id":Stock_id})
                        except:
                                print "==========Error========%s" % (n)

                return Buy_list

        def predict(self, Stock_id, day, mlp_name = None):
            if self.machine.has_key(Stock_id) == False:
                print "No this Stock Data"
                return False
            if self.machine[Stock_id].has_key(mlp_name) == False:
                print "No this machine"
                return False
            if mlp_name == None:
                mlp_name = "Neural Network"
            #=============================================Predict
            predict=[]
            if self.InputData[Stock_id].has_key(day) == True:
                for k in self.InputData[Stock_id][day]:
                    predict.append(self.InputData[Stock_id][day][k])
                predict_array=np.array(predict)
                predict_array=predict_array.reshape(1, -1) # Trasfer to sigle sample pattern
                print self.machine[Stock_id][mlp_name]["machine"].predict(predict_array)
                return self.machine[Stock_id][mlp_name]["machine"].predict(predict_array)
            else:
                print "There is no stock this day %s" % (day)

def StockFlow(_trainingmonth, _predictdate, _RSIcaldate, _KDcaldate):

    training_month = _trainingmonth
    predict_date = _predictdate
    RSI_caldate = _RSIcaldate
    KD_caldate = _KDcaldate

    s=StockObj(get_month = training_month) #Get stock raw data

    Buy_list=[]
    stock_all_no=[]

    with open("Stock_id",'rb') as stockfile:

        #Get Stock id to insert to stock_all_no dict
        stock_list=stockfile.read().split()
        for i in stock_list:
            if len(i) == 4:
                stock_all_no.append({"id":i})

        #Stock Info set (Input/Output),
        #Input=>Evaluate parameter
        #Output=>Evalaute buy/Not buy
        for stock_list in stock_all_no:

            Stock_id=stock_list['id']
            print "Get %s Stock Data" % (Stock_id)
            #Input & Output
            s.cal_RSIBox(Stock_id, RSI_caldate)
            s.cal_KDBox(Stock_id, KD_caldate)
            Input = s.InputData
            Output = s.cal_BuyorNotbuy(Stock_id, earn=10, countday=14)
        print "------------------Get done-----------------------"
                
        #Classifier Training
        classifier_machine = ClassifyObj(s.InputData, s.OutputData)
        #r = all(value == 0 for value in s.OutputData.values())
        #print r
        for stock_list in stock_all_no:
            Stock_id = stock_list['id']
            #--------------------Training

            classifier_machine.Train(Stock_id)
            '''
            done_flag = classifier_machine.training_each_classifiers(Stock_id)
            if done_flag == 0:
                continue
            '''
            #--------------------Predict for date

            result = classifier_machine.predict(Stock_id, predict_date)
            print result
            if result == 1:
                Buy_list.append({"id":Stock_id})
            '''
            Buy_list = classifier_machine.predict_each_classifiers(Stock_id, predict_date)
            '''

        #=========================================================Show Result
        result_file = open(re.sub("/", "-", predict_date)+".txt",'wb')
        result_file.write(predict_date+'\n')
        for buy_stock in Buy_list:
            print "These can buy"
            print "Stock id %s, Training Score=%f, Testing Score=%f" % (    buy_stock['id'],
                                                                                        classifier_machine.machine[buy_stock['id']]["TrainScore"],
                                                                                        classifier_machine.machine[buy_stock['id']]["TestScore"])
            result_file.write(buy_stock['id']+',')
            result_file.write(str(s.stockdata[buy_stock['id']][predict_date])+',')
            result_file.write(str(classifier_machine.machine[buy_stock['id']]["TrainScore"])+',')
            result_file.write(str(classifier_machine.machine[buy_stock['id']]["TestScore"])+'\n')
        result_file.close()

def main():
    training_month = 12
    predict_date = "2017/09/12"
    RSI_caldate = 9
    KD_caldate = 9
    '''
    s = StockObj(3)
    s.stockGet("0059", 12)
    print s.stockdata
    '''
    StockFlow(_trainingmonth = training_month,
                _predictdate = predict_date,
                _RSIcaldate = RSI_caldate,
                _KDcaldate = KD_caldate)
    
if __name__=="__main__":
    main()



