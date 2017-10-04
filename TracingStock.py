import os, sys

import numpy as np
from numpy import *

from collections import OrderedDict
from grs import Stock
from grs import TWSENo
import re

import cPickle as pickle
import matplotlib.pyplot as plt

def readStockdata(filename):
  result_list = []
  buy_date=''
  
  readobj = open(filename, 'rb')
  content = readobj.read()
  readobj.close()

  rowlist = content.split('\n')
  
  for i,row in enumerate(rowlist):
    if i==0:
      buy_date = row.replace("/","-")
    if i != 0 and row.strip():
      result_list.append(map(eval, row.split(',')))
  
  return buy_date, mat(result_list)


def calProb(stockmap):
  buylist=[]
  win_prob = 1.0
  lose_prob = 1.0
  
  for r in stockmap:
    if r[0,3] > 0.85 and r[0,1]!=0:
      buylist.append(r)
      win_prob = win_prob * r[0,3] 
      lose_prob = lose_prob * (1.0 - r[0,3]) 

  #print buylist
  #print "Win:",win_prob
  #print "Lose:",lose_prob
  return buylist

def Tracing(buy_date, stockmap):  #row[Stock_id, Stock dollar $, Training Probility, Test Probility]

  Earn = 0
  today = ""
  show_list=[]
  
  filename="Tracing_"+buy_date.replace("-","")+'.txt'
  
  fd = os.open(filename, os.O_RDWR | os.O_CREAT)
  os.close(fd)
  
  for row in stockmap:
    stock_id = str(int(row[0,0]))
    s = Stock(stock_id,1)

    nowlist = s.raw[len(s.raw) - 1]
    
    now_dollar = float(nowlist[6])
    buy_dollar = float(row[0,1])
    diff = (now_dollar - buy_dollar)

    log ="%s %s, %f (Buy:%f)=>%f" % (stock_id, s.info[1], now_dollar, buy_dollar, diff)
    show_list.append(log+"\n")
    print "Stock %s %s, %f (Buy:%f)=>%f" % (stock_id, s.info[1], now_dollar, buy_dollar, diff)
    
    Earn = Earn + diff

    if today == '':
      today = nowlist[0]
  print today
  print "Now Stock Earn=%f !!!" % (Earn)
  print "Now $ Earn=%f !!!" % (Earn*1000)

  #============================================Write Tracing Data
  wrt_flag = 1
  f = open(filename,'rb')
  for i in f.readlines():
    if i.find(today) >= 0:
      wrt_flag = 0
    
  if wrt_flag == 1:
    
    reload(sys)
    sys.setdefaultencoding('utf-8')
    
    f = open(filename,'ab')
    #-------------------------
    f.write("==========================="+today+"==========================="+'')
    f.write(str(Earn)+'\n')
    for data in show_list:
      data.decode('utf-8')
      f.write(data)

    #-------------------------
    f.close()

def TracingFlow(_tracingfile):
  buy_date, stockmap = readStockdata(_tracingfile)
  buylist = calProb(stockmap)
  Tracing(buy_date, buylist)
  
def main():

  tracing_file_list = []
  remachine = re.compile("(\d+-\d+-\d+.txt)")
  cur_dirlist = os.listdir('.')
  for f in cur_dirlist:
    g=remachine.match(f)
    if g != None:
      tracing_file_list.append(g.group(0))
  print tracing_file_list
  for tracing_file in tracing_file_list:
    print "================Tracing ",tracing_file
    TracingFlow(tracing_file)

  input("ENTER:")

if __name__=="__main__":
  main()
  

