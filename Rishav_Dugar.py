import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import time
import requests
import sys

filename = sys.argv[1]

#Initializations

threshold_wick = 0.05      #To control wick length
threshold_slope = 0.20     #To control deviation in slopes
threshold_duplicate = 0.05 #To eliminate duplicates
roll_size = 5              #Determine the window to find swing highs and swing lows
channel_size = 25          #Number of ticks to consider in channel
channel_end_selector = -3  #To optimize the process
breakout_range = int(0.25*channel_size)        #To reduce the search


#Accessory Functions
def value(df,g,i,n,slope):
    return slope*(df.Timestamp[g]-df.Timestamp[i]) + n

#Retrieving Data
def get_data(currency_pair, start, end, interval):
    
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    url = '/spot/candlesticks'
    query_param = {"currency_pair":currency_pair, "_from":start, "to":end, "interval":interval}
    r = requests.request('GET', "https://api.gateio.ws/api/v4/spot/candlesticks",params = query_param, headers=headers)
    
    return(np.array(r.json(), dtype = "float"))


#Function to plot the desired graphs and return a datasheet to print

def plot(df, name, i, number, counter):

    container = []
    
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(x=df['Timestamp'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']))
                      
    for j in counter:
        fig.add_vrect(x0=j[0], x1=j[1],
                   y0=0, y1=1,
                   fillcolor="LightSalmon", opacity=0.25,
                   layer="below", line_width=0)
    
    slope1 = (df['High'][i[1]]-df['High'][i[0]])/(df['Timestamp'][i[1]]-df['Timestamp'][i[0]])
    slope2 = (df['Low'][i[3]]-df['Low'][i[2]])/(df['Timestamp'][i[3]]-df['Timestamp'][i[2]])
    
    fig.add_trace(go.Scatter(x=df['Timestamp'][[min(i[0],i[2]),max(i[1],i[3], min(i[0],i[2]) + channel_size)]], 
                             y = value(df,
                                       [min(i[0],i[2]),max(i[1],i[3], min(i[0],i[2]) + channel_size)],
                                       i[0],
                                       df['High'][i[0]],
                                       slope1
                                      ),
                             line=dict(color='cyan'),
                             mode='lines'))
    
    fig.add_trace(go.Scatter(x=df['Timestamp'][[min(i[0],i[2]),max(i[1],i[3], min(i[0],i[2]) + channel_size)]], 
                             y = value(df,
                                       [min(i[0],i[2]),max(i[1],i[3], min(i[0],i[2]) + channel_size)],
                                       i[2],
                                       df['Low'][i[2]],
                                       slope2
                                      ),
                             line=dict(color='cyan'),
                             mode='lines'))
    
    title = name + " -> Channel No : " + str(number) + "<br><sup>Channel from " + str(datetime.utcfromtimestamp(df['Timestamp'][min(i[0],i[2])]).strftime('%d-%m-%Y %H:%M:%S')) + " to " + str(datetime.utcfromtimestamp(df['Timestamp'][max(i[1],i[3], min(i[0],i[2]) + channel_size)]).strftime('%d-%m-%Y %H:%M:%S')) + "</sup>"
    
    container.append([str(datetime.utcfromtimestamp(df['Timestamp'][min(i[0],i[2])]).strftime('%d-%m-%Y %H:%M:%S')),str(datetime.utcfromtimestamp(df['Timestamp'][max(i[1],i[3], min(i[0],i[2]) + channel_size)]).strftime('%d-%m-%Y %H:%M:%S'))])
    
    for j in counter:
        container.append([str(datetime.utcfromtimestamp(j[0]).strftime('%d-%m-%Y %H:%M:%S')),str(datetime.utcfromtimestamp(j[1]).strftime('%d-%m-%Y %H:%M:%S'))])
        title = title + "<br><sup>Breakout from " +  str(datetime.utcfromtimestamp(j[0]).strftime('%d-%m-%Y %H:%M:%S')) + " to " + str(datetime.utcfromtimestamp(j[1]).strftime('%d-%m-%Y %H:%M:%S')) + "</sup>"   
                                                    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        title = title
    )
    
    fig.show()
    
    return container

#Finding the breakout range

def breakout_ranges(df,i,j,k,l):
    
    counter = []
    slope1 = (df.High[k] - df.High[i])/(df.Timestamp[k] - df.Timestamp[i])
    slope2 = (df.Low[l] - df.Low[j])/(df.Timestamp[l] - df.Timestamp[j])
    
    m = min(i,j)
    stop = min(min(i,j)+channel_size,df.shape[0])

    while (m<stop):
        
        if(value(df,m,i,df.High[i],slope1) < df.High[m]):
            for n in range(m+1,min(m+breakout_range + 2,df.shape[0]),1):
                
                if(value(df,n,i,df.High[i],slope1) > df.High[n]):
                    if n>m+1:
                        counter.append([df.Timestamp[m],df.Timestamp[n-1]])
                    m = n+1
                    break
                if n>= stop:
                    counter.append([df.Timestamp[m],df.Timestamp[n]])
                    m = n+1
                    break
        else:
            m = m+1
            
    m = min(i,j)
    while (m<stop):
        
        if(value(df,m,j,df.Low[j],slope2) > df.Low[m]):
            for n in range(m+1,min(m+breakout_range + 2,df.shape[0]),1):
                
                if(value(df,n,j,df.Low[j],slope2) < df.Low[n]):
                    if n>m+1:
                        counter.append([df.Timestamp[m],df.Timestamp[n-1]])
                    m = n+1
                    break
                if n>= stop:
                    counter.append([df.Timestamp[m],df.Timestamp[n]])
                    m = n+1
                    break
        else:
            m = m+1

    return counter







##FUNCTIONS TO CARRY OUT THE ACTUAL TASK

#Finding the swing highs and swing lows
def swing_high_low(df):
    
    high_indices = []
    low_indices = []
    med = int(roll_size/2)
    
    for i in df.rolling(roll_size):
        if i.shape[0] == roll_size :
            if i.High.max() == i.High[i.index.start+med]:
                df["Swing_High"][i.index.start+med] = 1
                high_indices.append(i.index.start+med)
            if i.Low.min() == i.Low[i.index.start+med]:
                df["Swing_Low"][i.index.start+med] = 1
                low_indices.append(i.index.start+med)
    return (df,np.array(high_indices),np.array(low_indices))

#Analysing for neighbourhood breakouts

def breakout(df,slope,g,i,mode):
    
    flag = 1
    if mode == "high":
        for m in range(g+2,min(g+breakout_range + 1,df.shape[0]),1):
            if(value(df,m,i,df.High[i],slope) > df.High[m]):
                flag = 0
                return True
        if flag:
            return False
    elif mode == "low":
        for m in range(g+2,min(g+breakout_range + 1,df.shape[0]),1):
            if(value(df,m,i,df.Low[i],slope) < df.Low[m]):
                flag = 0
                return True
        if flag:
            return False 

#Feasibility Checker

def feasible(df,high_indices,low_indices,i,j,k,l):

    slope1 = (df.High[k] - df.High[i])/(df.Timestamp[k] - df.Timestamp[i])
    slope2 = (df.Low[l] - df.Low[j])/(df.Timestamp[l] - df.Timestamp[j])
    
    #Parallel nature of slope
    
    if abs((slope1-slope2)/max(slope1,slope2)) > threshold_slope :
        return False
    
    #Additional Constraints
    
    for g in range(min(i,j)+1,min(min(i,j)+channel_size+1, df.shape[0]),1):
        
        val1 = value(df,g,i,df.High[i],slope1)
        val2 = value(df,g,j,df.Low[j],slope2)
        
        if k==70 and g == 73:
            print(df.Low[g],val1,val2)
        
        if df.High[g]<=val1 and df.Low[g]>=val2:
            continue
        
        elif (val1<max(df.Open[g],df.Close[g]) and breakout(df,slope1,g,i,"high")  is False) or (val2>min(df.Open[g],df.Close[g]) and breakout(df,slope2,g,j,"low") is False):
            return False
        
        elif (df.High[g]/val1 > (1+threshold_wick) and df.High[g+1] < value(df,g+1,i,df.High[i],slope1)) or (df.Low[g]/val2 < (1-threshold_wick) and df.Low[g+1] > value(df,g+1,j,df.Low[j],slope2)):
            return False
        
        elif (df.High[g]>val1 and breakout(df,slope1,g,i,"high")  is False) or (df.Low[g]<val2 and breakout(df,slope2,g,j,"low") is False):
            return False

    return True









##FUNCTIONS TO ASSIST IN ANALYSIS OF DUPLICATES
def flaws(df,i,channel):
    flaw = 0
    slope1 = (df.High[channel[i][2]] - df.High[channel[i][0]])/(df.Timestamp[channel[i][2]] - df.Timestamp[channel[i][0]])
    slope2 = (df.Low[channel[i][3]] - df.Low[channel[i][1]])/(df.Timestamp[channel[i][3]] - df.Timestamp[channel[i][1]] )                                                           
    for g in range(min(channel[i][0],channel[i][1]), min(min(channel[i][0],channel[i][1]) + channel_size, df.shape[0]), 1):
        if value(df,g,channel[i][0],df.High[channel[i][0]],slope1) < df.High[g]:
            flaw = flaw + 1
        if value(df,g,channel[i][1],df.Low[channel[i][1]],slope2) > df.Low[g]:
            flaw = flaw + 1
    return flaw

def compare(df,i,channel):
    a = flaws(df,i,channel)
    b = flaws(df,i+1,channel)
    if a>b:
        return i
    else:
        return i+1
    
def duplicates(channel,df):
    
    flag = 1
    while flag:
        is_duplicate = 0
        for i in range(len(channel)-1):
            is_duplicate = 0
            if abs(min(channel[i][0],channel[i][1]) - min(channel[i+1][0],channel[i+1][1])) < 5:
                slope1 = [(df.High[channel[i][2]] - df.High[channel[i][0]])/(df.Timestamp[channel[i][2]] - df.Timestamp[channel[i][0]]), (df.Low[channel[i][3]] - df.Low[channel[i][1]])/(df.Timestamp[channel[i][3]] - df.Timestamp[channel[i][1]])]
                slope2 = [(df.High[channel[i+1][2]] - df.High[channel[i+1][0]])/(df.Timestamp[channel[i+1][2]] - df.Timestamp[channel[i+1][0]]), (df.Low[channel[i+1][3]] - df.Low[channel[i+1][1]])/(df.Timestamp[channel[i+1][3]] - df.Timestamp[channel[i+1][1]])]
                diff = np.abs(np.array(slope1) - np.array(slope2))
                if (diff<threshold_duplicate*np.abs(np.array(slope1))).all:
                    is_duplicate = 1
                    m = compare(df,i,channel)
                    del channel[m]
            if is_duplicate:
                break
            if i == len(channel)-2 + is_duplicate:
                flag = 0
    return channel
 
    
    

    


#Iterating through all combinations

def channels(df, high_indices, low_indices, word):
    
    container = []
    channel = []
    
    for i in high_indices:
        for j in low_indices:
            for k in high_indices[np.where((high_indices>max(i,j)) & (high_indices < min(i,j) + channel_size))][channel_end_selector:]:
                for l in low_indices[np.where((low_indices>j)  & (low_indices < min(i,j) + channel_size))][channel_end_selector:]:
                    
                    #Basic Analysis of Highs and Lows(Generally exists in alternating patterns hence this generalisation)
                    if i<j:
                        if l<k:
                            continue
                    else:
                        if l>k:
                            continue
                            
                    s1 = df.High[k] - df.High[i]
                    s2 = df.Low[l] - df.Low[j]
                    
                    if s1*s2 > 0:
                        
                        if feasible(df,high_indices,low_indices,i,j,k,l) :
                            channel.append([i,j,k,l])
                        
    channel = duplicates(channel,df)
    for index,r in enumerate(channel):
        counter = breakout_ranges(df,r[0],r[1],r[2],r[3])
        container.append(plot(df,word,[r[0],r[2],r[1],r[3]], index+1,counter))
                                                             
    print(word,":",len(channel),"\n")
    for j in container:
        for index,k in enumerate(j):
            if index == 0:
                print(k)
            else:
                print("Breakout at ",k)
        print("\n")

        
        
        
#Main Program

with open(filename,"r") as f:
    line = f.readline()
    n = int(line)
    line = f.readline()
    c = int(line)
    for i in range(n):
        line = f.readline()
        line = line.replace("\n","")
        word = line.split(", ")
        currency_pair = word[0]
        currency_pair = currency_pair + "_USDT"
        interval = word[1]
        start = datetime.strptime(word[2],"%m/%d/%Y")
        start = np.int64(time.mktime(start.timetuple()))
        end = datetime.strptime(word[3],"%m/%d/%Y")
        end = np.int64(time.mktime(end.timetuple()))
        data = get_data(currency_pair, start, end, interval)
        df = pd.DataFrame(data, columns = ["Timestamp", "Volume","Close", "High", "Low", "Open"])
        zero = np.zeros((df.shape[0],1))
        df["Swing_High"] = zero
        df["Swing_Low"] = zero
        df,high_indices,low_indices = swing_high_low(df)
        channels(df, high_indices, low_indices, word[0])      
    f.close()
