
# coding: utf-8

# In[203]:

#imports
import math
# This line configures matplotlib to show figures embedded in the notebook, 
# instead of opening a new window for each figure. More about that later. 
# If you are using an old version of IPython, try using '%pylab inline' instead.
get_ipython().magic('matplotlib inline')
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[204]:

#open and record data from a file

def openN(fileName):

    #a list of time with evolving x,y coordinates
    wholeList = []

    #open and record
    with open(fileName,"r") as f:
        #for each time
        for line in f:
            #coordinates
            listX = []
            listY = []
            a = line.split(" ")
            #delete the first element which is the time
            words = a[1:]
            m = round((len(words))/2)
            for x in range(0, m):
                listX.append(float(words[2*x]))
                listY.append(float(words[2*x+1]))
            #append xy of each time
            listTime = []
            
            listTime.append(listX)
            listTime.append(listY)
            
            wholeList.append(listTime)
    return wholeList
#print(lister[8])

#open one file
useList = openN("2_1_2_n1_n2_n1#0.00501187.dat")

#start place of points
#startL = useList[0]
#print(useList[50][8])


# In[205]:

#get the distance for each point
def findD(list):
    listD = []
    
    #get start time positions
    listS = list[0]
    #start x
    listS0 = listS[0]
    #start y
    listS1 = listS[1]
    
    #get end
    listE = list[-1]
    #end x
    listE0 = listE[0]
    #end y
    listE1 = listE[1]

    
    #for each pair
    m = len(listS1)
    #print (len(listS))
    for x in range (0,m):
        
        x1 = listS0[x]
        y1 = listS1[x]
        xn = listE0[x]
        yn = listE1[x]
        d = ((xn-x1)**2+(yn-y1)**2)**(1/2)
        listD.append(d)
    return listD

#find distance
listDist = []
listDist = findD(useList)
#print(listDist[0])
#print(len(listDist))


# In[206]:





#k = round(len(startL)/2)
#for j in range (0,k):
    #listxx.append(startL[2*j])
    #print(startL[h])
    #print(h)
    #listyy.append(startL[2*j+1])
    

#plot according to distance

def plotD(listD):
    #plot
    figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
    xlabel('x')
    ylabel('y')
    title('distance')
    print(max(listD))
    
    for x in range(0,len(listD)):
        l = str(round(listD[x]*20))
        if(round(listD[x]*20)<10):
            l = "0"+l
        myColor = '#'+l+l+l
        #myColor = [0*l*3,0*l*11,0*l*0.5]
        #myColor = Color()
        #print(l)
        #print (listxx[x])
        plt.scatter(listxx[x],listyy[x],color = myColor)
        
#plotD(listDist)

print("DONE")


# In[207]:

from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import numpy as NP
gridsize=30
PLT.subplot(111)

# if 'bins=None', then color of each hexagon corresponds directly to its count
# 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# the result is a pure 2D histogram 
x = listxx
y = listyy
z = [2,5,3,5,7]
#gridsize=gridsize,
PLT.hexbin(x, y, C=listDist,  cmap=CM.jet, bins=None)
#PLT.axis([x.min(), x.max(), y.min(), y.max()])

cb = PLT.colorbar()
cb.set_label('mean value')
PLT.show() 


# In[208]:



#cut the list according to the good index
def cutList(listW,index):
    #hold the new list
    list1 = []
    #time
    for m in range (0,len(listW)):
        listime = []
        listx = []
        listy = []
        #go through the index
        for x in range (0,len(index)):
            #add every pair to the acoording time
            listx.append(listW[m][0][index[x]])
            listy.append(listW[m][1][index[x]])
        #add all the pairs to the time
        listime.append(listx)
        listime.append(listy)
        #add time to the larger list
        list1.append(listime)
    
    #return the trimmed list
    return list1




#get rid of start coordinates of points that does not move much during the whole time
def getR(listW,bound):
    #list of good index of points
    listRest = []
    #loop through all the points
    for y in range(0,len(listW[0][0])):
        
        #start place
        x0 = listW[0][0][y]
        y0 = listW[0][1][y]
        boo = 0
        #append place of all time
        for x in range (0,len(listW)):
            #dis from start through the whole time
            d = ((listW[x][0][y]-x0)**2+(listW[x][1][y]-y0)**2)**(1/2)
            #if the point goes out
            
            if(d>bound):
                #get out of the loop
                x = len(listW)
                #change bool
                boo = 1
        
            
        #if the coordinates do go out of the boundary
        if(boo == 1):
            listRest.append(y)
    return listRest




# In[209]:


#find paths of points with a distance smaller than a certain value
#input the list of all data & distance
def findSmall(listD, bound):
    listPa = []
    for x in range(0,len(listD)):
        if(listD[x]<bound):
            
            listPa.append(x)
    return listPa


    


# In[210]:

def getDis(m1,m2):
    k = ((m1[0]-m2[0])**2+(m1[1]-m2[1])**2)**(1/2)
    return k

#find start coordinates of points in the rod
def findDisk(listW,d1,d2,d3,r):
    listStart = []
    #temp = findStartP(listW)

    for y in range(0,len(listW[0][0])):
        #go through the whole time
        onePoint = []
        #start place
        x0 = listW[0][0][y]
        y0 = listW[0][1][y]
        
        onePoint.append(x0)
        onePoint.append(y0)
        #print("this are the coordinates",x0,y0)
        #if point is not within the disk
        if(getDis(d1,onePoint)>r and getDis(d2,onePoint)>r and getDis(d3,onePoint)>r):
            #print(y,"this is the distance",getDis(d1,onePoint))
            listStart.append(y)
    
    #out put good index
    return listStart


# In[211]:

def findTra(listW,index):
    #all points
    listR = []
    
    for x in range (0,len(listW[0][0])):
        
        if (x in index):
            #all time
            listx = []
            listy = []
            for y in range (0,len(listW)):
                listx.append(listW[y][0][x])
                #print("append",listW[y][0][x])
                listy.append(listW[y][1][x])
            onePoint = []
            #print(listx)
            onePoint.append(listx)
            onePoint.append(listy)
            listR.append(onePoint)
        
    return listR


# In[212]:

def findLine(listW):
    num = 0
    for x in range(1,len(listW[0][0])):
        temp = listW[0][0][x]
        #if we have 
        if (temp < listW[0][0][x-1]):
            return x
            

def findIndex(listW,i1,i2,i3):
    #for every point
    listt = []
    for x in range (0,len(listW[0][0])):
        #print(x in i1 and x in i2 and x in i3)

        if x in i1 and x in i2 and x in i3:
            listt.append(x)
    return listt




#fina local minium
def findLM(listD,listW,index):
    #list index of these points
    listM = []
    
    #for each point
    for x in range (0,len(index)):
        m = index[x]
        listAround = [m-1+line,m+line,m-1+2*line,m-1,m,m+1,m-1-line,m-line,m-2*line]
        #print(x,listAround)
        listSquare = []
        
        for l in range (0,len(listAround)):
            #print("this is len", len(listD))
            #print(l,listAround[l])
            #if index is legal
            if (listAround[l]<len(listD)):
                 
                listSquare.append(listD[listAround[l]])
            
            
        minIndex0 = listSquare.index(min(listSquare))
        minIndex = listAround[minIndex0]
        if(minIndex == index[x]):
            listM.append(minIndex)
            
        #for ones are not the smallest,remove
        #for k in range (0,len(listAround)):
            #if in the min list and not min
            #if(listAround[k] in listM && listAround[k] != minIndex):
                #listM.remove(listAround[k])
            
    return listM



# In[213]:

#start place of points
startL = useList[0]
listxx = startL[0]
listyy = startL[1]


#points that travel back
goodI2 = findSmall(listDist,0.05)

#points that move
goodI1 = getR(useList,0.3)



#coordinates of center of disks
d1 = [-1,0]
d2 = [0,0]
d3 = [1,0]
#the radius we want to include
r = 0.25

#points not on the disk
goodI3 = findDisk(useList,d1,d2,d3,r)

#number of columns   
line = findLine(useList)  
print("THIS IS THE COLUMN",line)
#overlapping part
finalIndex = []
finalIndex = findIndex(useList, goodI2, goodI1,goodI3)
print("check",len(useList[0][0]))
#local minium
goodI4 = findLM(listDist,useList,finalIndex)
#print("wowo",finalIndex)
#print("goodI1", goodI1)
#print("goodI2", goodI2)
#print("goodI3", goodI3)
print("goodI4", goodI4)
a =set(goodI1).intersection(goodI2)
a = list(a)
#print(a)

#the list of local minium
listD = cutList(useList,goodI4)


#plot
figure()
xlabel('x')
ylabel('y')
title('distance')
listxxx = []
listyyy = []
for p in range (0,len(goodI4)):
    
    listxxx.append(useList[0][0][goodI4[p]])
    listyyy.append(useList[0][1][goodI4[p]])
#print(listxxx)
plt.scatter(listxxx, listyyy)


# In[214]:

#plot
figure()
xlabel('x')
ylabel('y')
title('distance')
listxxx1 = []
listyyy1 = []
for p in range (0,len(goodI2)):
    
    listxxx1.append(useList[0][0][goodI2[p]])
    listyyy1.append(useList[0][1][goodI2[p]])
    #print(goodI1[p])
plt.scatter(listxxx1, listyyy1)


# In[215]:

#plot
figure()
xlabel('x')
ylabel('y')
title('distance')
listxxx0 = []
listyyy0 = []
for p in range (0,len(goodI3)):
    
    listxxx0.append(useList[0][0][goodI3[p]])
    listyyy0.append(useList[0][1][goodI3[p]])
    #print(goodI1[p])
plt.scatter(listxxx0, listyyy0)


# In[216]:

#plot
figure()
xlabel('x')
ylabel('y')
title('distance')
listxx1 = []
listyy1 = []
for p in range (0,len(goodI2)):
    
    listxx1.append(useList[0][0][goodI1[p]])
    listyy1.append(useList[0][1][goodI1[p]])
    #print(goodI1[p])
plt.scatter(listxx1, listyy1)


# In[217]:

traPO = []
traPO = findTra(useList,goodI4)
#print(traPO[0])


# In[229]:

#plot
figure(figsize=(20,20))
xlabel('x')
ylabel('y')
title('distance')
listxxx1 = traPO[12][0]
listyyy1 = traPO[12][1]

for x in range(0,len(traPO)-4):
    listxxx1 = traPO[x][0]
    listyyy1 = traPO[x][1]

    plt.scatter(listxxx1, listyyy1, s=1)

listx0 = traPO[0][0]
listy0 = traPO[0][1]

#plt.scatter(listx0, listy0)


# In[ ]:

#test if a shape is concave
def testConcave(points):
    isConcave = False 
    #from small to large
    # 2 3
    # 0 1
    #points = [[x1,y1],[2],[3],[4]]
    #check for each pair of points
    pair = [[points[0],points[1]],
            [points[0],points[2]],
            [points[2],points[3]],
            [points[3],points[1]]]
    for x in range (0,len(pair)):
        
        #get line
        line1 = findKB(pair[x])
        unconnect = []
        for y in range(0,len(points)):
            if points[y] not in pair[x]:
                unconnect.append(points[y])
                #unconnected = [[x1,y1],[x2,y2]] and x1<x2
        #get unconnected line
        unline = findKB(unconnect)

        #get intersection
        inter = findInter(line1,unline)
        #if the intersection is on the unconnected line section, it is concave
        if(inter[0]<=unconnected[0][0] or inter[0]>=unconnected[1][0] or inter[1]<=unconnected[0][1] or inter[1]>=unconnected[1][1]):
            isConcave = True
    
    return isConcave




    


#search with a certain radius 
#return the indices of the surrounding points
def searchR(listW,point,t):
    #find the time slice
    listS = listW[t]
    #list for return = [[xxx],[yyy],[ididid]]
    listR = [[],[],[]]
    #check for all the points
    for x in range (0,len(listS)):
        #get coordinates from the list
        xx = listS[0][x]
        yy = listS[1][x]
        
        #if coordinates are within the bounds
        if(getDis([xx,yy],point)<r ):
                
            listR[0].append(xx)
            listR[1].append(yy)
            listR[2].append(x)
            
    return listR

#get the smallest three triangle
def getThree(sub,point):
    #find the distance of all these points
    listDistance = []
    for x in range (0,len(sub)):
        #calculate the distance
        j = ((point[0]-sub[0][x])**2+(point[1]-sub[1][x])**2)**(1/2)
        #append the distance and the index
        listDistance.append([j,x])
        #sort according to the distance
    listDistance.sort()
    tri = [sub[listDistance[0][1]],sub[listDistance[1][1]],sub[listDistance[2][1]]]
    i = 0
    while(not checkInside(tri,point)):
        tri = [sub[listDistance[i][1]],sub[listDistance[i+1][1]],sub[listDistance[i+2][1]]]
        i = i+1
        
        
    if(checkInside(tri,point)):
       
        #return the closest three points
        return tri
    
            
#test if we need to update the searching subset
def testSubset(error):
    #1. use the error from averaging three intersections
    #2. use the largest distance from the subset to the center point
    #3. use the error from averaging the subset
    if(error>setError):
        updateSet()
        
        
        
#find a set of certain number of pointsn within c certain distance from a point at a certain time
def updateSet(listW,t,point,size):
    # get the slice of that time period
    listS = listW[t]
    
    
    
    #calculate all the distance
    #find the distance of all these points
        listDistance = []
        for x in range (0,len(listS[0])):
            #calculate the distance
            j = ((point[0]-listR[0][x])**2+(point[1]-listR[1][x])**2)**(1/2)
            #append the distance and the index
            listDistance.append([j,x])
        #sort according to the distance
        listDistance.sort()
    

    
    #get the smallest ones
    for x in range(0,size):
        m = listS[0][listDistance[x][1]]
        n = listS[1][listDistance[x][1]]
        listM.append(m)
        listN.append(n)
        
    listR.append(listM)
    listR.append(listN)
    return listR
    
#the step to increase the radius with 
increase = 0.01
#find the closest three points in circle
#given the point, the list and the radius to start with,m-increase step

#when first input, listR = [[]]!!!
def findCircle(point,listW,t,listR,r,number):
    #
    #point = [x,y]
    #a list of a certain time listS = [[x1,x2,x3...],[y1,y2,y3...]]
    #the result list to go recursivly listR = [[x1,],[y1,]]
    #r is radius, changing for each round
    
    listS = listW[t]
    #if we have more than three points in our return list
    if(len(listR[0]>3)):
        #find the distance of all these points
        listDistance = []
        for x in range (0,len(listR)):
            #calculate the distance
            j = ((point[0]-listR[0][x])**2+(point[1]-listR[1][x])**2)**(1/2)
            #append the distance and the index
            listDistance.append([j,x])
        #sort according to the distance
        listDistance.sort()
        #return the closest three points
        return [listR[listDistance[0][1]],listR[listDistance[1][1]],listR[listDistance[2][1]]]
            
            
    #if we have exactly three points
    elif(len(listR[0]==3)):
        #get out of the loop 
        return listR
        
    #when we do not have enough points to return
    else:
        #check for all the points
        for x in range (0,len(listS)):
            #if the point in the circle
            xx = listS[0][x]
            yy = listS[1][x]
        
            #if coordinates are within the bounds
            if(getDis([xx,yy],point)<r ):
                
                listR[0].append(xx)
                listR[1].append(yy)
            
            
        #search for a larger radius
        r = r + increase
        findCircle(point,listW,t,listR,r)
        
    

#return the line function given two points on the line
def findKB(a,b):
    #a = [x1,y1]
    #b = [x2,y2]

    k = (y1-y2)/(x1-x2)
    b = y1 -k*x1
    lineKB = [k,b]
    return lineKB
    
#input a point and get the SQUARE grid it is on
def getSur(listW,point):
    #check  if the point is on the
    columnN = findLine(listW)
    rowN = len(listW[0][0])//columnN
    grid = []
    #ERROR
    for x in range (0,len(listW[0][0])):
        #if the coordinate is larger
        if(point[0]<listW[0][0][x] ):
            if(point[1]<listW[0][1][x]):
                
                grid = [x-1-line,x-line,x-1,x]
                #return the indices of the points around it
                return grid
            x = x + columnN - 1
            
        



#given the former the point find the point in the next moment
def nextMoment(listW,fpoint,t):
    #t is the current time period
    surPoint = findCircle(fpoint,listW,t,[[]],0.1)
    #use the 3 points to find the crossing
    #use point and line to find distance 1
    dis1 = finddistance(surPoint,fpoint)
    thisCro = findCross(surPoint,dis1)
    
    #use the crossing to find the next surround
    sur2 = findCircle(thisCro,listW,t+1,[[]],0.1)
    
    #use the next 3 to find the next crossing
    #use area and base to find distance2
    dis2 = findnewD(sur2,)
    nextP = findCross(sur2,dis2)
    return nextP



#check if a point is inside a triangle
def checkInside(tri,p):
    #tri = [[xxx],[yyy]]
    #point = [x,y]
    #vector
    p0 = [tri[0][0],tri[1][0]]
    p1 = [tri[0][1],tri[1][1]]
    p2 = [tri[0][2],tri[1][2]]
    #p = p0 + (p1 - p0) * s + (p2 - p0) * t
    #on x/y directions
    
    s = 1/(2*Area)*(p0[1]*p2[0] - p0[0]*p2[1] + (p2[1] - p0[1])*p[0] + (p0[0] - p2[0])*p[1]);
    t = 1/(2*Area)*(p0[0]*p1[1] - p0[1]*p1[0] + (p0[1] - p1[1])*p[0] + (p1[0] - p0[0])*p[1]);
    
    #go to one point and then use two lines
    if(0 <= s <= 1 and 0 <= t <= 1 and s + t <= 1):
        return True
    else:
        return False
    
    
    
#given a random point, find track
def findTrack(listW, point):
    #point = [x,y]
    #find the closest three points to start with
    #close = [[xxx],[yyy]]
    close = findCircle(point,listW[0],[[]],0.1)
    #the return list
    listTRA = [[],[]]
    listTRA[0].append()
    listTRA[1].append()
    
    #update the track
    for x in range (0,len(listW)):

        
#given the point and surrounding ones, calculate distance
def finddistance(sur,point):
    #sur = [[xxx],[yyy]]
    #point = [x,y]
    a = [sur[0][0],sur[1][0]]
    b = [sur[0][1],sur[1][1]]
    c = [sur[0][2],sur[1][2]]
    b1 = getDis(a,b)
    b2 = getDis(b,c)
    b3 = getDis(a,c)
    dis = [height(a,b,point,b1),height(b,c,point,b2),height(a,c,point,b3)]
    
    
    return dis

#six points, area from first, base from second, get second height
def findnewD(former, now, fpoint):
    #f = [[xxx],[yyy],[ididid]]
    #fp = [x,y,index]
    d1 = height([former[0][0],former[1][0],fpoint],getDis([now[0][0],now[1][0]]))
    d2 = height([former[0][1],former[1][2],fpoint],getDis([now[0][1],now[1][1]]))
    d3 = height([former[0][2],former[1][2],fpoint],getDis([now[0][2],now[1][2]]))
    
    return [d1,d2,d3]

#use the three new points and the heights to find three intersections
def getDir(sur,h):
    #sur = [[xxx],[yyy]]
    #h = [hhh]
    #the points are marked counter clock wise
    x = [sur[0][0],sur[1][0]]
    y = [sur[0][1],sur[1][1]]
    z = [sur[0][2],sur[1][2]]  
    #print("this is xyz",x,y,z)
    
    
    #vector representation of lines of the triangle
    #line = [startx, sy,ux,uy,s]
    #xy
    mxy = getDis(x,y)
    olxy = [x[0],x[1],(y[0]-x[0])/mxy,(y[1]-x[1])/mxy,mxy]
    #yz
    myz = getDis(y,z)
    olyz = [y[0],y[1],(z[0]-y[0])/myz,(z[1]-y[1])/myz,myz]
    #zx
    mzx = getDis(z,x)
    olzx = [z[0],z[1],(x[0]-z[0])/mzx,(x[1]-z[1])/mzx,mzx]
    
    #print("this is the old triangle lines",olxy,olyz,olzx)
    
    #find direction
    #(a*b)*c = b(a.c)-a(b.c)
    #(a*b)*a = b(a.a)-a(b.a)
    #the vector that starts from the origin
    xy = [y[0]-x[0],y[1]-x[1]]
    yz = [z[0]-y[0],z[1]-y[1]]
    zx = [x[0]-z[0],x[1]-z[1]]
    #reverse direction, just add minus sign
    xyxz = -(xy[0]*zx[0]+xy[1]*zx[1])
    yxyz = -(xy[0]*yz[0]+xy[1]*yz[1])
    zxzy = -(zx[0]*yz[0]+zx[1]*yz[1])
    
    #dxy = xz(mxy^2)-xy(xz.xy)
    d1 = [-zx[0]*(mxy**2)-xy[0]*(xyxz),-zx[1]*(mxy**2)-xy[1]*(xyxz)]
    #yz
    d2 = [-xy[0]*(myz**2)-yz[0]*(yxyz),-xy[1]*(myz**2)-yz[1]*(yxyz)]
    #zx
    d3 = [-yz[0]*(mzx**2)-zx[0]*(zxzy),-yz[1]*(mzx**2)-zx[1]*(zxzy)]
    
    #print("directions",d1,d2,d3)
    
    #move the two points in d1 of h to get two new points to define a newline
    #xy
    dm1 = ((d1[0])**2+(d1[1])**2)**(1/2)
    p1 = h[0]/dm1
    v1 = [d1[0]*p1,d1[1]*p1]
    #yz
    dm2 = ((d2[0])**2+(d2[1])**2)**(1/2)
    p2 = h[1]/dm2
    v2 = [d2[0]*p2,d2[1]*p2]
    #zx
    dm3 = ((d3[0])**2+(d3[1])**2)**(1/2)
    p3 = h[2]/dm3
    v3 = [d3[0]*p3,d3[1]*p3]
    
    #newline
    #xy
    nlxy = [olxy[0]+v1[0],olxy[1]+v1[1],olxy[2],olxy[3],olxy[4]]
    
    #yz
    nlyz = [olyz[0]+v2[0],olyz[1]+v2[1],olyz[2],olyz[3],olyz[4]]
    
    #zx
    nlzx = [olzx[0]+v3[0],olzx[1]+v3[1],olzx[2],olzx[3],olzx[4]]
    
    #intersections
    
    
    i1 = InterVec(nlxy,nlyz)
    i2 = InterVec(nlyz,nlzx)
    i3 = InterVec(nlzx,nlxy)
    
    return[i1,i2,i3]

    

#find intersection of two vectors
def InterVec(l1,l2):
    #l1l2 magnitude m and n
    #l1[0]+l1[2]m = l20+ l2[2]n
    #1313
    n = (l1[3]*l1[0]-l1[1]*l1[2]-l1[3]*l2[0]+l2[1]*l1[2])/(l1[3]*l2[2]-l2[3]*l1[2])
    x = l2[0]+l2[2]*n
    y = l2[1]+l2[3]*n
    return[x,y]
    
    
    
    

#find area 
def findArea(x,y,z):
    area = abs(1/2*(x[0]*(y[1]-z[1]+y[0]*(z[1]-x[1]+z[0]*(x[1]-y[1])))
    return area 
                    
    
#find height of a trangle
def height(x,y,z,base):
    area = findArea(x,y,z)
    height = 2*area/base
    
                
                    
#find intersection of two lines
def findInter(l1,l2):
    #l = [k,b]
    #if the slope is the same
    if(l1[0] == l2[0]):
        print("the slope is too close")
        return "ERROR"
    point = []
    x = (l2[1]-l1[1])/(l1[0]-l2[0])                
    y = l1[0]*x +l1[1]    
    point = [x,y]  
    return point                
                    

                    
                    
                    

