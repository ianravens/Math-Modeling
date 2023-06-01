#!/usr/bin/env python
# coding: utf-8

# # HW 1

# # 1) Ex. 1.1

# Two bodies with masses equal to $m_{1}$ and $m_{2}$. The distance between their centers of mass is $r$. The law of universal gravitation states that two bodies attract each other with an attraction equal to

# $F_{g} = G\frac{m_{1}m_{2}}{r^{2}}$

# Find the dimension of the universal gravitational constant G.

# $G = \frac{F_{g}r^{2}}{m_{1}m{2}}$
# 
# $[F] = [m][a] = MLT^{-2}$
# 
# $[a] = LT^{-2}$

# \begin{align*}
# [G] = (MLT^{-2})L^{2}M^{-2}\\
#     = M^{-1}L^{3}T^{-2}
# \end{align*}

# # 2) Ex. 1.4(a)

# Dimensional analysis with experimental data for the free-fall of a body.

# (a) Make a dimensional analysis for falling distance h as a function of mass $m$, gravitational acceleration g, and time t.

# Universal model(1.20):
# 
# $X = \alpha X_1^{n_1}X_2^{n_2}X_3^{n_3}...X_k^{n_k}$

# $h = \alpha m^{n_{1}}g^{n_{2}}t^{n_{3}}$ 
# 
# Letting $a$, $b$ and $c$ equal $n_{1}, n_{2}, n_{3}$.
# 
# $[h] = [\alpha][m]^{a}[g]^{b}[t]^{c}$
# 
# Substitute dimension for each variable.
# 
# $L = 1 \times M^{a}(LT^{-2})^{b}T^{c} = M^{a}L^{b}T^{-2b+c}$
# 
# Compare exponents in equation: 
# 
# $L = M^{a}L^{b}T^{-2b+c}$
# 
# We notice that:
# 
# \begin{align*}
# a = 0\\
# b = 1\\
# -2b+c = 0
# \end{align*}

# Which yields, $a = 0, b = 1, c = 2$ and the updated equation,

# \begin{align*}
# L = M^{a}(LT^{-2})^{b}T^{c}\\
# = M^{0}(LT^{-2})^{1}T^{2}\\
# = (LT^{-2})T^{2}\\
# \end{align*}

# Therefore our equation becomes: $h = \alpha g t^{2}$.

# To determine $\alpha$, we need another constraint. A law of nature such as the law of energy conservation can be used to determine the value or experimental data can be used. We will use experimental data here.

# For this I dropped a role of tape from a height of 8ft and timed it using a stopwatch for a time of 0.68 seconds. So, my data is as follows:

# Converting feet to meters:
# $h = 8.0 \times 0.3048 = 2.4384m$
# 
# Time of fall:
# $t = 0.68s$
# 
# Gravitational acceleration:
# $g = 9.8 \frac{m}{s^{2}}$
# 
# My equation is then:
# 
# \begin{align*}
# \alpha = \frac{h}{gt^{2}} = \frac{2.4384}{9.8 \times (0.68)^{2}} = 0.538 \approx 0.5
# \end{align*}

# Therefore my final equation is:
# 
# \begin{align*}
# h = \frac{1}{2}gt^{2}
# \end{align*}

# # 3) Ex. 1.6

# (a) Draw at least two diagrams to illustrate the nuclear shock wave problem in Section 1.5.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


#Takes time as parameter to calculate and plot shock wave radius
def plotShockWave(t1, t2):
    
    #Letting alpha equal 1 as estimated by G.I. Taylor
    E = 9.4*10**13 #joules of energy. Data taken from pg. 18.
    p = 1.1839 #air density in kg/m^3
    

    #Calculate radius of blast using derived equation
    radius1 = ((E/p)**(1/5))*t1**(2/5)
    radius2 = ((E/p)**(1/5))*t2**(2/5)
    
    #Plot circles from calculated radii
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    a = radius1 * np.cos( theta )
    b = radius1 * np.sin( theta )
    c = radius2 * np.cos( theta )
    d = radius2 * np.sin( theta )
    figure, axes = plt.subplots( 1 )
    axes.plot( a, b, color = 'r', label = f"Radius: {radius1:.2f} meters at {t1} seconds")
    axes.plot( c, d, label = f"Radius: {radius2:.2f} meters at {t2} seconds")
    axes.set_aspect( 1 )
     
    plt.title( 'Radius of shock wave at time $t_{1},t_{2}$' )
    plt.xlabel("meters", size = 15)
    plt.ylabel("meters", size = 15)
    plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))
    plt.show()

#Plot the projected nuclear shock radius at given time t
plotShockWave(0.025,3)
plotShockWave(1, 12)
plotShockWave(0.9,1.5)


# The above are plots of the calculated shock wave radius for a given time t. This simulates the the explosion as if the viewer were looking directly down upon the explosion. The explosion would be spherical but not necessarily perfect as it may appear on the graph. At 0.025 seconds, I calculated the radius would be 137.76 meters while the book showed 135 meters. For 3 seconds, I showed 935 meters while the Trinity explosion was estimated to be 909 meters. The 2nd plot shows the radius of the shock wave at 1 second and 12 seconds. If a person were within 1 mile of the blast, the explosion would have reached them in around 12 seconds. The bottom plot displays how quickly the initial shock wave propogates up until 1 second and then slows down after.

# (b) Write down the derivation details for the result equation: $R = \alpha(\frac{E}{p})^{\frac{1}{5}}t^{\frac{2}{5}}$

# Universal model(1.20):
# 
# $X = \alpha X_1^{n_1}X_2^{n_2}X_3^{n_3}...X_k^{n_k}$

# Critical elements involved in calculating radius are supersonic push and compressible air. To account for this, we include density $p$ of compressible air. We also include total energy $E$ and time $t$. Gravity is excluded because atmospheric pressure is small relative to the pressure of the nuclear explosion. Then the equation can be made from the universal model:
# 
# \begin{align*}
# R = \alpha E^{a}p^{b}t^{c}
# \end{align*}
# 
# Inserting dimension for variables:
# 
# \begin{align*}
# [R] = [\alpha][E]^{a}[p]^{b}[t]^{c}
# \end{align*}
# 
# $[R] = L$
# 
# $[E] = ML^{2}T^{-2}$
# 
# $[p] = ML^{-3}$
# 
# $[t] = T$
# 
# \begin{align*}
# L = 1 \times (ML^{2}T^{-2})^{a}(ML^{-3})^{b}T^{c}\\
# L = (M^{a}L^{2a}T^{-2a})M^{b}L^{-3b}T^{c}\\
# L = M^{a+b}L^{2a-3b}T^{-2a+c}
# \end{align*}
# 
# Comparing exponents of right hand side to left yields:
# 
# \begin{align*}
# a + b = 0\\
# 2a - 3b = 1\\
# -2a + c = 0
# \end{align*}

# Solving system of equations:

# \begin{align*}
# a = -b\\
# -2b - 3b = 1 \Rightarrow b = -\frac{1}{5} \Rightarrow a = \frac{1}{5}\\
# -2(\frac{1}{5}) + c = 0 \Rightarrow c = \frac{2}{5}\\
# a = \frac{1}{5}\\
# b = -\frac{1}{5}\\
# c = \frac{2}{5}
# \end{align*} 

# Inputting values back into our equation yields:
# 
# \begin{align*}
# R = \alpha E^{\frac{1}{5}}p^{-\frac{1}{5}}t^{\frac{2}{5}}\\
# R = \alpha(\frac{E}{p})^{\frac{1}{5}}t^{\frac{2}{5}}
# \end{align*} 

# (c) Discuss the problem assumptions and the result.

# The beginning of the problem discusses the supersonic compression of air from one side so that the air mass builds to a large difference than the other side and creates the shock. The compression of air mass is why density of compressible air is incorporated into the derivation of our equation. Then the energy of the bomb and the amount of time after the explosion help calculate radius. The first second of the explosion is where a significant amount of the radius expansion occurs. The last time noted in the text was at 12 seconds. My calculation had the radius at around 1600 meters. This means that if a person were within 1 mile of the explosion in any direction, they would have right around 10-12 seconds to find shelter. The wave would be 2 miles across at this point.

# # 4)
# 
# Derive the formula (1.56) on Page 12 for the speed of a free fall body from Eq. (1.42) in Example 1.5 on Page 11. You basically repeat what was done in the book, fill in the details not included in the book, and explain each of your steps.

# Universal model(1.20):
# 
# $X = \alpha X_1^{n_1}X_2^{n_2}X_3^{n_3}...X_k^{n_k}$
# 
# To calculate the speed of a free fall body, we must consider the mass $m$ of the object, the time length $t$ since the object was dropped, gravitational acceleration $g$ and the distance $h$ the object has traveled since time $t$. Plugging this into the universal model  and letting n1-n4 be noted as a-d yields:
# 
# \begin{align*}
# v = \alpha m^{a}t^{b}g^{c}h^{d}
# \end{align*} 
# 
# Inputting dimensions for variables:
# \begin{align*}
# [v] = [\alpha] [m]^{a}[t]^{b}[g]^{c}[h]^{d}
# \end{align*} 
# 
# Identifying dimension: 
# 
# $[v] = LT^{-1}$
# 
# $[m] = M$
# 
# $[t] = T$
# 
# $[g] = LT^{-2}$
# 
# $[h] = L$
# 
# 
# 
# Replacing variables with dimension and simplifying:
# \begin{align*}
# LT^{-1} = 1 \times M^{a}T^{b}(LT^{-2})^{c}L^{d}\\
# LT^{-1} = M^{a}T^{b}L^{c}T^{-2c}L^{d}\\
# LT^{-1} = M^{a}T^{b-2c}L^{c+d}
# \end{align*} 

# Compare exponents of right hand side to left hand side. We notice the following system of equations:
# 
# \begin{align*}
# a = 0\\
# b - 2c = -1\\
# c + d = 1
# \end{align*} 

# Solving this system will be difficult because of the four variables for three equations. One of the variables is unnecessary.  The two variables that are most similar are length of time $t$ from drop and distance $h$ the object has traveled since drop. One should be eliminated and we will choose $h$, thus making $d = 0$. Our equation becomes:
# 
# \begin{align*}
# v = \alpha m^{a}t^{b}g^{c}
# \end{align*} 
# 
# Our system of equations becomes:
# 
# \begin{align*}
# a = 0\\
# b - 2c = -1\\
# c = 1
# \end{align*} 
# 
# Solving for b:
# 
# \begin{align*}
# b - 2(1) = -1 \Rightarrow b = 1
# \end{align*}
# 
# Then our values are:
# \begin{align*}
# a = 0\\
# b = 1\\
# c = 1
# \end{align*}
# 

# Inputting these back into our equation we have:
# 
# \begin{align*}
# v = \alpha m^{0}t^{1}g^{1}\\
# v = \alpha gt
# \end{align*} 

# Note: This is the speed formula for a free fall body.

# If we were to return to our system of equations from before:
# 
# \begin{align*}
# a = 0\\
# b - 2c = -1\\
# c + d = 1
# \end{align*} 

# Instead of dropping $h$, let us drop $t$. This will result in $b = 0$ and our equations become:
# 
# \begin{align*}
# a = 0\\
# -2c = -1\\
# c + d = 1
# \end{align*}
# 
# Solving:
# \begin{align*}
# c = \frac{1}{2}\\
# \frac{1}{2} + d = 1 \Rightarrow d = \frac{1}{2}\\
# \end{align*}
# 

# Our values are then:
# 
# \begin{align*}
# a = 0\\
# c = \frac{1}{2}\\
# d = \frac{1}{2}
# \end{align*} 

# Inputting solutions into equation:
# 
# \begin{align*}
# v = \alpha m^{0}g^{\frac{1}{2}}h^{\frac{1}{2}}\\
# v = \alpha \sqrt{gh}
# \end{align*} 

# To solve $\alpha$, we can apply the law of energy conservation. This states that energy is never destroyed or lost when changing from potential to kinetic energy. Then we use the equivalence:
# 
# \begin{align*}
# E_{p} = mgh = \frac{1}{2}mv^{2} = E_{k}
# \end{align*}

# Substituting our derived equation into the v value of the above formula yields:
# 
# \begin{align*}
# mgh = \frac{1}{2}m (\alpha \sqrt{gh})^{2}\\
# mgh = \frac{1}{2}mgh \alpha^{2}\\
# 1 = \frac{1}{2} \alpha^{2}\\
# \alpha = \sqrt{2}
# \end{align*}

# Then our equation for a free fall body becomes:
# 
# \begin{align*}
# v = \sqrt{2}\sqrt{gh}\\
# v = \sqrt{2gh}
# \end{align*}

# # 5) Exercise 1.13: 
# From Fig. 1.6 and Eq. (1.61), derive the period  𝜏 formula Eq. (1.60) of the harmonic oscillation using an assumed model. The problem is already solved in the book. You need to fill in more details and present a complete solution in your own way. You can choose to determine your constant  𝛼 either by a law of physics or by experiments.

# Universal model(1.20):
# 
# $X = \alpha X_1^{n_1}X_2^{n_2}X_3^{n_3}...X_k^{n_k}$

# To determine the period $\tau$ formula, we need to consider the mass, length and gravity acceleration. Inputting these values into the universal model yields:
# 
# \begin{align*}
# \tau = \alpha m^{a}l^{b}g^{c}
# \end{align*}
# 
# Change notation and finding dimension of variables:
# 
# \begin{align*}
# [\tau] = [\alpha] [m]^{a}[l]^{b}[g]^{c}
# \end{align*}
# 
# $[\tau] = T$
# 
# $[m] = M$
# 
# $[l] = L$
# 
# $[g] = LT^{-2}$
# 
# \begin{align*}
# T = 1 \times M^{a}L^{b}(LT^{-2})^{c}\\
# T = M^{a}L^{b}L^{c}T^{-2c}\\
# T = M^{a}L^{b+c}T^{-2c}
# \end{align*}

# Evaluating exponents of right hand side to left. We notice the system of equations:
# 
# \begin{align*}
# a = 0\\
# b + c = 0\\
# -2c = 1
# \end{align*}

# Solving:
# 
# \begin{align*}
# c = -\frac{1}{2}\\
# b + (-\frac{1}{2}) = 0\\
# b = \frac{1}{2}
# \end{align*}

# Our values become:
# 
# \begin{align*}
# a = 0\\
# b = \frac{1}{2}\\
# c = -\frac{1}{2}
# \end{align*}

# Inputting these values, our equation becomes:
# 
# \begin{align*}
# \tau = \alpha m^{0}l^{b\frac{1}{2}}g^{-\frac{1}{2}}\\
# \tau = \alpha l^{b\frac{1}{2}}g^{-\frac{1}{2}}\\
# \tau = \alpha \sqrt{\frac{l}{g}}
# \end{align*}

# To determine the alpha, I collected data using a piece of 36 inch paracord tied to a water jug. My homemade pendulum took 4.15 seconds to complete two periods. So equation is:
# 
# \begin{align*}
# \tau = 2\alpha \sqrt{\frac{l}{g}}
# \end{align*}
# 
# My collected values and gravity acceleration are:
# 
# $\tau = 4.15$ seconds
# 
# $l = 36in \times 0.0254 = 0.9144m$
# 
# $g = 9.8 \frac{m}{s^{2}}$
# 
# My equation is now:

# \begin{align*}
# 4.16 = 2\alpha \sqrt{\frac{0.9144}{9.8}}\\
# 4.16 = \alpha \times 0.6109\\
# \alpha = \frac{4.16}{0.6109}\\
# \alpha = 6.79
# \end{align*}

# Dividing our alpha by $\pi$ yields:
# 
# \begin{align*}
# \alpha = \frac{6.79}{\pi} = 2.16\\
# \Rightarrow \alpha \approx 2\pi
# \end{align*}

# Then our equation is:
# 
# \begin{align*}
# \tau = 2\pi \sqrt{\frac{l}{g}}
# \end{align*}

# # 6) 2.1

# In[2]:


t = np.linspace(2015,2018, 100)
y1 = np.sin(2*np.pi*(t-0.1))
y2 = np.cos(2*np.pi*t)**2
plt.plot(t,y1, 'k-', label="$y_{1}$", linewidth=2)
plt.plot(t,y2, 'b-', label="$y_{2}$", linewidth=2)

plt.title("$y_{1}$ and $y_{2}$ Plot")
plt.xlabel("$x$", size=20)
plt.ylabel("$y$", size=20)
plt.legend(loc='center left',bbox_to_anchor=(1, 0.5))


# # 7) 2.4

# In[3]:


A = np.array([[-3,2,1],[-2,-1,1],[2,1,-4]])
B = np.array([1,2,0])

Z = np.linalg.solve(A,B)
print(f"\nThe values for the system of equations are: \nx = {Z[0]} \ny = {Z[1]} \nz = {Z[2]}")


# # 8) 2.5

# In[4]:


import pandas as pd
import os

os.chdir(r"C:\Users\Tyler\Desktop\School\MATH336\HW\HW1")
surfAirTemp = pd.read_csv("CA042239T.csv", header=0, index_col=0)


# # a) TMAX
# Arrange monthly Cuyamaca Tmax sequence from January 1961 to December 1990 as a matrix with each row as year and each column as month.

# In[5]:


#Get data from tmax 
tMaxData = surfAirTemp['TMAX (F)'].values

#Reshape all TMAX data into 12 columns
rows = 128 #1536/12months = 128rows
cols = 12
tMaxData = tMaxData.reshape(rows, cols)

#Columns and rows for dataframe
years = np.arange(1887,2015, 1)
months =['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

#Create TMAX dataframe
maxDat = pd.DataFrame(tMaxData, index = years, columns = months )

#Retrieve data for desired years
maxDat.loc['1961':'1990',:'Dec']


# # b) TMIN

# In[6]:


#Get data from tmin
tMinData = surfAirTemp['TMIN (F) '].values

#Reshape all TMIN data into 12 columns
tMinData = tMinData.reshape(rows, cols)

#Create TMIN dataframe
minDat = pd.DataFrame(tMinData, index = years, columns = months)

#Retrieve data for desired years
minDat.loc['1961':'1990',:'Dec']


# # c) TMEAN

# In[7]:


#Get data from tmean 
tMeanData = surfAirTemp['TMEAN (F)'].values

#Reshape all TMEAN data into 12 columns
tMeanData = tMeanData.reshape(rows, cols)

#Create TMEAN dataframe
meanDat = pd.DataFrame(tMeanData, index = years, columns = months )

#Retrieve data for desired years
meanDat.loc['1961':'1990',:'Dec']


# # 9) 2.7

# Plot the linear trend lines of Cuyamaca January Tmin time series from 1951 to 2010 with a continuous curve.

# In[8]:


#Retrieve desired data from min dataframe
minDataArr = np.asarray(minDat.loc['1951':'2010',:'Jan'],dtype=float)

#Data used in for loop
years = np.arange(1951,2011,1)
fmt = ['b-', 'g-', 'c-', 'm-']
indices = [0, 10, 20, 30]
startYear = [1951, 1961, 1971, 1981]


def linear_model(x):
    """
    This function accepts a numpy array of length 2, and returns a linear
    lambda function using the elements of the array as the coefficients,
    like so: f(t) = x[0]*t + x[1]
    """
    return lambda t: x[0]*t + x[1]


# In[9]:


#Plot trend lines
plt.figure(figsize=(12,9))
plt.plot(years, minDataArr, 'ko-',mfc='none')

for (col, i, year) in zip(fmt, indices, startYear):
    
    #Calculates coefficients(m,b) of polynomial w/ degree 1: y = mx + b
    #Assigns values to 2d array minTrend: [m,b]
    minTrend = np.polyfit(years[i:], minDataArr[i:], 1)
    
    #lambda function of f(t) = [m]*t + [b] with coefficients from polyfit
    temporalLine = linear_model(minTrend)
    
    #Generates label for legend with trending slope of current time period
    lab = f"{year}--2010 Trend: {minTrend[0]} $\degree$F per interval"
    
    #Plot temporal line for current time period using previous lambda function
    #x = year starting at beginning of period to 2010
    #y = calculated by lambda function with x starting year to 2010.
    plt.plot(years[i:], temporalLine(years[i:]), col, linewidth=7 if year == 1961 else 4, label=lab) 
    
#Formatting    
plt.xlabel("Year",size=15)
plt.ylabel("Temperature [$\degree$F]",size=15)
plt.legend(loc='upper left')
plt.title("Cuyamaca Minimum Temps for January",fontweight="bold");
plt.tick_params(labelsize=15)
plt.grid()


# # 10) 2.9

# Use the gridded NOAA global monthly temperature anomaly data NOAAGlobal-Temp from the following website or another data source
# 
# Or use the NOAAGlobalT.csv data file from the book's data.zip file downloaded

# Choose two 5x5 grid boxes. Go to the map link he provided, find lat and lon for San Diego and Wallace, NE. Plot the oscillation of temperatuer anomalies like he showed in class. Remember, his data wraps east 360 degrees. So SD is 32.5, 244. 
# 

# In[10]:


os.chdir(r"C:\Users\Tyler\Desktop\School\MATH336\HW\HW1")
globTemp = pd.read_csv("NOAAGlobalT.csv", header=0, index_col=0)

#Replace missing data with NaN so -999.9 is not included in our plot
globTemp = globTemp.replace(-999.9,np.NaN)
globTemp


# Latitude and longitude from https://4dvd.sdsu.edu/ for chosen locations:
# 
# Wallace, NE - Lat: 42.5N, Lon: 102.5W
# 
# Cape Horn, Chile - Lat: 57.5S Lon: 67.5W
# 

# In[11]:


#Convert longitude to match up with data file
print(360-102.5)
print(360-67.5)


# Wallace, NE - Lat: 42.5, Lon: 257.5
# 
# Data located at row 1923
# 
# Cape Horn, Chile - Lat: 57.5 Lon: 292.5
# 
# Data located at row 2146

# In[12]:


#Retrieve temperature anomalies for Wallace, NE
walDat = globTemp.iloc[1923,2:]
walDat


# In[13]:


#Retrieve temperature anomalies for Cape Horn, Chile
capDat = globTemp.iloc[2146,2:]
capDat


# In[14]:


#Plot
t = np.linspace(1880,2017,1645)
plt.plot(t,walDat,'-b',label = 'Wallace, NE')
plt.plot(t,capDat, 'r',label = 'Cape Horn, Chile')
plt.title("Temp. Anomalies: Wallace, NE and Cape Horn, Chile")
plt.xlabel("Year", size = 15)
plt.ylabel("Anomaly", size = 15)
plt.legend(loc = 'center left', bbox_to_anchor=(1,0.5))


# In[ ]:




