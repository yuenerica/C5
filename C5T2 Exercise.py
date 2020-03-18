#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport


# # -------------------------------------------------------------------------------------

# # CREDIT DATA TUTORIAL

# # -----------------------------------------------------------------------------------

# In[2]:


credit = pd.read_csv('DefaultCreditCardClients.csv')


# In[3]:


credit.head()


# ##### Understand the summary of data

# In[4]:


credit.describe()


#  ##### Check the data types of each variable
# 

# In[5]:


credit.info()


# In[6]:


credit.head()


# In[7]:


credit.tail()


# In[8]:


print(credit['SEX'].describe())


# ##### Data transformation - Check any missing data in dataframe

# In[9]:


credit.isnull().values.any()


# In[10]:


credit.isnull().sum()


# In[11]:


credit.isnull()


# ##### Data reduction - Drop "ID" column from dataframe

# In[12]:


credit = credit.drop("ID", axis = 1)


# In[13]:


credit.info


# ##### Data discretization

# ###### Age grouping

# In[14]:


credit['age_bins'] = pd.cut(x=credit['AGE'], bins=[20,29,39,49,59,69,79])


# In[15]:


credit


# In[16]:


credit['age_by_decade']=pd.cut(x=credit['AGE'], bins=[20,29,39,49,59,69,79], labels=['20','30','40','50','60','70'])


# In[17]:


credit


# ##### Limit Balance Grouping

# In[18]:


credit['limit_bins'] = pd.cut(x=credit['LIMIT_BAL'], bins=[9999,199999,299999,399999,499999,599999,699999,799999,899999,999999,1999999])


# In[19]:


credit


# In[20]:


credit['limit_bal_gp']=pd.cut(x=credit['LIMIT_BAL'], bins=[9999,199999,299999,399999,499999,599999,699999,799999,899999,999999,1999999], labels=['10000','200000','300000','400000','500000','600000','700000','800000','900000','1000000'])


# In[21]:


credit


# ##### saving a new dataframe

# In[22]:


credit.to_csv(r'C:\Users\User\Desktop\new_credit.csv', index=False) 


# ##### Visualizing the data

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt


# ##### Showing column names in dataframe

# In[24]:


header = credit.dtypes.index
print(header)


# ##### Histogram

# In[25]:


plt.hist(credit['LIMIT_BAL'])
plt.show()


# In[26]:


plt.hist(credit['LIMIT_BAL'], bins=5)


# ##### Line plot

# In[27]:


plt.plot(credit['LIMIT_BAL'])


# ##### Scattor plot

# In[28]:


x = credit['PAY_0']
y = credit['PAY_2']


# In[29]:


plt.scatter(x,y)


# ##### Box plot

# In[30]:


header = credit.dtypes.index
print(header)


# In[31]:


A = credit['BILL_AMT1']
plt.boxplot(A,0,'gD')
plt.show()


# ##### Correlation

# In[32]:


corrMat = credit.corr()


# In[33]:


print(corrMat)


# ##### covariance - measure how changes in one variable are associated with changes in a second variable

# In[34]:


covMat = credit.cov()
print(covMat)


# In[35]:


credit.to_csv(r'C:\Users\User\Desktop\new_credit1.csv', index = False)


# # ----------------------------------------------------------------------------

# # CREDIT DATA - PREPARATION & EDA

# # -------------------------------------------------------------------------------

# ### Libraries

# In[36]:


import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from pandas import Series, DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# ### Reading Data Set using Pandas

# In[37]:


credit1 = pd.read_csv('new_credit1.csv')


# ### Analysis

# In[38]:


credit1


# In[39]:


credit1.head()


# In[40]:


credit1.tail()


# In[41]:


credit1.info()


# In[42]:


credit1.describe()


# In[43]:


credit.isnull().sum()


# In[44]:


credit1.columns


# ##### plots

# In[45]:


plt.hist(credit1['LIMIT_BAL'], bins=5)


# In[46]:


plt.hist(credit['limit_bal_gp'])


# In[47]:


A = credit1['BILL_AMT1']
plt.boxplot(A,0,'gD')
plt.show()


# In[48]:


corrMat = credit1.corr()
print(corrMat)


# In[49]:


covMat = credit1.cov()
print(covMat)


# ##### Attributes in split

# In[50]:


credit1.groupby('SEX')['SEX'].count()


# In[82]:


sns.factorplot('SEX',data=credit1,kind='count',aspect=1.5)


# In[88]:


#Default next month and gender
credit1.groupby(['SEX','default_payment_next_month'])['default_payment_next_month'].count()


# In[85]:


#Default next month and gender chart
defaultgender = sns.factorplot('SEX', data=credit1, hue='default_payment_next_month', kind='count', aspect=1.75)
defaultgender.set_xlabels('SEX')


# In[102]:


sns.factorplot('SEX',data=credit1, kind='count', hue='age_by_decade',col='default_payment_next_month',aspect=1.25, size=5)


# In[51]:


credit1.groupby('MARRIAGE')['MARRIAGE'].count()


# In[89]:


sns.factorplot('MARRIAGE',data=credit1,kind='count',aspect=1.5)


# In[90]:


#Default next month and marriage
credit1.groupby(['MARRIAGE','default_payment_next_month'])['default_payment_next_month'].count()


# In[91]:


#Default next month and marriage chart
defaultgender = sns.factorplot('MARRIAGE', data=credit1, hue='default_payment_next_month', kind='count', aspect=1.75)
defaultgender.set_xlabels('MARRIAGE')


# In[52]:


credit1.groupby('age_by_decade')['age_by_decade'].count()


# In[92]:


sns.factorplot('age_by_decade',data=credit1,kind='count',aspect=1.5)


# In[93]:


#Default next month and age by decade
credit1.groupby(['age_by_decade','default_payment_next_month'])['default_payment_next_month'].count()


# In[94]:


#Default next month and age chart
defaultgender = sns.factorplot('age_by_decade', data=credit1, hue='default_payment_next_month', kind='count', aspect=1.75)
defaultgender.set_xlabels('age_by_decade')


# In[53]:


credit1.groupby('EDUCATION')['EDUCATION'].count()


# In[95]:


sns.factorplot('EDUCATION',data=credit1,kind='count',aspect=1.5)


# In[97]:


#Default next month and education
credit1.groupby(['EDUCATION','default_payment_next_month'])['default_payment_next_month'].count()


# In[98]:


#Default next month and education chart
defaultgender = sns.factorplot('EDUCATION', data=credit1, hue='default_payment_next_month', kind='count', aspect=1.75)
defaultgender.set_xlabels('EDUCATION')


# In[54]:


credit1.groupby('limit_bal_gp')['limit_bal_gp'].count()


# In[99]:


sns.factorplot('limit_bal_gp',data=credit1,kind='count',aspect=1.5)


# In[100]:


#Default next month and limit bal gp
credit1.groupby(['limit_bal_gp','default_payment_next_month'])['default_payment_next_month'].count()


# In[101]:


#Default next month and limit bal gp chart
defaultgender = sns.factorplot('limit_bal_gp', data=credit1, hue='default_payment_next_month', kind='count', aspect=1.75)
defaultgender.set_xlabels('limit_bal_gp')


# ##### Define those whose default

# In[56]:


default_clients = credit1[credit1['default_payment_next_month']==1]


# In[56]:


len(default_clients)


# ## Client who default group by different attributes

# In[57]:


credit1.pivot_table('default_payment_next_month','SEX','age_by_decade',aggfunc=np.sum,margins=True)


# In[58]:


credit1.pivot_table('default_payment_next_month','SEX','EDUCATION',aggfunc=np.sum,margins=True)


# ###### Payment Status factor - Payment status in Sep

# In[59]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_0',aggfunc=np.sum,margins=True)


# In[153]:


sns.factorplot('PAY_0','default_payment_next_month', data=credit1)


# In[155]:


sns.factorplot('PAY_0','default_payment_next_month', hue='SEX', data=credit1)


# ###### Payment Status factor - Payment status in Aug

# In[60]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_2',aggfunc=np.sum,margins=True)


# In[156]:


sns.factorplot('PAY_2','default_payment_next_month', data=credit1)


# In[159]:


sns.factorplot('PAY_2','default_payment_next_month', hue='EDUCATION', data=credit1)


# In[157]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_2',aggfunc=np.sum,margins=True)


# ###### Payment Status factor - Payment status in Jul

# In[160]:


sns.factorplot('PAY_3','default_payment_next_month', data=credit1)


# In[161]:


sns.factorplot('PAY_3','default_payment_next_month', hue='SEX', data=credit1)


# In[61]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_3',aggfunc=np.sum,margins=True)


# ###### Payment Status factor - Payment status in Jun

# In[62]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_4',aggfunc=np.sum,margins=True)


# In[162]:


sns.factorplot('PAY_4','default_payment_next_month', data=credit1)


# ######  Payment Status factor -  Payment status in May

# In[63]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_5',aggfunc=np.sum,margins=True)


# In[163]:


sns.factorplot('PAY_5','default_payment_next_month', data=credit1)


# In[164]:


sns.factorplot('PAY_5','default_payment_next_month', hue='SEX', data=credit1)


# In[166]:


sns.factorplot('PAY_5','default_payment_next_month', hue='EDUCATION', data=credit1)


# In[167]:


sns.factorplot('PAY_5','default_payment_next_month', hue='MARRIAGE', data=credit1)


# ###### Payment Status factor - Payment status in Apr

# In[64]:


credit1.pivot_table('default_payment_next_month','SEX','PAY_6',aggfunc=np.sum,margins=True)


# In[169]:


sns.factorplot('PAY_6','default_payment_next_month', data=credit1)


# In[170]:


sns.factorplot('PAY_6','default_payment_next_month', hue='SEX', data=credit1)


# In[171]:


sns.factorplot('PAY_6','default_payment_next_month', hue='EDUCATION', data=credit1)


# In[172]:


sns.factorplot('PAY_5','default_payment_next_month', hue='MARRIAGE', data=credit1)


# ##### age factor

# In[139]:


sns.factorplot('age_by_decade','default_payment_next_month', data=credit1)


# ##### limit bal factor

# In[133]:


sns.factorplot('limit_bal_gp','default_payment_next_month', data=credit1)


# In[115]:


sns.factorplot('limit_bal_gp','default_payment_next_month', hue='SEX', data=credit1)


# In[122]:


#linear plot default (1) vs limit bal gp
sns.lmplot('limit_bal_gp','default_payment_next_month', data=credit1)


# In[124]:


#linear plot default (1) vs limit bal gp by sex
sns.lmplot('limit_bal_gp', 'default_payment_next_month', hue='SEX', data=credit1)


# ##### age factor

# In[117]:


sns.factorplot('default_payment_next_month', data=credit1, hue='age_by_decade', kind='count', 
               col='SEX')


# In[134]:


sns.factorplot('age_by_decade','default_payment_next_month', data=credit1)


# In[129]:


sns.factorplot('age_by_decade','default_payment_next_month', hue='SEX', data=credit1)


# In[121]:


#linear plot default (1) vs age
sns.lmplot('age_by_decade','default_payment_next_month', data=credit1)


# In[120]:


#linear plot default (1) vs age by sex
sns.lmplot('age_by_decade','default_payment_next_month', hue='SEX', data=credit1)


# ##### Education factor

# In[132]:


sns.factorplot('EDUCATION','default_payment_next_month', data=credit1)


# In[131]:


sns.factorplot('EDUCATION','default_payment_next_month', hue='SEX', data=credit1)


# In[143]:


sns.factorplot('EDUCATION','default_payment_next_month', hue='age_by_decade', data=credit1)


# In[125]:


#linear plot default (1) vs edu
sns.lmplot('EDUCATION','default_payment_next_month', data=credit1)


# In[126]:


#linear plot default (1) vs edu by sex
sns.lmplot('EDUCATION','default_payment_next_month', hue='SEX', data=credit1)


# In[128]:


#linear plot default (1) vs edu by age
sns.lmplot('age_by_decade','default_payment_next_month', hue='EDUCATION', data=credit1)


# ##### marriage factor

# In[135]:


sns.factorplot('MARRIAGE','default_payment_next_month', data=credit1)


# In[136]:


sns.factorplot('MARRIAGE','default_payment_next_month', hue='SEX', data=credit1)


# ### Those who use revolving credit or payment delay for 2 months are likely will be default next month. Portion of female is slightly higher

# In[74]:


table1 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.age_by_decade], columns=[credit1.EDUCATION, credit1.SEX])


# In[75]:


table1


# In[76]:


table.columns, table.index


# In[80]:


#Change column name
table1.columns.set_levels(['other0','Grad school','university','high school','other4','other5','other6'], level=0, inplace=True)
table1.columns.set_levels(['Male','Female'], level=1, inplace=True)
table1.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table1


# In[144]:


table2 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.limit_bal_gp], columns=[credit1.EDUCATION, credit1.SEX])


# In[145]:


table2


# In[146]:


#Change column name
table2.columns.set_levels(['other0','Grad school','university','high school','other4','other5','other6'], level=0, inplace=True)
table2.columns.set_levels(['Male','Female'], level=1, inplace=True)
table2.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table2


# In[147]:


table3 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.limit_bal_gp], columns=[credit1.MARRIAGE, credit1.SEX])


# In[148]:


table3


# In[149]:


#Change column name
table3.columns.set_levels(['others','married','single','divorce'], level=0, inplace=True)
table3.columns.set_levels(['Male','Female'], level=1, inplace=True)
table3.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table3


# In[150]:


table4 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.age_by_decade], columns=[credit1.MARRIAGE, credit1.SEX])


# In[151]:


table4


# In[152]:


#Change column name
table4.columns.set_levels(['others','married','single','divorce'], level=0, inplace=True)
table4.columns.set_levels(['Male','Female'], level=1, inplace=True)
table4.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table4


# In[175]:


table5 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.limit_bal_gp], columns=[credit1.SEX, credit1.PAY_5])


# In[176]:


table5


# In[177]:


#Change column name
table5.columns.set_levels(['Male','Female'], level=0, inplace=True)
table5.columns.set_levels(['no consumpt','paid in full','use credit','delay 1mth','delay 2mth', 'delay 3mth', 'delay 4mth', 'delay 5mth', 'delay 6mth','delay 7 mth'], level=1, inplace=True)
table5.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table5


# In[178]:


table6 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.limit_bal_gp], columns=[credit1.SEX, credit1.PAY_6])


# In[179]:


table6


# In[180]:


#Change column name
table6.columns.set_levels(['Male','Female'], level=0, inplace=True)
table6.columns.set_levels(['no consumpt','paid in full','use credit','delay 1mth','delay 2mth', 'delay 3mth', 'delay 4mth', 'delay 5mth', 'delay 6mth','delay 7 mth'], level=1, inplace=True)
table6.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table6


# In[181]:


table7 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.age_by_decade], columns=[credit1.SEX, credit1.PAY_5])
table7


# In[182]:


#Change column name
table7.columns.set_levels(['Male','Female'], level=0, inplace=True)
table7.columns.set_levels(['no consumpt','paid in full','use credit','delay 1mth','delay 2mth', 'delay 3mth', 'delay 4mth', 'delay 5mth', 'delay 6mth','delay 7 mth'], level=1, inplace=True)
table7.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table7


# In[183]:


table8 = pd.crosstab(index=[credit1.default_payment_next_month,credit1.age_by_decade], columns=[credit1.SEX, credit1.PAY_6])
table8


# In[185]:


#Change column name
table8.columns.set_levels(['Male','Female'], level=0, inplace=True)
table8.columns.set_levels(['no consumpt','paid in full','use credit','delay 1mth','delay 2mth', 'delay 3mth', 'delay 4mth', 'delay 5mth', 'delay 6mth','delay 7 mth'], level=1, inplace=True)
table8.index.set_levels(['not default next month','default next month'], level=0,inplace=True)
table8


# In[69]:


import pandas_profiling as pp
import os


# In[70]:


credit1profile=ProfileReport(credit1, title='Credit1 Report',html={'style':{'full_width':True}})


# In[71]:


credit1profile.to_file(output_file = "credit1_report.html")


# In[ ]:




