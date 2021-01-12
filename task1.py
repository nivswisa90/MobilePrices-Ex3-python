import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import numpy as np


class Summary:
    def __init__(self, csvFile):
        self.csv = pd.read_csv(csvFile, index_col='id')

    def addResolution(self):
        self.csv['resolution'] = self.csv['px_height'] * self.csv['px_width']

    def dotsPerInch(self):
        #Need to check division by 0
        if self.csv['sc_w'].isnull:
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     print(self.csv['sc_w'])
            x = self.csv['px_width'] / self.csv['sc_w']
            self.csv['DPI_w'] = x * 2.54
        else:
            print('else')
            self.csv['DPI_w'] = np.nan
        # self.csv['DPI_w'] = (self.csv['px_width'] / self.csv['sc_w']) * 2.54

        # with pd.option_context('display.max_rows', None, 'display.max_columns',None):
        #     print(self.csv)

    def callRatio(self):
        self.csv['call_ratio'] = self.csv['battery_power']/ self.csv['talk_time']

    def fromMBToGB(self):
        self.csv['memory'] = self.csv['memory'] / 1000

    def printDescription(self):
        print(self.csv.describe())

    def getHistogram(self):
        values = self.csv['price']
        plt.hist(values, bins=5)
        plt.show()

    def dataHeatMap(self):
        corr = self.csv.corr()
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        # f, ax = plt.subplots(figsize=(8, 6))
        # cmap = sns.diverging_palette(200, 10, as_cmap=True)
        # # sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
        # #             square=True, linewidths=.5)
        # sns.heatmap(corr,
        #             xticklabels=corr.columns,
        #             yticklabels=corr.columns)
        plt.figure(figsize=(8,8))
        sns.heatmap(corr,annot=True,cmap="coolwarm")
        plt.show()

    def getCategoricalCorrelation(self):
        # csv_oh = pd.concat([self.csv,pd.get_dummies(self.csv['bluetooth'])],axis=1)
        csv_oh = pd.concat([self.csv, pd.get_dummies(self.csv['wifi'])], axis=1)
        corr = csv_oh.corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.show()

    def threeCorrelatedFeatures(self):
        ram = pd.qcut(self.csv['ram'],4)
        battery = pd.qcut(self.csv['battery_power'], 4)
        pivot = self.csv.pivot_table(index=[battery,'battery_power'],columns=['ram',ram],
                                     aggfunc={'price':'mean', 'gen': sum})
        # , 'gen', ['battery_power',battery])
        print(pivot.unstack())

        # titanic.pivot_table(index='Sex', columns='Pclass',
        #                     aggfunc={'Survived': sum, 'Fare': 'mean'})

    def ordinalToNumerical(self):
        wifi_dict = ['none','b','a','g','n']
        wifi = pd.Categorical(self.csv.wifi,
                                     ordered=True,
                                     categories=wifi_dict
                                     )
        self.csv['wifi_o'] = wifi.codes

        cores_dict = ['single','dual','triple','quad','penta','hexa','hepta','octa']
        cores = pd.Categorical(self.csv.cores,
                               ordered=True,
                               categories=cores_dict)
        self.csv['cores_o'] = cores.codes
        print(self.csv)