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
        #Need to do to every categorical
        # csv_oh = pd.concat([self.csv,pd.get_dummies(self.csv['bluetooth'])],axis=1)
        csv_oh = pd.concat([self.csv, pd.get_dummies(self.csv['sim'])], axis=1)
        corr = csv_oh.corr()
        plt.figure(figsize=(8, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.show()

    def threeCorrelatedFeatures(self):
        #Need to check again
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
        self.csv['wifi_ord'] = wifi.codes

        cores_dict = ['single','dual','triple','quad','penta','hexa','hepta','octa']
        cores = pd.Categorical(self.csv.cores,
                               ordered=True,
                               categories=cores_dict)
        self.csv['cores_ord'] = cores.codes

        gen_dict = [2,3,4]
        gen = pd.Categorical(self.csv.gen,
                             ordered=True,
                             categories=gen_dict)
        self.csv['gen_ord'] = gen.codes

        speed_dict = ['low','medium','high']
        speed = pd.Categorical(self.csv.speed,
                               ordered=True,
                               categories=speed_dict)

        self.csv['speed_ord'] = speed.codes

        sim_dict = ['Single','Dual']
        sim = pd.Categorical(self.csv.sim,
                             ordered=True,
                             categories=sim_dict)
        self.csv['sim_ord'] = sim.codes

        print(self.csv)

    def nominalToBinary(self):
        bluetooth = pd.get_dummies(self.csv.bluetooth, prefix='bluetooth',drop_first=True)
        self.csv[str(bluetooth.columns.array[0]) + '_bin'] = bluetooth

        screen = pd.get_dummies(self.csv.screen, prefix='screen',drop_first=True)
        self.csv[str(screen.columns.array[0]) + '_bin'] = screen
        print(self.csv)
