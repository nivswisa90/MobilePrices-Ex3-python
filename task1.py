import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')
import seaborn as sns
import numpy as np


class Summary:
    def __init__(self, csvFile):
        # class constructor.
        # Load the data into pandas dataframe
        self.csv = pd.read_csv(csvFile, index_col='id')

    def addResolution(self):
        # Add a column with name resolution that holds the total screen resolution for each device
        self.csv['resolution'] = self.csv['px_height'] * self.csv['px_width']

    def dotsPerInch(self):
        # Add a column that holds the DPI (dots per inch) of the screen width and name it DPI_w.
        self.csv['sc_w'] = (self.csv['px_width'] / self.csv['sc_w']).replace(np.inf, np.nan) * 2.54
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.csv['sc_w'])

    def callRatio(self):
        # Add a column that holds the ratio battery_power/talk_time and name it call_ratio
        self.csv['call_ratio'] = self.csv['battery_power'] / self.csv['talk_time']

    def fromMBToGB(self):
        # Change the memory column to hold the memory in GB instead of MB
        self.csv['memory'] = self.csv['memory'] / 1000

    def printDescription(self):
        # Show table description
        print(self.csv.describe())

    def getPricesHistogram(self):
        # Prices histogram
        values = self.csv['price']
        plt.hist(values, bins=15)
        plt.show()

    def dataHeatMap(self):
        # Plot a correlation heatmap of the data set
        corr = self.csv.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        f, ax = plt.subplots(figsize=(7, 6))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5)
        plt.savefig('heatmap.png')
        plt.show()
        # mask = np.triu(np.ones_like(corr, dtype=bool))
        # f, ax = plt.subplots(figsize=(8, 6))
        # cmap = sns.diverging_palette(200, 10, as_cmap=True)
        # # sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
        # #             square=True, linewidths=.5)
        # sns.heatmap(corr,
        #             xticklabels=corr.columns,
        #             yticklabels=corr.columns)
        # plt.figure(figsize=(8,6))
        # sns.heatmap(corr,annot=True,cmap="coolwarm")
        # plt.show()

    def getCategoricalCorrelation(self, features):
        # Checks if a categorical feature has correlation with the price
        csv_oh = pd.concat([self.csv, pd.get_dummies(self.csv[features])], axis=1)
        corr = csv_oh.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        # f, ax = plt.subplots(figsize=(7, 6))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5)
        plt.show()

    def plotRelationshipWithPrice(self, feature):
        # Need to check what to do with wifi
        sns.regplot(x=feature, y="price", data=self.csv, scatter_kws={"color": "black"}, line_kws={"color": "red"})
        plt.title("price " + feature + " relationship")
        plt.savefig('plot.png')
        plt.show()

    def threeCorrelatedFeatures(self):
        #Need to check again
        ram = pd.qcut(self.csv['ram'], 4)
        battery = pd.qcut(self.csv['battery_power'], 4)
        pivot = self.csv.pivot_table(index=[battery, 'battery_power'], columns=['gen'],
                                     aggfunc={'price': 'mean', 'ram': 'sum'})
        # , 'gen', ['battery_power',battery])
        print(pivot.unstack())

        # titanic.pivot_table(index='Sex', columns='Pclass',
        #                     aggfunc={'Survived': sum, 'Fare': 'mean'})

    def ordinalToNumerical(self):
        # For each ordinal feature <O>, add a column to the dataframe which holds the ordered values
        # representing each original value of the feature. This new column will be named <O>_ord
        wifi_dict = ['none', 'b', 'a', 'g', 'n']
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
        # For each nominal feature <N>, add a binary column OR one-hot encoding to the dataframe
        # representing the original values. Name binary columns <N>_bin, and prefix one-hot encodings with <N>
        bluetooth = pd.get_dummies(self.csv.bluetooth, prefix='bluetooth',drop_first=True)
        self.csv[str(bluetooth.columns.array[0]) + '_bin'] = bluetooth

        screen = pd.get_dummies(self.csv.screen, prefix='screen',drop_first=True)
        self.csv[str(screen.columns.array[0]) + '_bin'] = screen
        print(self.csv)

    def saveIntoCsvFile(self):
        # write the data frame into a csv file
        # Check what to delete
        # del self.csv['bluetooth']
        # del self.csv['screen']
        self.csv.to_csv('mobile_prices_converted.csv', sep=',')
