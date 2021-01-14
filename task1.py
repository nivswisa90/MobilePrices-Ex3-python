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

    def callRatio(self):
        # Add a column that holds the ratio battery_power/talk_time and name it call_ratio
        self.csv['call_ratio'] = self.csv['battery_power'] / self.csv['talk_time']

    def fromMBToGB(self):
        # Change the memory column to hold the memory in GB instead of MB
        self.csv['memory'] = self.csv['memory'] / 1024

    def printDescription(self):
        # Show table description
        print(self.csv.describe())

    def getPricesHistogram(self):
        # Prices histogram
        values = self.csv['price']
        plt.hist(values, bins=15)
        plt.title('Price histogram')
        plt.xlabel("Price")
        plt.ylabel("Sum")
        plt.show()

    def dataHeatMap(self):
        # Plot a correlation heatmap of the data set
        corr = self.csv.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5)
        plt.savefig('plt.png')
        plt.show()

    def getCategoricalCorrelation(self, feature):
        # Checks if a categorical feature has correlation with the price
        csv_oh = pd.concat([self.csv, pd.get_dummies(self.csv[feature])], axis=1)
        corr = csv_oh.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(200, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                    square=True, linewidths=.5)
        plt.title("Categorical correlation with "+ feature)
        plt.savefig(str(feature) + '-catCorr')
        plt.show()

    def plotRelationshipWithPrice(self, feature):
        # For each feature correlated with the price, plot its relationship with price
        sns.regplot(x=feature, y="price", data=self.csv, scatter_kws={"color": "black"}, line_kws={"color": "red"})
        plt.title("price " + feature + " relationship")
        plt.savefig('plot.png')
        plt.show()

    def threeCorrelatedFeatures(self):
        # Select 3 features(ram, battery_power and gen) that are correlated with price and create a pivot table showing
        # average price with relation to cross sections of those 3 features
        ram = pd.qcut(self.csv['ram'], 4)
        battery = pd.qcut(self.csv['battery_power'], 4)
        pivot = self.csv.pivot_table(index='gen', columns=[battery, ram],
                                     aggfunc={'price': 'mean'})
        print(pivot.unstack())

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

    def nominalToBinary(self):
        # For each nominal feature <N>, add a binary column OR one-hot encoding to the dataframe
        # representing the original values. Name binary columns <N>_bin, and prefix one-hot encodings with <N>
        bluetooth = pd.get_dummies(self.csv.bluetooth, prefix='bluetooth',drop_first=True)
        self.csv[str(bluetooth.columns.array[0]) + '_bin'] = bluetooth

        screen = pd.get_dummies(self.csv.screen, prefix='screen',drop_first=True)
        self.csv[str(screen.columns.array[0]) + '_bin'] = screen

    def deleteRedundantIndex(self):
        # Delete the redundant index on dataframe
        del self.csv['bluetooth']
        del self.csv['screen']
        del self.csv['sim']
        del self.csv['speed']
        del self.csv['gen']
        del self.csv['cores']
        del self.csv['wifi']

    def saveIntoCsvFile(self):
        # write the data frame into a csv file
        self.deleteRedundantIndex()
        self.csv.to_csv('mobile_prices_converted.csv', sep=',')

    def relationshipBetweenFourFeatures(self, feature1, feature2, feature3, feature4):
        # Choose 4 features and use a 2-d plot to show the relationships between each pair
        tmp_dataframe = self.csv[[feature1, feature2, feature3, feature4]].copy()
        sns.pairplot(tmp_dataframe, kind='reg', height=5, corner=True, plot_kws={'line_kws': {'color': 'red'}})
        plt.show()

    def plotRelationshipBetweenWidthHeightPriceCore(self):
        # Plot the relationship between px_width, px_height, price and core. Px_width and px_height should be
        # the X and Y coordinates respectively
        width, height = self.csv.px_width, self.csv.px_height
        price, cores = self.csv.price, self.csv.cores_ord

        plt.figure(figsize=(20, 15))

        plt.scatter(width, height, c=price, s=cores * 130, alpha=0.5, linewidths=0, cmap='coolwarm')
        plt.title('Plot of pixel width and height with price and cores')
        plt.xlabel('pixel width')
        plt.ylabel('pixel_height')
        plt.colorbar(label='price')
        plt.show()

    def getConclusions(self):
        csv2 = pd.read_csv('mobile_price_2.csv')

        # New column with relation between prices
        self.csv['newPriceRel'] = csv2.price_2 / self.csv.price
        plt.figure()
        plt.scatter(csv2.id, self.csv['newPriceRel'])
        plt.xlabel('id')
        plt.ylabel('newPriceRel')
        plt.title("Relationship between price_2 and price")
        plt.savefig('conclusion')
        plt.grid()
        plt.show()

        # See correlation between prices relationship and rest of the columns
        self.dataHeatMap()


        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(self.csv)