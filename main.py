from task1 import *


def main():
    suma = Summary("mobile_price_1.csv")
    suma.addResolution()
    suma.dotsPerInch()
    suma.callRatio()
    suma.fromMBToGB()
    # suma.printDescription()
    # suma.getHistogram()
    # suma.dataHeatMap()
    # suma.getCategoricalCorrelation()
    # suma.threeCorrelatedFeatures()
    suma.ordinalToNumerical()


if __name__ == '__main__':
    main()

