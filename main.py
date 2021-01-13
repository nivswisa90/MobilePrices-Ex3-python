from task1 import *


def main():
    suma = Summary("mobile_price_1.csv")
    suma.addResolution()
    suma.dotsPerInch()
    suma.callRatio()
    suma.fromMBToGB()


    # suma.printDescription()
    # suma.getPricesHistogram()
    # suma.dataHeatMap()
    # suma.getCategoricalCorrelation('bluetooth')
    # suma.plotRelationshipWithPrice('resolution')
    # suma.threeCorrelatedFeatures()
    suma.ordinalToNumerical()
    suma.nominalToBinary()
    suma.dataHeatMap()
    # suma.saveIntoCsvFile()


if __name__ == '__main__':
    main()

