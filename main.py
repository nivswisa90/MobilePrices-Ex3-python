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
    # suma.getCategoricalCorrelation('cores')
    # suma.plotRelationshipWithPrice('resolution')
    # suma.threeCorrelatedFeatures()
    suma.ordinalToNumerical()
    suma.nominalToBinary()
    # suma.dataHeatMap()
    # suma.saveIntoCsvFile()
    # suma.relationshipBetweenFourFeatures('price', 'battery_power', 'px_height', 'px_width')
    # suma.plotRelationshipBetweenWidthHeightPriceCore()
    suma.getConclusions()


if __name__ == '__main__':
    main()

