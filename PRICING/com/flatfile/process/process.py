'''
Created on 4 May 2015

@author: sduprey
'''
from pyquery import PyQuery as pq
import csv
import codecs

def process_url_row(urlcdiscount):
    print('Processing URL : ' +urlcdiscount)
    d = pq(urlcdiscount)
    prices = []
    
    for price in d("div[class='productInfos clearfix']"):
        innerprices = pq(price)
        for innerprice in innerprices('.currentPrice'):
            prices.append(innerprices(innerprice).text())
    
    #for price in d('.currentPrice'):
    #    prices.append(d(price).text())
    if len(prices) > 0 :
        return prices[0]
    else:
        return 'No price found'
    
def main():
    # here change to the path of your files
    my_path = '/home/sduprey/My_Data/My_Pricing/'
    my_complete_entering_path = my_path+'Exemple_fichier_pour_recup_prix_confo.csv'
    my_complete_outputing_path = my_path+'pricesResults.txt'
    pricesResults = codecs.open(my_complete_outputing_path, "w", "utf-8")
    with open(my_complete_entering_path, 'rb') as csvfile:
        urlToPriceReader = csv.reader(csvfile, delimiter=';', quotechar='|')
        next(urlToPriceReader, None)  # skip the headers
        for row in urlToPriceReader: 
            my_utf8_row = [unicode(cell, 'utf-8') for cell in row]          
            try:
                price=process_url_row(my_utf8_row[2])
                print 'Price found : '+price
                pricesResults.write(';'.join(my_utf8_row)+';'+price+"\n")
            except KeyboardInterrupt:
                print("Interrupt received, proceeding")
                urlToPriceReader.close()
                pricesResults.close()
    pricesResults.close()
if __name__ == "__main__":
    main()