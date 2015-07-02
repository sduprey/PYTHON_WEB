from xlrd import open_workbook
from pyquery import PyQuery as pq
import urllib2
import csv
from lxml import etree
import sys

def parse_xls(my_workbook_name):
    wb = open_workbook(my_workbook_name)
    # the workbook must have only one sheet with the data in it
    sheet = wb.sheet_by_index(0)
    print 'Sheet:',sheet.name
    # Displaying the sheet information
    shopname=[]
    skuid=[]
    integrationprice=[]
    remaining_quatity=[]
    column_values = []
    given_pricing = {}
    given_shop = {}
    given_remaining = {}
    for row in range(sheet.nrows):
        if row == 0:
            for col in range(sheet.ncols):
                column_values.append(sheet.cell(row,col).value)
            print ','.join(column_values)
        else:
            given_shopname=''
            given_skuid=''
            given_price=''
            given_remaining_quantity=''
            for col in range(sheet.ncols):
                if col == 0:
                    given_shopname=sheet.cell(row,col).value
                    shopname.append(sheet.cell(row,col).value)
                if col == 1:
                    given_skuid=sheet.cell(row,col).value
                    skuid.append(sheet.cell(row,col).value)
                if col == 2:
                    given_price=sheet.cell(row,col).value
                    integrationprice.append(sheet.cell(row,col).value)  
                if col== 3:
                    given_remaining_quantity=sheet.cell(row,col).value
                    remaining_quatity.append(sheet.cell(row,col).value)  
            given_pricing[given_skuid]=given_price
            given_shop[given_skuid]=given_shopname
            given_remaining[given_skuid]=given_remaining_quantity
    return given_shop, given_skuid, given_pricing, given_remaining, skuid

def processurl(myurl):
    content = urllib2.urlopen(myurl).read()
    return content
 
if __name__ == '__main__':
    print 'Number of arguments:', len(sys.argv), 'arguments.'
    if len(sys.argv) != 3:
        print 'You must specify the python file then the input xls file and then the output csv file \n'
        print 'Example : python jqueryprocessxls.py /home/sduprey/My_Data/My_Market_Place_Data/sku_list.xlsx /home/sduprey/My_Data/My_Market_Place_Data/sku_list_results.csv'
    my_path = sys.argv[1];
    output_path = sys.argv[2];
    given_shop, given_skuid, given_pricing, given_remaining, skuid = parse_xls(my_path)
    # we open 
    b = open(output_path, 'w')
    csv_writer = csv.writer(b)
    # we here loop over the skuid
    for sku in skuid:
        # we here build a dedicated URL
        given_price=given_pricing[sku]
        given_remaining_quantity=given_remaining[sku]
        given_shopname=given_shop[sku]
        
        print sku
        print given_price
        print given_remaining_quantity
        print given_shopname
        parser = etree.HTMLParser()
        my_url='http://www.cdiscount.com/mp-9000-'+sku+'.html'
        d = pq(my_url)
        # we here fetch vendor and prices
        resellers = []
        prices = []
                
        for reseller in d('.slrName'):
            resellers.append(d(reseller).text())
            
        for price in d('p.price'):
            prices.append(d(price).text())
 
 
        if len(prices) != len(resellers):
            print 'Number of vendors does not fit the number of prices : some problem might have happened in XPATH'
        vendor_price_mapping = {}
        index=0
        for reseller in resellers:
            vendor_price_mapping[reseller]=prices[index]
            index += 1
        # we here map vendor and price from the crawl using a dictionnare
          
        # we here look if we find the requested vendor   
        matching =  [s for s in resellers if given_shopname in s]
        matching_price=''
        status=''
        if len(matching) == 1:
            print 'OK vendor found'
            matching_price = vendor_price_mapping[given_shopname]
            print matching_price
            matching_price=matching_price.replace(u'\u20ac','.');
         #   matching_price=matching_price.replace('\\u20ac','.');
            print matching_price
            status='OK'
        else :
            print 'KO vendor not found'
            status='KO'
        # we skip non unicode in reseller trade mark
        given_shopname=given_shopname.replace(u'\xae','')    
        given_shopname=given_shopname.replace(u'\xe9','')    
        
        to_write = [sku,given_shopname,given_price,given_remaining_quantity,status,matching_price];
        print sku
        print given_shopname
        print given_price
        print given_remaining_quantity
        print status
        print matching_price
        csv_writer.writerow(to_write)
        # writing a csv result file

    #results have now been written, we close the file
    b.close()
  