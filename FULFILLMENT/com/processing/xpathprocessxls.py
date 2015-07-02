from xlrd import open_workbook
import urllib2
import csv
from lxml import etree

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
    given_shop, given_skuid, given_pricing, given_remaining, skuid = parse_xls('/home/sduprey/My_Data/My_Market_Place_Data/sku_list.xlsx')
    # we open 
    b = open('/home/sduprey/My_Data/My_Market_Place_Data/sku_list_results.csv', 'w')
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
        tree = etree.parse(my_url, parser)
        vendor_cells = tree.xpath('//div[3]/div[3]/div/div/a/text()')
        price_cells = tree.xpath('//div[3]/div[3]/div/div/form/p[@class="price"]')
        vendor_price = {}
        print len(price_cells)
        print len(vendor_cells)
        # we here map vendor and price from the crawl using a dictionnare
        index=0
        for vending_machine in vendor_cells:
            vending_price = price_cells[index].text
            vendor_price[vending_machine]=vending_price
            index += 1
            
        # we here look if we find the requested vendor   
        matching =  [s for s in vendor_cells if given_shopname in s]
        matching_price=''
        status=''
        if len(matching) == 1:
            print 'OK vendor found'
            matching_price = vendor_price[given_shopname]
            status='OK'
        else :
            print 'KO vendor not found'
            status='KO'
        to_write = [sku,given_shopname,given_price,given_remaining_quantity,status,matching_price];
        csv_writer.writerow(to_write)
        # writing a csv result file

    #results have now been written, we close the file
    csv_writer.close()
  