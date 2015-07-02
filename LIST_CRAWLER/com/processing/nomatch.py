#!/usr/bin/python
# coding: utf-8
import urllib2
from lxml import etree

fichier = open("D:\My_Python_Workspace\LIST_CRAWLER\list.txt", "r")
urllist = fichier.readlines()
"""print(urllist)"""

for i in urllist:
                #response = urllib2.urlopen(i)
                #html = response.read()
                
                user_agent = 'CdiscountBot-crawler'
                headers = { 'User-Agent' : user_agent }
                parser = etree.HTMLParser()
                #code = str(response.getcode())
                #url = str(response.geturl())
             #   tree = etree.parse(html)
                tree = etree.parse(i, parser)
                xpathtest = tree.XPath("//text()")
               
                vendor_cells = tree.xpath('//div[3]/div[3]/div/div/a/text()')
                price_cells = tree.xpath('//div[3]/div[3]/div/div/form/p[@class="price"]')
               # print(url+"|"+code+"|"+xpathtest)

