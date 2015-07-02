#!/usr/bin/python
# coding: utf-8
'''
Created on 14 avr. 2015

@author: stefan.duprey
'''
import urllib2
import codecs
from lxml import etree
import socket
urlfile = open('/home/sduprey/My_Data/My_List_Crawler/urllist.txt', 'r')
urllist = urlfile.readlines()
resultat = codecs.open('/home/sduprey/My_Data/My_List_Crawler/resultat.txt', 'w', 'utf-8')
num_lines = len(urllist)
resultat.write("source|destination|ztd\n")
count = 0
for my_url in urllist:
    count = count + 1
    if count > 1 :
        try:
            print 'Requesting URL : ' +  my_url
            response = urllib2.urlopen(my_url)
            reponse = response.read()
            user_agent = 'Python-crawler'
            headers = { 'User-Agent' : user_agent }
            tree = etree.HTML(reponse)
            code = str(response.getcode())
            url = str(response.geturl())
            my_url = my_url.replace("\n", "")
            ztd = tree.xpath('//p[@class="lpZtdTxt"]/text()')
            ztd=','.join(ztd)   
            print (str(100*count/num_lines)+" % --- "+my_url+"|"+url)
            resultat.write(my_url+"|"+url+"|"+ztd+"\n")
        except socket.timeout, e:
            print("There was an error: %r" % e)
            print (str(100*count/num_lines)+" % --- "+my_url+"|"+url)
            resultat.write(my_url+"|"+url+"|"+"Erreur !"+"\n")
        except urllib2.HTTPError, e:
            print('Erreur - %s.' % e.code)
            print (str(100*count/num_lines)+" % --- "+my_url+"|"+url)
            resultat.write(my_url+"|"+url+"|"+"Erreur !"+"\n")
print("Terminé !")
resultat.close()
urlfile.close()
