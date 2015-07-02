'''
Created on 24 Apr 2015

@author: sduprey
'''
import urllib2
import csv
import socket
import time
import sys, traceback

def process_url_to_file(urlcdiscount):
    urlcdiscount=urlcdiscount.rstrip();
    user_agent = 'CdiscountBot-crawler'
    print("User agent choisi : "+user_agent)
    headers = { 'User-Agent' : user_agent }
    request = urllib2.Request(urlcdiscount,None,headers)
    # we just don't need cookies or proxy
#    cj = cookielib.CookieJar()
#    ck = cookielib.Cookie(version=0, name='Cassandra', value='1', port=None, port_specified=False, domain='www.cdiscount.com', domain_specified=False, domain_initial_dot=False, path='/', path_specified=True, secure=False, expires=None, discard=True, comment=None, comment_url=None, rest={'HttpOnly': None}, rfc2109=False)
#    cj.set_cookie(ck)
#    opener = urllib2.build_opener(urllib2.ProxyHandler({"http":"http://localhost:3128"}), urllib2.HTTPCookieProcessor(cj))
#    urllib2.install_opener(opener)
    data = urllib2.urlopen(request).read()
   
def main():
    filepath = '/home/sduprey/My_Data/My_GWT_Scores/my_extract_2015_23_04.txt' 
    with open(filepath,'rb') as fichierurlscores:
        # the quote char is so awesome
        #spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        urlscorereader=csv.reader(fichierurlscores, delimiter=';')
        for row in urlscorereader:
            print ', '.join(row)
            url = 'http://www.cdiscount.com'+row[0].rstrip()
            print 'Processing URL : '+ url
            score = row[1]
            print 'Having score '+str(score)
            try:
                process_url_to_file(url)
            except socket.timeout, e:
                print("socket timeout")
            except urllib2.HTTPError, e:
                print 'you got caught' 
                traceback.print_exc(file=sys.stdout)
                time.sleep(3600)
            except KeyboardInterrupt:
                print("Interrupt received, proceeding")
                fichierurlscores.close()
            
        fichierurlscores.close()

if __name__ == "__main__":
    main()