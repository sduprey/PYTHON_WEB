from scrapy.contrib.exporter import CsvItemExporter
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.selector import HtmlXPathSelector
from twisted.web import http

from cdscrawler.items import cdscrawlerItem


class MyCrawlerSpider(CrawlSpider):
   
    name = 'cdscrawler'

    allowed_domains = ['www.cdiscount.com']

 
    start_urls = ['http://www.cdiscount.com/']
    handle_httpstatus_list = [404]
   # handle_httpstatus_list = [301]
	
    rules = [
	Rule(SgmlLinkExtractor(allow=[r'/lf-']), follow=True, callback='parse_item'),
	Rule(SgmlLinkExtractor(allow=[r'/l-', r'/v-'], deny=[r'/f-', r'/l-(\d+)-(\d+).html']), follow=True),
        ]

    def parse_item(self, response):
        hxs = HtmlXPathSelector(response)
        item = cdscrawlerItem()
	item['URL']= response.url
	item['CodeHttp']= response.status
        return item
   
    def response_status_message(self,status):
    	return '%s %s' % (status, http.responses.get(int(status)))

