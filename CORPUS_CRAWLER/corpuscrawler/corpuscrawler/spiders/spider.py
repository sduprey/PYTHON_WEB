from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor
from scrapy.selector import HtmlXPathSelector
from corpuscrawler.items import CorpuscrawlerItem
from scrapy.selector import Selector
import urlparse

class MyCrawlerSpider(CrawlSpider):
    name="corpuscrawler"
    allowed_domains  = ['www.gamekult.com']

    start_urls = ['http://www.gamekult.com']
    rules = (Rule (SgmlLinkExtractor(allow=('/jeux/'),deny=('/forum/'),restrict_xpaths=()), callback="parse_o", follow= True),)
    def parse_o(self,response):
        sel = Selector(response)
        item = CorpuscrawlerItem()
        item['url'] = response.url
        item['contenu'] = sel.xpath('//div[@class="summary"]').extract()
        item['contenu']+= sel.xpath('//div[@class="story-body"]').extract()
        item['contenu']+= sel.xpath('//div[@class="story-conclusion"]').extract()
        item['titre'] = sel.xpath('//title/text()').extract()
        yield item 
        


