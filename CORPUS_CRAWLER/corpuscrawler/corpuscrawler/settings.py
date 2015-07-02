# -*- coding: utf-8 -*-

# Scrapy settings for corpuscrawler project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#

BOT_NAME = 'corpuscrawler'

SPIDER_MODULES = ['corpuscrawler.spiders']
NEWSPIDER_MODULE = 'corpuscrawler.spiders'

CONCURRENT_REQUESTS = 2
#nombre de scrap simultané

CONCURRENT_REQUESTS_PER_DOMAIN = 1
#nombre de scrap simultané par nom de domaine

LOG_LEVEL = 'DEBUG'
#verbeux

COOKIES_ENABLED = False
#pas de cookie

RETRY_ENABLED = True
# en cas d'échec lors du scrap, retry

DOWNLOAD_TIMEOUT = 600
# attente avant échec

REDIRECT_ENABLED = True
#redirection active

REDIRECT_MAX_TIMES = 5
# nombre de redirection maximales

ROBOTSTXT_OBEY = True
DEPTH_LIMIT = 20

#HTTP_PROXY = 'http://127.0.0.1:8118'
#FEED_URI = '/var/www/scrapy/corpus/corpus.json'
#FEED_FORMAT = 'jsonlines'
#FEED_EXPORTERS_BASE = {
#    'jsonlines': 'scrapy.contrib.exporter.JsonLinesItemExporter',
#}


#DOWNLOADER_MIDDLEWARES = {
#     'corpus.middlewares.ProxyMiddleware': 410,
#     'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware': None
#}

#DOWNLOADER_MIDDLEWARES_BASE = {
#    'scrapy.contrib.downloadermiddleware.robotstxt.RobotsTxtMiddleware': 100,
#    'scrapy.contrib.downloadermiddleware.redirect.RedirectMiddleware':120,
#    'scrapy.contrib.downloadermiddleware.downloadtimeout.DownloadTimeoutMiddleware':200,
#    'scrapy.contrib.downloadermiddleware.retry.RetryMiddleware':250,
#}
#USER_AGENT = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'