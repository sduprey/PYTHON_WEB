# Scrapy settings for cdscrawler project
#
# For simplicity, this file contains only the most important settings by
# default. All the other settings are documented here:
#
#     http://doc.scrapy.org/topics/settings.html
#

BOT_NAME = 'cdscrawler'

SPIDER_MODULES = ['cdscrawler.spiders']
NEWSPIDER_MODULE = 'cdscrawler.spiders'
#LOG_ENABLED = False
LOG_LEVEL = 'DEBUG'
CONCURRENT_REQUESTS = 200
CONCURRENT_ITEMS = 200
DEPTH_PRIORITY = 0
DEPTH_STATS = True
DEPTH_STATS_VERBOSE = True
#LOG_FILE = "logtest.csv"



#DEPTH_LIMIT = 0
COOKIES_ENABLED = False


# Store
FEED_URI='mycrawler-results.csv'
FEED_FORMAT='csv'

# Crawl responsibly by identifying yourself (and your website) on the user-agent
USER_AGENT = 'cdiscountbotNocache'
