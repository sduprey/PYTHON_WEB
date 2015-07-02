from scrapy.item import Item, Field


class CorpuscrawlerItem(Item):
    url = Field()
    contenu = Field()
    titre = Field()
    pass
