def locate_best_serp_inspector(google_request): return 
def bring_back_local_mesh(url): return
def aggregate_semantics_for_request(list_urls): return
def ecommerce_page_type_classifier(url): page_type
def isSemanticFoundInCdiscount(list_urls) : return 
def market_place_create_offer(semantics): return 
def edito_create_content(semantics): return 
def checkPandaRatio(semantics): return
def checkPenguinRatio(semantics) : return 
def checkThematicPageRank(semantics): return 
def getOutSiteIncomingLinks(CONCURRENCY_URL_REFERENTIAL) : return
def etalonnate_quality_ranker(panda_ratios,penguin_ratios,semantics): return 
def concurrency_detection(): CDISCOUNT_CONCURRENTS = []; return CDISCOUNT_CONCURRENTS
def semantic_crawl(CONCURRENCY_REQUEST_REFERENTIAL): return CONCURRENCY_REQUEST_REFERENTIAL
def search_engine_completion_scraping(CONCURRENCY_REQUEST_REFERENTIAL): return CONCURRENCY_REQUEST_REFERENTIAL
def etalonnate_page_type_classifier(CONCURRENCY_URL_REFERENTIAL) : return
def populate_semantic_referential(CDISCOUNT_CONCURRENTS) :
    CONCURRENCY_URL_REFERENTIAL={};# (url, semantics)
    CONCURRENCY_REQUEST_REFERENTIAL=[]; # [google_requests]
    for concurrent in CDISCOUNT_CONCURRENTS :  
        # semantic crawl (TF/IDF, NGRAMS, TITLE, H1, METADESCR, ANCHORS, ANCHORS TF_IDF) 
        # and (TF PAGE, TF ANCHORS) for the panda & penguin filters
        # classifier features : DEPTH,NB_OUTLINKS,NB_INLINKS,PAGE_RANK, NB_URL_UNDER, NB_BREADCRUMBS,NB_AGGREGATED_RATINGS,NB_RATINGS_VALUES,
        # NB_PRICES,NB_AVAILABILITIES,NB_REVIEWS,NB_IMAGES,NB_SEARCH_IN_URL,NB_ADD_IN_TEXT,NB_FILTER_IN_TEXT,NB_SEARCH_IN_TEXT,NB_GUIDE_ACHAT_IN_TEXT,
        # NB_PRODUCT_INFO_IN_TEXT, NB_LIVRAISON_IN_TEXT, NB_GARANTIES_IN_TEXT, NB_PRODUITS_SIMILAIRES_IN_TEXT,NB_IMAGES_TEXT
        CONCURRENCY_URL_REFERENTIAL,CONCURRENCY_REQUEST_REFERENTIAL= semantic_crawl(CONCURRENCY_REQUEST_REFERENTIAL,
                                                                        CONCURRENCY_URL_REFERENTIAL, concurrent);
        # getting incoming links from outside thanks to Majestic
        CONCURRENCY_URL_REFERENTIAL=getOutSiteIncomingLinks(CONCURRENCY_URL_REFERENTIAL)
        # request built from NGRAMS, TITLE, H1, ANCHORS from the site crawl
        # request built from auto completion and suggestion
        CONCURRENCY_REQUEST_REFERENTIAL=search_engine_completion_scraping(CONCURRENCY_REQUEST_REFERENTIAL, concurrent);
    return CONCURRENCY_URL_REFERENTIAL, CONCURRENCY_REQUEST_REFERENTIAL;

if __name__ == '__main__':
    # concurrency data collected from SearchMetrics
    CDISCOUNT_CONCURRENTS = concurrency_detection() 
    # we then use site crawling and auto-completion scraping to build our semantic request referential
    CONCURRENCY_URL_REFERENTIAL,CONCURRENCY_REQUEST_REFERENTIAL=populate_semantic_referential(CDISCOUNT_CONCURRENTS)
    # we etalonnate a classifier thanks to the data previously collected
    PAGE_TYPE_CLASSIFIER = etalonnate_page_type_classifier(CONCURRENCY_URL_REFERENTIAL)
     # we then inspect the concurrency position and how we can remedy to dislodge them
    for request in CONCURRENCY_REQUEST_REFERENTIAL :
        # the most difficult part to scale : you need proxys and multiple IPs to request heavily google
        url = locate_best_serp_inspector(request)
        page_type = PAGE_TYPE_CLASSIFIER.classify(url)
        list_urls = bring_back_local_mesh(url)
        semantics = aggregate_semantics_for_request(list_urls)
        if not isSemanticFoundInCdiscount(semantics) and page_type is 'FicheProduit' :
            # vendors must create offers
            market_place_create_offer(semantics)
            # the page is navigation, products list, search results
        elif not isSemanticFoundInCdiscount(semantics) and not page_type is 'FicheProduit' :           
            # edito must create content
            edito_create_content(semantics)
        else :
            # check the panda filter is not hitting
            panda_ratios=checkPandaRatio(semantics)#TF body page
            # check the penguin filter is not hitting
            penguin_ratios=checkPenguinRatio(semantics)#TF anchors
            checkThematicPageRank(semantics)# semantic similarity between links
            # try to build a page quality classifier
            etalonnate_quality_ranker(panda_ratios,penguin_ratios,semantics)