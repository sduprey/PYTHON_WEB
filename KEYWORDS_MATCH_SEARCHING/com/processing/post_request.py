import requests
LOC = "/home/sduprey/My_Data/"
DATA_DIR = LOC + "My_Acquisition_Data/"
def extract(query, nb, flag = "sl0", loc = DATA_DIR):
    """
    Extract exalead data
    
    Param:
    -------
    flag: str
        model name, such as "sl0" for prod, "sltestmetier" for test
    query: str
        target query
    nb: int
        sku number demanded
    """    
    query = "+".join(query.split(" "))
    
    data={}
    data['lang'] = 'fr'
    data['sl'] = flag
    
    data['f.20.field'] = 'facet_mut_technical'
    data['f.20.in_hits'] = 'False'
    data['f.20.in_synthesis'] = 'False'
    data['f.20.max_per_level'] = 10
    # data['f.20.root'] = 'Top%2FClassProperties%2Fis_best_total_offer'
    data['f.20.root'] = 'Top/ClassProperties/is_best_total_offer'
    data['f.20.sort'] = 'num'
    data['f.20.type'] = 'category'
    # data['refine'] = '%2Bf%2F20%2F1'
    data['refine'] = '+f/20/1'
    
    data['use_logic_facets'] = 'false'
    data['use_logic_hit_metas'] = 'false'
    
    # another field: 'offer_flash_sale_type_copy'
    data['add_hit_meta'] = ['offer_product_id', 'offer_price', 'offer_seller_id']
    # data['add_hit_meta'] = 'offer_price'
    
    data['hit_meta.termscore.expr'] = '100000*@term.score'
    data['hit_meta.proximity.expr'] = '@proximity'
    data['hit_meta.categoryweight.expr'] = '100000*offer_category_weight'
    data['hit_meta.ca14.expr'] = 'offer_stats_income14_global'
    data['hit_meta.ca7.expr'] = 'offer_stats_income7_global'
    data['hit_meta.ca1.expr'] = 'offer_stats_income1_global'
    data['output_format'] = 'csv'
    data['nresults'] = nb
    data['q'] = query
   
    site = "http://ldc-exa6-search01.cdweb.biz:10010/search-api/search"
    # site = "http://exasearchv6.gslb.cdweb.biz:10010/search-api/search"
    try:
        req = requests.post(site, data)
    except:
        print query
        raise ValueError
        
    filename = "result_%d_%s.csv"%(nb, query)
    with open(loc + filename, 'w') as f:
        f.write(req.content)
    return loc + filename