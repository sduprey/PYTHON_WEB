# curating data where libelle and description are the same
SELECT libelle, description, count(*) FROM training_data GROUP BY libelle, description

-- title description identique => -500 000 lignes (restent 5.1 M) - 20'
delete FROM training where product_id in (
	select training.product_id FROM training 
	LEFT OUTER JOIN (
		SELECT MIN(product_id) as id, title, description
		FROM training
		GROUP BY title, description
	    ) as t1 
	    ON training.product_id = t1.id
	WHERE t1.id IS NULL
);

select * FROM training where product_id in (
	select training.product_id FROM training 
	LEFT OUTER JOIN (
		SELECT MIN(product_id) as id, title, description
		FROM training
		GROUP BY title, description
	    ) as t1 
	    ON training.product_id = t1.id
	WHERE t1.id IS NULL
);

select count(distinct categorie_3) from training_data where produit_cdiscount = true;

select * into cleaned_training_data from training_data where categorie_3 not in (select distinct categorie_3 from training_data where produit_cdiscount = true) ;
select * into cleaned_training_data from training_data where produit_cdiscount = true;

CREATE TABLE IF NOT EXISTS CLEANED_TRAINING_DATA (
IDENTIFIANT_PRODUIT VARCHAR(50), 
CATEGORIE_1 VARCHAR(50), 
CATEGORIE_2 VARCHAR(50), 
CATEGORIE_3 VARCHAR(50), 
DESCRIPTION TEXT,
LIBELLE VARCHAR(200),
MARQUE VARCHAR(150),
PRODUIT_CDISCOUNT BOOLEAN,
PRIX NUMERIC,
IS_IN_TFIDF_INDEX BOOLEAN,
TO_FETCH BOOLEAN) TABLESPACE mydbspace

INSERT INTO CLEANED_TRAINING_DATA(IDENTIFIANT_PRODUIT, CATEGORIE_1, CATEGORIE_2, CATEGORIE_3, DESCRIPTION, LIBELLE, MARQUE, PRODUIT_CDISCOUNT,PRIX,IS_IN_TFIDF_INDEX,TO_FETCH) 
select * from training_data where produit_cdiscount = true;
INSERT INTO CLEANED_TRAINING_DATA(IDENTIFIANT_PRODUIT, CATEGORIE_1, CATEGORIE_2, CATEGORIE_3, DESCRIPTION, LIBELLE, MARQUE, PRODUIT_CDISCOUNT,PRIX,IS_IN_TFIDF_INDEX,TO_FETCH) 
select * from training_data where categorie_3 not in (select distinct categorie_3 from training_data where produit_cdiscount = true);

# shrinking uniformly
select identifiant_produit as id, categorie_3_id as output, categorie_3 as output_string, concat_ws(' ', description::text, libelle::text) as description  from UNIFORMLY_RESTRAINED_TRAINING_DATA
select identifiant_produit as id, concat_ws(' ', description::text, libelle::text) as description  from TESTING_DATA