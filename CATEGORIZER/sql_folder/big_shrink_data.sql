drop table IF EXISTS stop_words;
CREATE TABLE stop_words (word varchar);
insert into stop_words values ('alors');
insert into stop_words values ('au');
insert into stop_words values ('aussi');
insert into stop_words values ('avec');
insert into stop_words values ('avoir');
insert into stop_words values ('car');
insert into stop_words values ('ce');
insert into stop_words values ('cela');
insert into stop_words values ('ces');
insert into stop_words values ('chaque');
insert into stop_words values ('ci');
insert into stop_words values ('comme');
insert into stop_words values ('dans');
insert into stop_words values ('des');
insert into stop_words values ('du');
insert into stop_words values ('depuis');
insert into stop_words values ('donc');
insert into stop_words values ('elle');
insert into stop_words values ('elles');
insert into stop_words values ('est');
insert into stop_words values ('et');
insert into stop_words values ('fois');
insert into stop_words values ('il');
insert into stop_words values ('ils');
insert into stop_words values ('je');
insert into stop_words values ('le');
insert into stop_words values ('la');
insert into stop_words values ('les');
insert into stop_words values ('ma');
insert into stop_words values ('me');
insert into stop_words values ('mes');
insert into stop_words values ('mon');
insert into stop_words values ('meme');
insert into stop_words values ('ni');
insert into stop_words values ('ou');
insert into stop_words values ('pas');
insert into stop_words values ('peu');
insert into stop_words values ('pour');
insert into stop_words values ('que');
insert into stop_words values ('qui');
insert into stop_words values ('quel');
insert into stop_words values ('quelle');
insert into stop_words values ('sa');
insert into stop_words values ('sans');
insert into stop_words values ('si');
insert into stop_words values ('ses');
insert into stop_words values ('son');
insert into stop_words values ('sous');
insert into stop_words values ('sur');
insert into stop_words values ('ta');
insert into stop_words values ('tes');
insert into stop_words values ('ton');
insert into stop_words values ('tu');
insert into stop_words values ('ete');
insert into stop_words values ('etre');
insert into stop_words values ('-');
insert into stop_words values ('+');
insert into stop_words values ('&');
insert into stop_words values ('*');
insert into stop_words values ('/');
insert into stop_words values (':');
insert into stop_words values ('.');
insert into stop_words values ('!');
insert into stop_words values ('?');

CREATE OR REPLACE FUNCTION unaccent_string(text)
RETURNS text
IMMUTABLE
STRICT
LANGUAGE SQL
AS $$
SELECT translate(
    $1,
    'âãäåàÁÂÃÄÅèééêëÈÉÉÊËìíîïìÌÍÎÏÌóôõöÒÓÔÕÖùúûüÙÚÛÜ',
    'aaaaaAAAAAeeeeeEEEEEiiiiiIIIIIooooOOOOOuuuuUUUU'
);
$$;

DROP FUNCTION IF EXISTS replace_stopword_rayon();
CREATE OR REPLACE FUNCTION replace_stopword_rayon() RETURNS void AS
$$
DECLARE
    word varchar;
BEGIN
  FOR word IN select sw.word from stop_words sw LOOP
	update rayons set category3_label = replace(category3_label, word||' ', '') where category3_label like word||' %';
	update rayons set category3_label = replace(category3_label, ' '||word||' ', ' ') where category3_label like '% '||word||' %';
	update rayons set category3_label = replace(category3_label, ' '||word, '') where category3_label like '% '||word;
  END LOOP;    
END
$$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS replace_stopword_test();
CREATE OR REPLACE FUNCTION replace_stopword_test() RETURNS void AS
$$
DECLARE
    word varchar;
BEGIN
  FOR word IN select sw.word from stop_words sw LOOP
	update test set description = replace(description, word||' ', '') where description like word||' %';
	update test set description = replace(description, ' '||word||' ', ' ') where description like '% '||word||' %';
	update test set description = replace(description, ' '||word, '') where description like '% '||word;
	update test set title = replace(title, word||' ', '') where title like word||' %';
	update test set title = replace(title, ' '||word||' ', ' ') where title like '% '||word||' %';
	update test set title = replace(title, ' '||word, '') where title like '% '||word;	
	update test set brand = replace(brand, word||' ', '') where brand like word||' %';
	update test set brand = replace(brand, ' '||word||' ', ' ') where brand like '% '||word||' %';
	update test set brand = replace(brand, ' '||word, '') where brand like '% '||word;	
  END LOOP;    
END
$$
LANGUAGE 'plpgsql' ;

DROP FUNCTION IF EXISTS replace_stopword_training();
CREATE OR REPLACE FUNCTION replace_stopword_training() RETURNS void AS
$$
DECLARE
    word varchar;
BEGIN
  FOR word IN select sw.word from stop_words sw LOOP
	update training set description = replace(description, word||' ', '') where description like word||' %';
	update training set description = replace(description, ' '||word||' ', ' ') where description like '% '||word||' %';
	update training set description = replace(description, ' '||word, '') where description like '% '||word;
	update training set title = replace(title, word||' ', '') where title like word||' %';
	update training set title = replace(title, ' '||word||' ', ' ') where title like '% '||word||' %';
	update training set title = replace(title, ' '||word, '') where title like '% '||word;	
	update training set brand = replace(brand, word||' ', '') where brand like word||' %';
	update training set brand = replace(brand, ' '||word||' ', ' ') where brand like '% '||word||' %';
	update training set brand = replace(brand, ' '||word, '') where brand like '% '||word;	
  END LOOP;    
END
$$
LANGUAGE 'plpgsql' ;


/*
-------------------------------
Importation RAYONS -- 5''
-- création table, import CSV, suppression doublons, création d'index, nettoyage des libellés de la catégorie 3 (minuscules, sans accent, sans stop word)
-------------------------------
*/
-- suppression table
drop table IF EXISTS rayons;
-- création table
CREATE TABLE rayons 
(category1_id int, category1_label varchar, category2_id int, category2_label varchar,category3_id int, category3_label varchar);
-- import depuis le CSV
COPY public.rayons FROM '/home/mlasserre/Downloads/rayon.csv' WITH CSV HEADER DELIMITER AS ';';
-- supprimer les doublons sur la catégorie 3 dans la table des rayons
DELETE FROM rayons WHERE category3_label = 'BOITE A AIR' and category3_id=1000005746;
DELETE FROM rayons WHERE category3_label = 'BACHE DE REMORQUE VENDU SEULE' and category3_id=1000005332;
DELETE FROM rayons WHERE category2_label = 'DESSERTS-AIDE PATISSERIE' and category3_id=1000000640;
DELETE FROM rayons WHERE category2_label = 'CONSOLES / JEUX VIDEO' and category3_id=267;
DELETE FROM rayons WHERE category2_label = 'CONSOLES / JEUX VIDEO' and category3_id=257;
-- création d'index
CREATE INDEX category1_rayons_idx ON rayons (category1_id);
CREATE INDEX category2_rayons_idx ON rayons (category2_id);
CREATE UNIQUE INDEX category3_label_rayons_idx ON rayons (category3_id);

-- nettoyage du référentiel des rayons > catégorie 3
-- minuscule
update rayons set category3_label = lower(category3_label);
-- suppression des carcatères accentués
update rayons set category3_label = unaccent_string(category3_label);
-- suppression des stop words
select replace_stopword_rayon();

/*
-------------------------------
Importation TEST -- 19''
-- création table, import CSV, création d'index, nettoyage des libellés des champs titre, description et marque (minuscules, sans accent, sans stop word)
-------------------------------
*/
-- suppression table
drop table IF EXISTS test;
-- création table
CREATE TABLE test
(product_id int, description varchar, title varchar, brand varchar, price decimal);
-- import depuis le CSV
COPY public.test FROM '/home/mlasserre/Downloads/test.csv' WITH CSV HEADER DELIMITER AS ';';
-- création d'index
CREATE INDEX brand_test_idx ON test (brand);
CREATE UNIQUE INDEX product_id_test_idx ON test (product_id);
-- ajout d'une colonne pour la catégorie 3
ALTER TABLE test ADD COLUMN category3_id int;

-- nettoyage du référentiel des rayons > catégorie 3
-- minuscule
update test set description = lower(description), title = lower(title), brand = lower(brand);
-- suppression des carcatères accentués
update test set description = unaccent_string(description), title = unaccent_string(title), brand = unaccent_string(brand);
-- suppression des stop words
select replace_stopword_test();
-- suppression de la marque dans le titre et la description du produit
update test set title = btrim(replace(title, brand, ''), ' '), description = btrim(replace(description, brand, ''), ' ') where brand <> '';

/*
-------------------------------
Importation TRAINING - partie 1 -- 3'
-- création table, import CSV
-------------------------------
*/
-- suppression table
drop table IF EXISTS training;
-- création table
CREATE TABLE training
(product_id int, category1_id int, category2_id int, category3_id int, description varchar, title varchar, brand varchar, product_cdiscount varchar, price decimal);

--insert into training values(1, 1, 12, 123, 'dvd michael jordan', 'dvd michael jordan', 'jordan', '', 9.23);
--insert into training values(2, 1, 12, 123, 'le DVD de michael jordan à voir dans les bacs', 'jordan dvd michael jordan', 'jordan', '', 9.23);

-- import depuis le CSV
COPY public.training FROM '/home/mlasserre/Downloads/training.csv' WITH CSV HEADER DELIMITER AS ';';

/*
-------------------------------
Traitement empirique des grosses catégories pour réduire le volume de training -- 3'
-------------------------------
*/
-- compte le nb de produits par categ3 les plus peuplées ( > 100k produits)
/*
select t.category3_id, count(*)
from training t
group by t.category3_id
having count(*)>100000
order by 2 desc;
*/
-- en observant les données de ces catégories, on peut créer des règles simples pour les plus grandes catégories
-- si libellé produit like '%<terme>%' alors on a de fortes chances d'être dans la catégorie associée au terme
-- exemples de termes associés à ces catégories :
-- coque	1000010653
-- batterie	1000004085
-- chargeur	1000004079
-- etuit	1000010667
-- sticker	1000012993
-- montre	1000010170
-- shirt	1000010533

-- mettre à jour la catégorie 3 des produits de test dont le titre possède les mots-clés des grosses catégories
update test set category3_id = 1000010647 where title like '%allume cigare%' and category3_id is null;
update test set category3_id = 1000010635 where title like '%batterie externe portable%' and category3_id is null;
update test set category3_id = 1000010653 where (title like 'coque%' or lower(title) like 'bumper%' or lower(title) like 'facade%') and category3_id is null;
update test set category3_id = 1000004085 where title like 'batterie%' and category3_id is null;
update test set category3_id = 1000004079 where title like 'chargeur%' and category3_id is null;
update test set category3_id = 1000010667 where (title like 'etuit%pour%' or title like 'housse%') and category3_id is null;
update test set category3_id = 1000012993 where (title like '%sticker%cm' or title like '%sticker%cm' or title like '%sticker%- couleur%' or title like '%sticker%- taille%' or description like 'sticker%voir la presentation' or description like 'sticker%' or description like 'deco soon - sticker%' or description like 'planche sticker%' or description like '%sticker%lot de%' or title like '%sticker%- couleur%') and category3_id is null;
update test set category3_id = 1000010170 where ((title like 'montre homme%' or title like 'montre femme%') and description like '%montre%voir la presentation') and category3_id is null;
update test set category3_id = 1000010533 where description like '%shirt%voir la pres%' and category3_id is null;
update test set category3_id = 1000015309 where (description like 'de %' or description like '%- collectif') and brand = 'aucune' and category3_id is null;
update test set category3_id = 1000008094 where (title like 'dalle ecran%' or title like 'ecran dalle%') and category3_id is null;
update test set category3_id = 1000010108 where title like '%boucle%d%oreille%' and category3_id is null;
update test set category3_id = 1000010136 where title like 'pendentif%' and brand = 'adamence' and category3_id is null;

-- supprimer de training les lignes sur les catégories précédents => réduction du training de 6.6M (15 à 9) - 3'
delete from training where category3_id in (1000010647, 1000010635, 1000010653, 1000004085, 1000004079, 1000010667, 1000012993, 1000010170, 1000010533, 1000015309, 1000008094, 1000010108, 1000010136);

/*
-------------------------------
Suppression des marques du training qui ne sont pas présentes dans la table de test pour réduire le volume de training -- 5'
-- hypothèse : toute marque connue doit être affectée à une catégorie possédant déjà cette marque
-- => réduction de 4M de lignes (9 à 5.6)
-------------------------------
*/
delete FROM training WHERE NOT EXISTS (SELECT 1 FROM test WHERE test.brand = lower(training.brand));

/*
-------------------------------
Purge des doublons (title/description) dans le référentiel training pour réduire le volume de training -- 10'
-------------------------------
*/
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
-- check : select title || description, count(*) from training group by title || description having count(*)>1 order by 2 desc => 0 row

/*
-------------------------------
Traitement sur les catégories qui ne contiennent qu'une seule marque pour réduire le volume de training -- 4'
-------------------------------
*/
-- suppression table
drop table IF EXISTS brand_category3;
-- création table : identifier toutes les marques qui ne sont associées qu'à une seule catégorie (on connaît la marque => on connaît la catégorie)
CREATE TABLE brand_category3
(brand varchar, category3_id int, products_nb int);
-- insérer les couples brand - catégorie 3 dont la marque n'est utilisée que dans 1 catégorie -- 1'
insert into brand_category3 ( 
	select distinct lower(brand), category3_id from training 
	where lower(brand) <> 'aucune' and lower(brand) <> 'generique'
	and lower(brand) in (
		select lower(brand)
		from (select lower(brand), category3_id, count(*) from training group by lower(brand), category3_id) r
		group by brand
		having count(*) = 1
	) 
);

-- mettre à jour la catégorie 3 des produits de test dont la marque associée qu'à une seule catégorie -- 17''
update test set category3_id = (select category3_id from brand_category3 bc where lower(bc.brand) = test.brand) where category3_id is null;
-- supprimer de training les lignes sur les marques précédents => réduction du training de 900k (5.1 à 4.2)
delete from training where category3_id in (select category3_id from brand_category3); -- 2'

/*
-------------------------------
Traitement sur les catégories dont on calcule la marque la plus représentative 
-- hypothèse : les autres marques associés à ces catégories sont des erreurs d'affectation : - de 10 produits par marque dans chaque catégorie
-------------------------------
*/
-- vidage table
truncate brand_category3;

-- insérer les couples brand - catégorie 3 dont la marque est fortement corrélée à une catégorie 3
-- on part de l'hypothèse que dans une catégorie s'il n'y a qu'une seule marque dont le nb de produits est > 10 et que cette marque n'est présente dans aucune autre catégorie, alors on peut associer la marque à la catégorie
insert into brand_category3 ( 
	select lower(brand), category3_id, 0 
	from (
		select lower(brand), category3_id, count(*) 
		from training 
		where lower(brand) <> 'aucune' and lower(brand) <> 'generique'
		group by lower(brand), category3_id 
		having count(*)>10
	) r
	group by lower(brand), category3_id
	having count(*) = 1
);
delete from brand_category3 where brand not in (select brand from brand_category3 c3 group by brand having count(*)=1);

-- mettre à jour la catégorie 3 des produits de test dont la marque associée qu'à une seule catégorie
update test set category3_id = (select category3_id from brand_category3 bc where lower(bc.brand) = test.brand) where category3_id is null;
-- supprimer de training les lignes sur les marques précédents => réduction du training de 1M (4.2 à 3.2) - 2'
delete from training where category3_id in (select category3_id from brand_category3);

/*
-------------------------------
Importation TRAINING - partie 2
-- création d'index, nettoyage des libellés des champs titre, description et marque (minuscules, sans accent, sans stop word)
-------------------------------
*/
-- création d'index
--CREATE INDEX category1_training_idx ON training (category1_id);
--CREATE INDEX category2_training_idx ON training (category2_id);
CREATE INDEX category3_training_idx ON training (category3_id);
CREATE INDEX brand_training_idx ON training (brand);
--CREATE INDEX price_training_idx ON training (price);
CREATE INDEX title_training_idx ON training (title);
--CREATE UNIQUE INDEX product_id_training_idx ON training (product_id);

-- nettoyage du référentiel training
-- minuscule
update training set description = lower(description), title = lower(title), brand = lower(brand);
-- suppression des caractères accentués
update training set description = unaccent_string(description), title = unaccent_string(title), brand = unaccent_string(brand);
-- suppression de la marque dans le titre et la description du produit
update training set title = btrim(replace(title, brand, ''), ' '), description = btrim(replace(description, brand, ''), ' ') where brand <> '';
-- suppression des stop words
select replace_stopword_training();

/*
-------------------------------
Traitement sur le titre (10 premiers caractères du titre produit = 10 premiers caractères des produits appartenant à une seule catégorie)
-------------------------------
*/
-- suppression table
drop table IF EXISTS title10_category3;
-- création table : identifier toutes les marques qui ne sont associées qu'à une seule catégorie (on connaît la marque => on connaît la catégorie)
CREATE TABLE title10_category3
(title10 varchar, category3_id int, products_nb int);

insert into title10_category3 ( 
	select title10, category3_id, 0 
	from (
		select left(title, 10) as title10, category3_id, count(*) 
		from training
		group by left(title, 10), category3_id
		having count(*) > 10	
	) r
	group by title10, category3_id
	having count(*) = 1
);
delete from title10_category3 where title10 not in (select title10 from title10_category3 c3 group by title10 having count(*)=1);

-- mettre à jour la catégorie 3 des produits de test dont les 10 premiers carcatères du titre appartiennent à la table précédente
update test set category3_id = (select category3_id from title10_category3 bc where bc.title10 = left(test.title, 10)) where category3_id is null;

/*
-------------------------------
Merge sur title en calculant la catégorie 3 la plus significative
-------------------------------
*/
drop table IF EXISTS temp_title_categ3;
CREATE TABLE temp_title_categ3 (title varchar, category3_id int, nb int);
insert into temp_title_categ3 (
	select temp_title_categ3.* from (
		select test.title, training.category3_id, count(*) as nb from test
		inner join training on training.title = test.title 
		where test.category3_id is null
		group by test.title, training.category3_id
	) temp_title_categ3
	inner join (
		select r1.title, max(r1.cpt) as compteur from (
		select test.title, training.category3_id, count(*) as cpt from test
		inner join training on training.title = test.title 
		where test.category3_id is null
		group by test.title, training.category3_id
		) r1
		group by r1.title
	) r2 on r2.compteur = temp_title_categ3.nb and r2.title = temp_title_categ3.title
);

-- mettre à jour la catégorie 3 des produits de test dont le titre appartient à la table précédente
update test set category3_id = (select category3_id from temp_title_categ3 bc where bc.title = test.title limit 1) where category3_id is null;

/*
-------------------------------
Mise à jour massive sur les comparaisons de titre et description avec le référentiel des rayons
-------------------------------
*/
-- mettre à jour la catégorie 3 des produits de test dont le titre = nom de catégorie 3
update test set category3_id = (select category3_id from rayons r where r.category3_label = test.title limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont la description = nom de catégorie 3
update test set category3_id = (select category3_id from rayons r where r.category3_label = test.description limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont le début du titre et le début de la description se retrouvent dans le libellé de la catégorie 3 en restant sur la même marque - 5h
update test set category3_id = (
	select c.category3_id from (
	select te.product_id, r.category3_id, r.category3_label, count(*)
	from test te
	inner join rayons r on (r.category3_label like '%'||left(te.description, 5)||'%' or r.category3_label like '%'||left(te.title, 5)||'%')
	inner join training tr on tr.category3_id = r.category3_id
	where te.category3_id IS NULL and tr.brand = te.brand and te.product_id = test.product_id
	group by te.product_id, r.category3_id, r.category3_label
	order by 4 desc
	limit 1) c
) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont le titre est contenu dans le nom de catégorie 3
update test set category3_id = (select category3_id from rayons r where r.category3_label like '%'||test.title||'%' limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont la description est contenue dans le nom de catégorie 3
update test set category3_id = (select category3_id from rayons r where r.category3_label like '%'||test.description||'%' limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont les mots-clés contenus dans le titre et la description s'apparentent au libellé d'une catégorie 3 - 30'
update test set category3_id = (select category3_id from rayons r where ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.title||' '||test.description)) > 0.01 order by  ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.title||' '||test.description)) desc limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont les mots-clés contenus dans le titre s'apparentent au libellé d'une catégorie 3 - 30'
update test set category3_id = (select category3_id from rayons r where ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.title)) > 0.01 order by  ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.title)) desc limit 1) where test.category3_id is null;

-- mettre à jour la catégorie 3 des produits de test dont les mots-clés contenus dans la description s'apparentent au libellé d'une catégorie 3 - 30'
update test set category3_id = (select category3_id from rayons r where ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.description)) > 0.01 order by  ts_rank(to_tsvector('french', r.category3_label), plainto_tsquery('french', test.description)) desc limit 1) where test.category3_id is null;