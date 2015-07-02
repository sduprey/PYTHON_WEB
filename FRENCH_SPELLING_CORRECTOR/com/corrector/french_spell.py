#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 29 janv. 2015

@author: stefan.duprey
'''
import re, collections

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

#training_path = 'D:\\My_Data\\My_Spelling_Corrector_Data\\titres_produits.csv'
training_path = '/home/sduprey/My_Data/My_Spelling_Corrector_Data/titres_produits.csv'
NWORDS = train(words(file(training_path).read()))


alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in s if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
   inserts    = [a + c + b     for a, b in s for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

################ Testing code from here on ################

def spelltest(tests, bias=None, verbose=False):
    import time
    n, bad, unknown, start = 0, 0, 0, time.clock()
    if bias:
        for target in tests: NWORDS[target] += bias
    for target,wrongs in tests.items():
        for wrong in wrongs.split():
            n += 1
            w = correct(wrong)
            if w!=target:
                bad += 1
                unknown += (target not in NWORDS)
                if verbose:
                    print 'correct(%r) => %r (%d); expected %r (%d)' % (
                        wrong, w, NWORDS[w], target, NWORDS[target])
    return dict(bad=bad, n=n, bias=bias, pct=int(100. - 100.*bad/n), 
                unknown=unknown, secs=int(time.clock()-start) )



french_test = {'tutoriel':'tutorial','manger':'menger',
               'accueil' : 'acceuil','inclue' : 'incluse','accompte' : 'acompte',
               'acoustique' : 'accoustique', 'abattre' : 'abbattre','aberration':'abberration',
               'addiction' : 'adiction', 'adresse' : 'addresse','apparaître':'aparaître apparaitre',
               'ascenseur' : 'acenseur', 'baïonnette' : 'bayonnette', 'bayer' : 'bailler',
               'bancaire' : 'banquaire', 'bissextile' : 'bisextile', 'bizarre' : 'bizard',
               'blanchiment':'blanchiement', 'blizzard' : 'blizard','carrousel':'carroussel',
               'catéchisme' : 'cathéchisme', 'cueillir' : 'ceuillir','châtain' : 'chatain',
               'cherté':'chèreté','coalition' : 'coallition','concurrence':'concurence',
               'conjugaison':'conjuguaison','controverse':'contreverse','courir':'courrir',
               'cigogne' : 'cygogne', 'déconnexion' : 'déconnection','dédicace' : 'dédicasse',
               'desservir' : 'déservir','dilemme' : 'dilemne', 'diluvien' : 'dilluvien',
               'emménager' : 'eménager', 'enivré':'ennivré', 'ermite':'hermite','exsangue':'exangue',
               'exaucer':'exhaucer', 'exorbitant' : 'exhorbitant','exciter':'exiter',
               'ferroviaire' : 'ferrovière', 'filigrane' : 'filligrane','gaufre' : 'gauffre',
               'genèse':'gennèse','groseillier':'groseiller','guyane':'guyanne',
               'hasard' : 'hazard', 'imbécilité' : 'imbécillité', 'inclus':'inclu',
               'inclura' : 'incluera', 'ingrat' : 'ingras','installation' : 'instalation',
               'japonais':'japonnais','journaux' : 'journeaux','littéralement':'littérallement',
               'magazine' : 'magasine', 'maligne' : 'maline','mnémotechnique':'mémotechnique',
               'obédience' : 'obédiance', 'obnubiler':'obnibubiler','oculaire' : 'occulaire',
               'occurence' : 'occurrence', 'ornithorynque': 'ornythorinque', 'pécuniaire':'pécunier',
               'permanence':'permannence','pertinemment' : 'pertinament','philharmonique':'phillharmonnique',
               'pilule':'pillule','planification':'plannification', 'pourrir' : 'pourir','puissamment':'puissament', 
               'quatuor':'quattuor','résonance' : 'résonnance','rébellion' : 'rebellion','recueillir':'receuillir',
               'recèle' : 'recelle', 'recommandation' : 'recommendation', 'récompense' : 'récompence', 'réédition':'rédition',
               'résorption' : 'résorbtion', 'résous' :'résouds','résout':'résoud', 'rôle' : 'role',
               'rhétorique' : 'rhétoric','séance' : 'scéance', 'soubresaut':'soubressaut','spatial' : 'spacial',
               'succédané' : 'succédanné', 'synchrone' : 'sinchrone',  'syphilis' : 'Syphillis', 'théâtre' : 'théatre',
               'souffrir' : 'soufrir', 'thème' : 'tème', 'tonnerre' : 'tonnère', 'toponymie':'toponomie',
               'trafic' : 'traffic', 'tranquillité':'tranquillitée','vaillamment' :'vaillemment',
               'voirie' : 'voirie','voierie' : 'voierie','vraisemblable':'vraissemblable','waggon':'wagon'}  
                
if __name__ == '__main__':
    filename = '/home/sduprey/My_Data/My_Spelling_Corrector_Data/dictionnary.csv'
    print 'Writing file to disk : '+filename
    f1=open(filename, 'w+')
    f1.write('name; value\n') 
    #NWORDS_sorted=sorted(NWORDS)
    for keys,values in sorted(NWORDS.items()):
        f1.write(keys);f1.write(';');f1.write(str(values));f1.write('\n');
    f1.close()
    print spelltest(french_test)

