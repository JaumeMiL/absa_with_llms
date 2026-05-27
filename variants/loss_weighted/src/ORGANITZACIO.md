holis

a part de tot el tema de configs i tal
ara mateix el codi em sembla bastant desorganitzat / es pot millorar un huevo / stats es una merda / hi ha les mateixes funcions a tot arreu

a més, necessitariem tambe com a fitxers nous

nous
----

augmentation.py -> fitxer per generar les dades noves sintetiques

config.py -> per llegir / carregar els configs nous

data / paths / prompts -> mes o menys lo que fa ara common pero igual separat / millor no?

experiments -> o sweeps o matrix o com li vulguis dir pero per pdoer executar la combinatoria 

summary/plots -> no tenim fitxers bons per llegir outputs / fer taules / fer plots, si els outputs nous estan ben estructurats es podria fer relativament fàcil i seria tope útil


refer / canviar / canviar nom
-----

stats.py -> per mi el refem la vd o si mes no posar-lo mes bonic

generation -> canviar zeroshot.py a aixo o inference o com li vulguis dir però al final es un fitxer per fer inferencia i ya. No te sentit tenir fitxers separats per inferencia de few shot, per inferencia de zero shot, per inferencia del finetune... rollo es tot inferencia del models (el codi d'abans ja ho feia, pe`ro dirli zeroshot em sembla bastant criminal quan realment despres el fas servir amb fewshot i el que sigui)


el finetune-train que serveixi tb per quantization i tot (rollo si carregues el config hauria d'anar)

fewshot ->generation esta tot junt, llavors fewshot podria ser un fitxer que simplement seleccioni els exemples i ya. No entenc quin sentit te la tonteria que fa el fewshot actual de fer les prediccions del model també?? fewshot pillem el gold i ya no?? no entenc i no crec que faci falta

-------------
coses extres a canviar:
- a part de carregar els configs i tal, que els logs vagin a carpetes indicades i els outputs també, tot organitzat i clar
- en general millorar el codi es un pel cutre ara :D