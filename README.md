# uoc_tfm
Treball Final de Màster (TFM).
d’Albert Baranguer i Codina
Màster Universitari d’Enginyeria Informàtica
Curs 2021-22 – 1

Títol del TFM
«Entrenament d’una xarxa neuronal amb el dataset HAM10000 per al diagnòstic de
lesions de la pell».

Paraules clau
Keras, PyTorch, Aprenentatge automàtic (Machine Learning), Aprenentatge profund
(Deep Learning), Xarxa Neuronal (Neural Network), Convolutional Neural network (CNN),
Classificació d’imatges (Image classification).

Temàtica escollida
La temàtica escollida és la classificació d’imatges (Image Classification) aplicada al
diagnòstic de lesions de la pell, com càncers de pell.
Per a la classificació es farà servir una xarxa neuronal que serà entrenada amb el dataset
HAM10000 mitjançant tècniques d’Aprenentatge Automàtic (Machine Learning, ML) i
d’Aprenentatge Profund (Deep Learning, DL).

Problemàtica a resoldre
Hi han diverses tècniques que es poden aplicar per al diagnòstic del càncer de pell.
A més de les tècniques de cirurgia menor (biopsia) es poden aplicar tècniques de
diagnòstic per la imatge.
Es tracta, doncs, d’un problema de diagnòstic per la imatge, és dir, de classificació
d’imatges. És un problema escaient per a ser resolt amb tècniques de ML/DL.

Objectius
El plantejament inicial que faig és:
• La revisió dels conceptes teòrics necessaris (papers, llibres, cursos...)
• Proves de concepte amb les llibreries Keras i PyTorch, amb llenguatge Python.
Sobre Jupyter Notebook (Google Colab) per a major agilitat en les proves.
• Desenvolupament d’una CNN per a la classificació d’imatges de lesions de la pell.
• Entrenament de la CNN amb el dataset HAM10000.
• Iteració i refinament de la CNN mitjançant l’aplicació de tècniques de ML i DL, fins
aconseguir uns objectius (per determinar) de certs paràmetres (com l’accuracy).

Què «entra», per tant?
• el desenvolupament de la CNN,
• l’entrenament de la CNN
Per aconseguir-ho caldrà establir uns criteris que permetin quantificar la qualitat del
classificador. Per assolir aquesta qualitat, o apropar-s’hi tant com sigui possible, caldrà
aplicar tècniques de ML i DL. És un objectiu del TFM el coneixement i la utilització
d’aquestes tècniques.

Què «no entra»?
El més important és desenvolupar una comprensió general dels conceptes de ML/DL
aplicada a la classificació d’imatges. I també el coneixement i aplicació de tècniques de
ML/DL per a la millora de la CNN classificadora.
No és objectiu prioritari del TFM assolir una determinada precisió del classificador. Tot i
que, certament, es tractarà d’obtenir els millors resultats.

# Setmana 0. Reunió 1 d’octubre de 2021
pasos a seguir:
1) lectura de papers 3-4 -- deadline 8 de Octubre (reunion 16:00-17:00)
====== decidir en la reunion del dia 8 octubre =========
2) decidir que backbone (uno sencillito resnet18).
    hardware google colab
    timm
    pipeline de training, torchvision trainin script
    seguir los pasos de best practices: overfitear un batch durante 100 iteraciones.
3) definir la funcion de coste

pagina para buscar papers: http://www.arxiv-sanity.com/
twit best practices: https://twitter.com/MattNiessner/status/1441027241870118913


# Setmana 1. Reunió 8 d’octubre de 2021
Implementacion
* use squeezenet | resnet18 | mobilenet-v3 | efficientnet (light weight backbone)
* dataset <---
    * 1000 x 1000
    * rescalar las imagenes a resolucion (con interpolacion bilinear, mantener el aspect ratio) de
        1) 64 x 64
        2) 96 x 96
        3) 128 x 128
        4) 192 x 192
        5) 224 x 224
* dataloader
        * [implementar el dataloader sea async con GPU] (https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py#L57)
            -- leer datos -- transformar datos -- forward pass -- loss -- backward -- optimizer
            -- IO               --  cpu              -- GPU  ----- GPU ----- GPU ------ GPU
* loss [cross entropy]
* learning scheduler [opcional]
* optimizer [SGD, Adam, AdamW]
* implementar la parte de evaluacion
* --- tensor board (visualizar los losses, accuracies) [optional] ---


Preguntas para la siguiente semana:
* distribucion de categorias
* cual es la métrica adecuada para evaluar un modelo en datasets de imagenes medicas
* como dividir el dataset
* reportar los resultados de experimentos realizados
* crear un documento csv (google drive) de los resultados (experimentos) interesantes para la comparacion en el futuro.


