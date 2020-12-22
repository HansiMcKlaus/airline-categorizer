#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on the day of dawn

@author: HansiMcKlaus
"""

import numpy as np
import glob
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rescale
from scipy import ndimage
import time
from skimage.filters import gaussian                                            #Muss importiert werden, auch wenn's rummeckert...
from skimage.feature import match_template                                      #Muss importiert werden, auch wenn's rummeckert...
import os

startTime = time.time()                                                         #Startzeit
limiter = 10 #Begrenzt, wieviele Bilder geladen werden
convolution = 0 #Grad an Faltung für Gradienten, 0 für keine

def currentTime():                                                              #Gibt die vergangene Zeit an
    endTime = time.time()
    timeDiff = endTime - startTime
    return str(timeDiff)

planesPath = glob.glob('./Ordner/*.jpg')                                        #Einlesen der Charaktere aus Ordner mit Bildern
planesLabel = []
for x in planesPath[:limiter]:
    planesLabel.append(x.split("/")[-1].split(".")[0].split("_")[0])
planes = []
for x in planesPath[:limiter]:
    planes.append(imread(x))
    
leitwerkMask = imread('./leitwerkMaske.png')                                    #Einlesen der Leitwerk Maske
leitwerkMaskMirrored = np.fliplr(leitwerkMask)                                  #Spiegeln der Leitwerk Maske
leitwerkMasks = []                                                              #Einfügen der Leitwerk Masken
leitwerkMasks.append(rescale(leitwerkMask, 0.8, order=0))
leitwerkMasks.append(rescale(leitwerkMask, 0.9, order=0))
leitwerkMasks.append(leitwerkMask)
leitwerkMasks.append(rescale(leitwerkMask, 1.1, order=0))
leitwerkMasks.append(rescale(leitwerkMask, 1.2, order=0))
leitwerkMasks.append(rescale(leitwerkMaskMirrored, 0.8, order=0))
leitwerkMasks.append(rescale(leitwerkMaskMirrored, 0.9, order=0))
leitwerkMasks.append(leitwerkMaskMirrored)
leitwerkMasks.append(rescale(leitwerkMaskMirrored, 1.1, order=0))
leitwerkMasks.append(rescale(leitwerkMaskMirrored, 1.2, order=0))

print ('Bilder und Masken eingelesen bei: ' + currentTime())


planesR = []
planesG = []
planesB = []
for plane in planes:                                                            #Bilder in ihre Kanäle zersetzen
    planesR.append(plane[:,:,0])
    planesG.append(plane[:,:,1])
    planesB.append(plane[:,:,2])

planesRGradient = []
for red in planesR:                                                             #Gradienten für Rot-Kanal
    red = skimage.filters.gaussian(red, convolution)
    redSobelHor = ndimage.sobel(red, axis = 0)
    redSobelVer = ndimage.sobel(red, axis = 1)
    redGradient = np.hypot(redSobelHor, redSobelVer)
    planesRGradient.append(redGradient)
    
planesGGradient = []
for green in planesG:                                                           #Gradienten für Grün-Kanal
    green = skimage.filters.gaussian(green, convolution)
    greenSobelHor = ndimage.sobel(green, axis = 0)
    greenSobelVer = ndimage.sobel(green, axis = 1)
    greenGradient = np.hypot(greenSobelHor, greenSobelVer)
    planesGGradient.append(greenGradient)
    
planesBGradient = []
for blue in planesB:                                                            #Gradienten für Blau-Kanal
    blue = skimage.filters.gaussian(blue, convolution)
    blueSobelHor = ndimage.sobel(blue, axis = 0)
    blueSobelVer = ndimage.sobel(blue, axis = 1)
    blueGradient = np.hypot(blueSobelHor, blueSobelVer)
    planesBGradient.append(blueGradient)
    
print ('Alle Gradienten (Farbkanäle) erstellt bei: ' + currentTime())


planesGradient = []                                                             #Gradient aus Kanal-Gradienten
for i in range(len(planes)):
    plane = planes[i]                                                           #Setzen des Flugzeuges als Bild
    redGradient = planesRGradient[i]                                            #Setzen der Kanal-Gradienten
    greenGradient = planesGGradient[i]
    blueGradient = planesBGradient[i]
    planeGradient = np.maximum(redGradient, np.maximum(greenGradient, blueGradient))    #Größten Wert der Farb-Gradienten
    planesGradient.append(planeGradient)                                        #Hinzufügen des Gradienten
    #print('Gradient Nr. ' + str(i) + ' erstellt bei: ' + currentTime())

print ('Alle Gradienten erstellt bei: ' + currentTime())


planesTemplates = []                                                            #Alle Templates aller Flugzeuge
planesBestTemplate = []                                                         #Bestes Template aller Flugzeuge
leitwerkePosition = []                                                          #Positionen der Leitwerke und Maske bestimmen   
counter = 0
for planeGradient in planesGradient:
    planeTemplates = []
    bestValues = []
    for leitwerkMask in leitwerkMasks:
        planeTemplated = skimage.feature.match_template(planeGradient, leitwerkMask)    #Erstellen des Templates, ohne pad_input=True
        planeTemplates.append(planeTemplated)
        bestValues.append(np.max(planeTemplated))                                       #Höchster Wert einer Maske bestimmen
    planesTemplates.append(planeTemplates)
    x, y = np.unravel_index(np.argmax(planeTemplates[np.argmax(bestValues)]), planeTemplates[np.argmax(bestValues)].shape)  #Koordinaten des höchsten Wertes aller Masken
    planesBestTemplate.append(planeTemplates[np.argmax(bestValues)])
    leitwerkePosition.append([[x, y], leitwerkMasks[np.argmax(bestValues)]])
    print('Position von Leitwerk Nr. ' + str(counter) + ' ermittelt bei: ' + currentTime())
    counter += 1

print ('Alle Positionen der Leitwerke ermittelt bei: ' + currentTime())


planesBinary = []                                                               #Erstellt Maske über Leitwerk
for j in range(len(planes)):
    position = leitwerkePosition[j][0]
    mask = leitwerkePosition[j][1]
    darknessBlackerThanBlackAndDarkerThanDark = np.zeros((planes[j].shape[0], planes[j].shape[1]))      #0-Bild mit Größe des Ursprungsbilds
    darknessBlackerThanBlackAndDarkerThanDark[position[0]:(position[0] + mask.shape[0]), position[1]:(position[1] + mask.shape[1])] = mask  #Maske wird eingesetzt
    planeBinary = darknessBlackerThanBlackAndDarkerThanDark
    planesBinary.append(planeBinary)

print ('Alle Binärbilder der Leitwerke erstellt bei: ' + currentTime())


leitwerkeMean = []                                                              #Mittelwerte der Leitwerke
for k in range(len(planesBinary)):
    maskInverted = np.invert(planesBinary[k].astype(bool))                      #Invertieren der Maske
    planesRMasked = np.ma.array(planesR[k], mask=maskInverted)                  #Maskierte Farb-Kanäle erstellen
    planesGMasked = np.ma.array(planesG[k], mask=maskInverted)
    planesBMasked = np.ma.array(planesB[k], mask=maskInverted)
    leitwerkeMean.append([np.mean(planesRMasked), np.mean(planesGMasked), np.mean(planesBMasked)])

print ('Alle Mittelwerte der Leitwerke erstellt bei: ' + currentTime())


leitwerkeMeanAirlines = []                                                      #Mittelwerte der einzelnen Airlines
leitwerkeMeanAirlines.append([[145, 58, 47], 'hk'])                             #hk, Hong Kong Airlines
leitwerkeMeanAirlines.append([[77, 63, 72], 'lh'])                              #lh, Lufthansa
leitwerkeMeanAirlines.append([[109, 60, 131], 'thai'])                          #thai, Thai Airways
leitwerkeMeanAirlines.append([[164, 140, 159], 'aa'])                           #aa, American Airlines
leitwerkeMeanAirlines.append([[102, 138, 176], 'csa'])                          #csa, China Southern Airlines
leitwerkeMeanAirlines.append([[100, 113, 141], 'ua'])                           #ua, United Airlines
leitwerkeMeanAirlines.append([[219, 142, 111], 'ej'])                           #ej, easyJet

vorhersage = []                                                                 #Kategorisierung anhand der Mittelwerte (Nearest Neighbor)
for mean in leitwerkeMean:
    differences = []                                                            #Berechnung euklidischer Distanzen
    for z in range(len(leitwerkeMeanAirlines)):
        difference = [mean[0] - leitwerkeMeanAirlines[z][0][0], mean[1] - leitwerkeMeanAirlines[z][0][1], mean[2] - leitwerkeMeanAirlines[z][0][2]]
        difference = [difference[0]**2, difference[1]**2, difference[2]**2]
        difference = np.sum(difference)
        difference = difference**(0.5)
        differences.append(difference)
    vorhersage.append(leitwerkeMeanAirlines[np.argmin(differences)][1])

print ('Alle Leitwerke klassifiziert bei: ' + currentTime())


print("Vergangene Zeit (komplett): " + currentTime())


def plotLeitwerkePosition():                                                    #Mittelpunkt aller Leitwerke
    for l in range(len(leitwerkePosition)):
        pos = leitwerkePosition[l][0]
        mask = leitwerkePosition[l][1]
        xpos = pos[1] + (mask.shape[1]/2)
        ypos = pos[0] + (mask.shape[0]/2)
        print('Der Mittelpunkt des Leitwerks von Flugzeug Nr. ' + str(l) + ' liegt bei: x =' + str(xpos) + ', y = ' + str(ypos))


def plotLeitwerkeMean():                                                        #Gibt Mittelwerte aller Leitwerke aus
    for m in range(len(leitwerkeMean)):
        print('Der Mittelwert des Leitwerks von Flugzeug Nr. ' + str(m) + ' beträgt: ' + str(leitwerkeMean[m]))


def savePlaneAndBinary(plane):                                                  #(Nr. des Flugzeugs), speichert das Ursprungsbild und das Binärbild
    if(os.path.isdir('./flugzeugeUndMaske') == 0):                              #Erstellt, falls nicht vorhanden, neuen Ordner
        os.mkdir('flugzeugeUndMaske')
    imsave('./flugzeugeUndMaske/plane_' + str(plane) + '.jpg', planes[plane])
    imsave('./flugzeugeUndMaske/plane_' + str(plane) + '_binary.jpg', planesBinary[plane])


def saveAllPlaneAndBinary():
    for plane in range(len(planes)):
        savePlaneAndBinary(plane)


def plot(plane, type):                                                          #(Nr. des Flugzeugs, Art des Plots)
    if(type == 'plane'):                                                        #Zeigt das Flugzeug an
        plt.imshow(planes[plane])
    elif(type == 'label'):                                                      #Schreibt das Label des Flugzeugs aus
        print('Flugzeug Nr. ' + str(plane) + ' hat das Label: ' + planesLabel[plane])
    elif(type == 'planeR'):                                                     #Zeigt den Rot-Kanal des Flugzeugs an
        plt.imshow(planesR[plane], cmap="Greys_r")
    elif(type == 'planeG'):                                                     #Zeigt den Grün-Kanal des Flugzeugs an
        plt.imshow(planesG[plane], cmap="Greys_r")
    elif(type == 'planeB'):                                                     #Zeigt den Blau-Kanal des Flugzeugs an
        plt.imshow(planesB[plane], cmap="Greys_r")
    elif(type == 'planeRGradient'):                                             #Zeigt den Gradienten des Rot-Kanals an
        plt.imshow(planesRGradient[plane], cmap="Greys_r")
    elif(type == 'planeGGradient'):                                             #Zeigt den Gradienten des Grün-Kanals an
        plt.imshow(planesGGradient[plane], cmap="Greys_r")
    elif(type == 'planeBGradient'):                                             #Zeigt den Gradienten des Blau-Kanals an
        plt.imshow(planesBGradient[plane], cmap="Greys_r")
    elif(type == 'gradient'):                                                   #Zeigt den Gradienten des Flugzeugs an
        plt.imshow(planesGradient[plane], cmap="Greys_r")
    elif(type == 'template'):                                                   #Zeigt das Template des Flugzeugs am
        plt.imshow(planesBestTemplate[plane], cmap="Greys_r")
    elif(type == 'mask'):                                                       #Zeigt die Maske des Leitwerks an
        plt.imshow(leitwerkePosition[plane][1], cmap="Greys_r")
    elif(type == 'positionul'):                                                 #Gibt die Position des Leitwerks (Oben links) an
        print('Leitwerk (Obere linke Ecke) von Flugzeug Nr. ' + str(plane) + ' liegt bei: x =' + str(leitwerkePosition[plane][0][1]) + ', y = ' + str(leitwerkePosition[plane][0][0]))
    elif(type == 'positionm'):                                                  #Gibt die Position des Leitwerks (Mitte) an
        print('Leitwerk (Mittelpunkt) von Flugzeug Nr. ' + str(plane) + ' liegt bei: x =' + str(leitwerkePosition[plane][0][1] + (leitwerkePosition[plane][1].shape[1]/2)) + ', y = ' + str(leitwerkePosition[plane][0][0] + (leitwerkePosition[plane][1].shape[1]/2)))
    elif(type == 'binary'):                                                     #Zeigt das Binärbild des Leitwerks an
        plt.imshow(planesBinary[plane], cmap="Greys_r")
    elif(type == 'leitwerkMean'):                                               #Gibt den Mittelwert des Leitwerks an
        print('Mittelwert des Leitwerks von Flugzeug Nr. ' + str(plane) + ' beträgt:' + str(leitwerkeMean[plane]))