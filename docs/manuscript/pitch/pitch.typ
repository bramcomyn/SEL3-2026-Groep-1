#import "../configuration.typ": *

#set document(
  title: "Software Engineering Lab 3: Pitch",
  author: "Bram Comyn"
)

#show: configuration

#align(center)[#title()]

Wij hebben ons het afgelopen semester ondergedompeld in de wereld van damage robustness en Multi-Agent Reinforcement Learning (MARL).
Het hoofddoel van ons onderzoek was om te bekijken of we robots schadebestendig kunnen maken aan de hand van Independent Q-Learning (IQL).
We stellen een eenvoudig Proof of Concept (PoC) voor als resultaat, waaruit blijkt dat deze aanpak kan werken in eenvoudige simulaties.

Geïnspireerd door de natuur gingen wij aan de slag met een robotmodel gebaseerd op slangsterren: aanverwanten van zeesterren met de unieke capaciteit om aanpassingen te doen aan hun voortbewegingsmanier als reactie op beschadiging of zelfs verlies van ledematen.

We gingen te werk met 5 eenvoudige bewegingen, geïmplementeerd door wijze van Central Pattern Generators (CPGs): wiskundige modellen om rhythmische, bijna reflexmatige bewegingen voor te stellen die niet afhankelijk zijn van externe sensorische input.
Onze robot bestaat uit een aantal armen, in dit geval toevallig ook vijf, die elk één van deze bewegingen uitkiezen en zo leren samenwerken om naar een doel in hun omgeving te bewegen.

De samenwerking tussen de armen optimaliseren we aan de hand van IQL: een MARL-uitbreiding op standaard Q-Learning.
Meer specifiek maken we gebruik van een MARL-variant van Deep Q-Learning, waarbij we de Q-functie proberen benaderen aan de hand van een neuraal netwerk.

In onze experimenten zijn we erin geslaagd om de robot schadebestendig te maken t.a.v. een eenvoudig schademodel waarin we de actuator-inputs voor één arm op een willekeurig punt in wegmaskeren door nullen -- hierdoor blijft de arm stijf vooruit gericht.
Onze robot leerde hiermee omgaan door deze arm als wijzer of staart te gebruiken en de beweging te coördineren met de vier overige armen.

Verder onderzoek kan zich richten op het uitbreiden van deze resultaten naar meer complexe scenario's, met ingewikkeldere schademodellen -- zoals random noise toevoegen aan actuator-inputs, rechtstreekse controle over de actuators i.p.v. CPGs en meer gespecialiseerde MARL-algoritmen, zoals policy gradient-based algorithms.

Meer weten over ons onderzoek? Dan verwijzen wij je graag door naar onze documentatie op Github!
