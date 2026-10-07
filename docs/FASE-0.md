# Fase 0 — meten met echte foto's

Alle drempels en de opgegeven onzekerheid (U95) van route A zijn tot nu toe afgesteld op gerenderde scans. Fase 0 meet hoe nauwkeurig de scans met échte foto's zijn, en of U95 eerlijk is: bij een eerlijke onzekerheid valt ongeveer 95% van de fouten binnen U95. Dit document beschrijft hoe je de meetset maakt en hoe je hem draait (verbeterpunt V1 in [ROUTE-A-VERBETERPUNTEN.md](ROUTE-A-VERBETERPUNTEN.md)).

Kort: per onderdeel een map met foto's en een `maten.json` met je schuifmaatmetingen, dan `camtocad valideer <map>`.

Sinds v0.13 kan alles ook in de browser van je telefoon (`camtocad server`):

1. fotograferen met de **livecamera**, die de weg wijst en zelf de foto's maakt (§3);
2. per scan de **schuifmaatmetingen invullen** (knop "maten"); de vergelijking met de scan verschijnt zodra die verwerkt is (§4, §5);
3. de **meetset** onderaan de pagina telt alle scans met metingen op (§6), en "Downloaden om te delen" geeft een zip zonder foto's (§7).

## 1. Wat je nodig hebt

- **De mat v2.** Maak hem met `camtocad mat --formaat A4` (of `A3`, `Letter`).
  - Print op 100% (werkelijke grootte), op mat papier, en leg hem vlak op een stijve plaat.
  - Meet beide meetlijnen van 100,0 mm na met een schuifmaat: **X** onder de mat en **Y** links. Noteer ze; ze komen in `maten.json`.
  - Een mat v1 (zonder stippenraster) wordt nog herkend, maar een zwart onderdeel lukt daarop niet. Op mat v2 wel.
- **Een digitale schuifmaat** (0,01 mm). Radiusmallen voor afrondingen zijn handig maar niet nodig.
- **10–20 onderdelen**, zo gevarieerd mogelijk:
  - materiaal: aluminium en staal (mat en blank), zwart, wit en gekleurd kunststof;
  - formaat: van ~20 mm (klein plaatje) tot 100–150 mm;
  - vorm: rechthoekig met afgeronde hoeken, een L- of U-vorm, een ring of flens;
  - dikte 3–20 mm, met doorgaande gaten van verschillende diameters.
  Alleen 2,5D-onderdelen: plat, met doorgaande gaten. Sinds v0.7 mag de bovenrand rondom afgeschuind of afgerond zijn, of mag er één rechte trede in zitten; neem er een paar van mee. Sinds v0.8 telt kleur mee: neem ook een gekleurd onderdeel dat even donker is als de zwarte vakken (donkerblauw geanodiseerd, donkergroen of donkerrood kunststof), en een zwart onderdeel met een gat boven een zwart vak. Sinds v0.9 worden verzonken gaten (90°, voor verzonken schroeven) herkend: neem een onderdeel met een paar verzinkingen mee, en meet de diameter van de verzinking aan het bovenvlak (in `maten.json` als `verzinkingen`). Sinds v0.10 ook kamerboringen (een cilindrische kamer voor een cilinderkopschroef): neem een onderdeel met een paar kamerboringen mee, en meet de diameter en de diepte van de kamer (`kamerboringen` en `kamerdieptes`). Sinds v0.10 schat de scan ook de toonkromme van de camera uit de mat zelf, sinds v0.11 ook de verscherping van de telefoon en een kromme die per stuk beeld verschilt (lokale toonbewerking, HDR); daarvoor hoef je niets te doen, maar gebruik geen filters of effecten. Sinds v0.11 ook blinde gaten (niet door het onderdeel heen): neem een onderdeel met een paar blinde gaten mee en meet hun diepte (`gatdieptes`; de diameter hoort bij `gaten`), en een gat met een kleine faas (0,5 mm) aan de bovenkant: die wordt een smalle verzinking (meet haar diameter aan het bovenvlak, als `verzinkingen`).
- **Een referentie met bekende maten**, bijvoorbeeld een eindmaat of een nauwkeurig gefreesd blokje, en een ring of ringkaliber. Daarmee zie je het verschil tussen een meetfout van de scan en een meetfout van de schuifmaat.
  - Een referentie die iedereen heeft: een **bankpas** (ISO/IEC 7810 ID-1: 85,60 × 53,98 mm, hoeken R3,18, 0,76 mm dik). Op de telefoonpagina vult de knop "Bankpas als referentie" die normmaten in.
  - Neem een pas zonder reliëf, liefst gekleurd of donker (meer contrast met het witte papier van de mat), en meet de dikte zelf na (reliëf maakt hem dikker).
  - In de stresstest (v0.14) komen een grijze en een witte pas in lengte en breedte binnen 0,01 mm uit, en in dikte binnen 0,02 mm. In v0.13 kwam een witte pas 0,06–0,08 mm te kort uit: de rand meet het grijs van het object vlak binnen de rand, en bij een pas is dat het bovenvlak, niet de dunne zijkant. Die randpunten tellen sinds v0.14 niet meer mee.

## 2. Meten met de schuifmaat

- Meet elke maat drie keer, op verschillende plaatsen, en noteer het gemiddelde. Laat het onderdeel eerst op kamertemperatuur komen (niet direct na het vasthouden).
- **Lengte en breedte:** de grootste en kleinste buitenmaat langs de hoofdrichtingen van het onderdeel.
- **Hoogte:** de dikte.
- **Gaten:** de diameter met de binnenbekken, op twee plaatsen 90° verdraaid.
- **Hartafstand** tussen twee gaten: meet de binnenmaat tussen de gaten en tel de halve diameters erbij op (of de buitenmaat over beide gaten min de halve diameters).
- **Diameter** van een rond onderdeel: de buitendiameter.
- **Afrondingen:** alleen als je een radiusmal hebt; anders weglaten.
- **Afschuining of afronding van de bovenrand** (`afschuining` in `maten.json`): bij een afschuining het been (hoogteverschil tussen bovenkant en onderkant van de afschuining, met de diepte-uitsteker), bij een afronding de straal met een radiusmal.
- **Trede** (`treden`): de hoogte van het lage deel, met de diepte-uitsteker of de schuifmaat vanaf de onderkant.
- **Verzinking** (`verzinkingen`): de diameter van de kegel aan het bovenvlak, met de binnenbekken van de schuifmaat plat op het bovenvlak, of met een verzinkingsmeter.
- **Kamerboring** (`kamerboringen`, `kamerdieptes`): de diameter van de kamer met de binnenbekken, en de diepte (bovenvlak tot de bodem van de kamer) met de diepte-uitsteker.
- **Blind gat** (`gaten`, `gatdieptes`): de diameter met de binnenbekken (bij de gaten), en de diepte tot de vlakke bodem met de diepte-uitsteker. Een geboord gat met een kegelvormige punt: meet tot waar de wand ophoudt (de scan ziet de onderrand van de wand).

## 3. Fotograferen

- Leg het onderdeel plat in het midden van de mat, bij diffuus licht (bewolkt daglicht, of een lamp tegen het plafond), met minstens 2 cm mat eromheen en niets anders op de mat. Raak het niet aan tot de laatste foto.
- **Drie extra scans om v0.12 op echte foto's te toetsen** (elk een aparte map, met dezelfde `maten.json`):
  - *duwtje*: schuif het onderdeel halverwege de foto's bewust ~1 mm opzij (tik er met een potlood tegen) en fotografeer verder. Het rapport hoort te melden "het onderdeel is tijdens het fotograferen verschoven", met de maten binnen hun U95;
  - *slagschaduw*: één scan met een lamp of zon schuin op de mat, zodat het onderdeel een scherpe schaduw werpt. Het rapport hoort "slagschaduw naast het onderdeel" te melden, en de maten horen binnen hun (ruimere) U95 te vallen;
  - *liniaal*: een liniaal of munt naast het onderdeel op de mat. Het rapport hoort die te melden en alleen het onderdeel te verwerken.
- Maak 30–60 foto's met dezelfde telefoon, zonder zoom:
  - 4–6 recht van boven, met de hele mat in beeld;
  - rondom op ongeveer 35° en 60° boven de mat, om de ~45°;
  - in kleur, zonder filter: kleur is bewijs voor het object (v0.8). JPG en HEIC (iPhone) werken allebei; voor HEIC is de extra `heic` nodig (`pip install -e ".[server]"` heeft hem al);
  - met één lens: een iPhone schakelt dichtbij vanzelf naar de macrolens (ultragroothoek). Blijf op 25–35 cm, of zet Macrobesturing aan en de macrostand uit. Foto's van een andere lens of met digitale zoom worden aan de EXIF-gegevens herkend en niet gebruikt; de fotocontrole meldt het.
- **Of gebruik de livecamera** (v0.13), dan hoef je de richtingen niet te onthouden:
  - Open op de telefoon de https-link uit de terminal (of scan de QR-code). De camera werkt in de browser alleen via https; de eerste keer waarschuwt de telefoon voor het certificaat van je pc: kies "doorgaan".
  - Tik op **Live camera**. Je ziet de omtrek van de mat over het beeld, een kruisje op het onderdeel en één aanwijzing, zoals "Volgende foto: loop ~45° naar rechts om het onderdeel" of "houd de telefoon lager, ~35° boven de mat".
  - De volgorde: eerst 5 foto's recht van boven, daarna rondom op ~60° en ~35°, 2 foto's per richting (8 richtingen).
  - Staat de telefoon goed en stil, dan maakt de pagina zelf de foto (vinkje "automatisch"; de rode knop kan altijd). De minikaart onderaan is gedraaid zodat jij onderaan staat: groen is genoeg, rood ontbreekt, wit omrand is de volgende.
  - Gebruik binnen één scan óf de livecamera óf de camera-app, niet allebei: hun foto's hebben een ander formaat en een andere beeldhoek. Op een iPhone zijn de foto's van de livecamera beelden uit de video (Safari kent geen ImageCapture), dus wat kleiner dan een gewone foto.
- **Controleer de set direct**, vóór je het onderdeel weghaalt:
  - Via de telefoonpagina (`camtocad server`) gebeurt dat vanzelf. Elke foto krijgt een oordeel (goed, matig of onbruikbaar), en een dekkingskaart toont in het rood welke richtingen nog ontbreken.
  - Via de opdrachtregel draai je `camtocad controleer <map-met-fotos>`. Dat kost ongeveer 0,1 s per foto.
  - Maak de foto's bij die de controle vraagt, bijvoorbeeld "nog 2 foto's recht boven het onderdeel".

## 4. Mappen en `maten.json`

```
validatie/
  beugel-alu/
    maten.json
    fotos/IMG_0001.jpg ...     (de foto's mogen ook direct in de map staan)
  plaatje-zwart/
    maten.json
    fotos/...
```

Een `maten.json`:

```json
{
  "naam": "beugel aluminium",
  "meetlijn": [100.05, 99.95],
  "maten": {
    "lengte": 80.02,
    "breedte": 40.01,
    "hoogte": 12.03,
    "gaten": [6.62, 6.60],
    "hartafstanden": [60.01],
    "afrondingen": [3.0, 3.0, 3.0, 3.0]
  }
}
```

| Veld | Betekenis |
|---|---|
| `meetlijn` | De gemeten meetlijnen X en Y van de geprinte mat, in mm. Eén getal geldt voor beide. Laat je hem weg, dan wordt 100,0 aangenomen, met de onzekerheid van een ongemeten print (0,3%) in de U95 |
| `mat` | Optioneel: `A4`, `A3`, `Letter`, `A4-v1` of `A3-v1`. Standaard wordt de mat herkend aan de markers |
| `lengte`, `breedte`, `hoogte`, `diameter` | Eén waarde. De hoogte is de totale hoogte, ook bij een afschuining of trede |
| `afschuining` | Eén waarde: het been van een afschuining, of de straal van een afronding, van de bovenrand rondom |
| `gaten`, `hartafstanden`, `afrondingen`, `treden`, `verzinkingen`, `kamerboringen`, `kamerdieptes`, `gatdieptes` | Een lijst. Elke waarde wordt gekoppeld aan de dichtstbijzijnde maat van het model; `treden` zijn de hoogtes van de lage delen, `verzinkingen` de diameters van verzinkingen aan het bovenvlak (v0.9; ook een kleine faas aan een gat, v0.11), `kamerboringen` en `kamerdieptes` de diameter en de diepte van de kamer van een kamerboring (v0.10), `gatdieptes` de diepte van blinde gaten (v0.11; hun diameter staat bij `gaten`) |

**In de browser** (v0.13): kies bij een scan **maten** en vul de metingen in, in mm. Meerdere waarden scheid je met een spatie of puntkomma (`6,62 6,60`); lege velden tellen niet mee. Dat schrijft dezelfde `maten.json` in de map van de scan, met de meetlijnen en de mat van de scan erin. De datamap van de server is daarmee ook een validatiemap: `camtocad valideer ~/camtocad-data` werkt er direct op.

## 5. Draaien

```bash
camtocad valideer validatie/
```

Per onderdeel wordt de scan verwerkt naar `resultaat/`. Een eerder resultaat wordt hergebruikt, tenzij er nieuwe foto's zijn of je `--opnieuw` meegeeft (bijvoorbeeld na een update van camtocad). Daarna volgen per maat:

- de referentie en de gefitte (ongesnapte) waarde;
- de fout: model min referentie;
- de U95 die de scan zelf opgeeft, en of de fout daarbinnen valt;
- de waarde na het snappen.

De samenvatting staat in de terminal en in `validatie/validatie.html` en `validatie.json`.

Voor een regressietest na een wijziging: `camtocad valideer validatie/ --opnieuw --eis-dekking 0.9` geeft een foutcode als minder dan 90% van de fouten binnen U95 valt.

**In de browser** (v0.13) staat de vergelijking onder de metingen van een scan, zodra die verwerkt is. Er wordt daarvoor niets opnieuw verwerkt: zijn er na de verwerking foto's bijgekomen, dan staat er "verouderd" tot je opnieuw verwerkt. De **meetset** onderaan de pagina telt alle verwerkte scans met metingen op, zoals `validatie.html`.

Ook de demo levert zo'n map op: `camtocad demo --uit demo` en daarna `camtocad valideer demo`.

## 6. Lezen

| Wat je ziet | Betekenis |
|---|---|
| ~95% binnen U95 | De onzekerheid is eerlijk |
| Duidelijk minder dan 90% | U95 is te optimistisch; de fouten zijn groter dan de scan beweert |
| Alles ruim binnen U95 | U95 is te ruim; de scan is beter dan hij zegt |
| Bias per soort, bijvoorbeeld "gat: systematisch −0,06 mm" | Een systematische fout, te corrigeren in een volgende versie |
| "U95 extra" per soort, bijvoorbeeld 0,04 mm bij gaten (v0.13) | Zoveel onzekerheid ontbreekt (kwadratisch bij de U95 opgeteld) om 95% van de fouten binnen U95 te krijgen; 0 betekent dat de U95 klopt. De basis om de systematiek in U95 bij te stellen |
| "Lengtes wijken gemiddeld +0,3% af" | Vrijwel altijd de printschaal: meet de meetlijnen na en zet ze in `maten.json` |
| "gat ontbreekt in het model" of extra gaten | Een topologiefout. Kijk in `resultaat/debug/` (maskers, bovenaanzicht) |
| Scan met `fout` | De foutmelding staat erbij; zie [ROUTE-A.md — Als het niet lukt](ROUTE-A.md#als-het-niet-lukt) |

## 7. Resultaten delen

Wil je helpen de drempels en U95 af te stellen, deel dan:

- `validatie.json`;
- per onderdeel `resultaat/report.json` en `resultaat/debug/diagnose.json`.

Die bevatten geen foto's. De foto's zelf zijn alleen nodig als een scan mislukt en de debugbeelden niet genoeg zeggen.

Op de telefoonpagina (v0.13) geeft **Downloaden om te delen** precies dat als zip: `validatie.json` en `validatie.html` van de hele set, en per scan `maten.json`, `resultaat/report.json`, `resultaat/debug/diagnose.json` en de status (mat, meetlijnen, aantal foto's, meldingen). Geen foto's, en geen paden van je pc.
