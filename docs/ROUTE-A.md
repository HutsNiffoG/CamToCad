# Route A — lokaal en open source (v0.4.1)

Dit is de uitvoering van profiel A uit [OPEN-SOURCE-LOKAAL.md](OPEN-SOURCE-LOKAAL.md): foto's van een onderdeel op een geprinte kalibratiemat gaan naar je eigen pc, die er een CAD-model van maakt. Er is geen cloud nodig en er zijn geen licentiekosten; alle afhankelijkheden hebben een permissieve open-sourcelicentie.

| | |
|---|---|
| **Status** | v0.4.1 — werkt end-to-end op synthetische scans en is gehard tegen storingen uit echte foto's (schaduw, weinig of scheve bovenaanzichten, OpenCV 4.x/5.x). Nieuw in v0.4.1, na de eerste echte fotoset: een controle of het onderdeel tussendoor is verplaatst, en maskers die zich per foto aanpassen aan onscherpte, posefout en glans. In v0.4: zwarte en witte onderdelen op mat v2, afrondingen en overbodige hoekpunten in de fit. Sinds v0.3: mat v2, fotocontrole vooraf, printschaal in X en Y, en gereedschap om scans met schuifmaatmetingen te vergelijken ([Fase 0](FASE-0.md)) |
| **Objectklasse** | 2,5D-onderdelen die plat op de mat liggen: extrusie van een contour (rechte randen met scherpe of afgeronde hoeken, of een cirkel) met doorgaande gaten |
| **Uitvoer** | `model.step`, `model.stl`, parametrisch CadQuery-script `model.py`, meetrapport `report.html` / `report.json` |
| **Platform** | Windows, macOS en Linux met Python 3.10–3.12; geen GPU nodig |

## Installeren

```bash
git clone https://github.com/HutsNiffoG/CamToCad.git
cd CamToCad
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[server]"      # met ".[server,dev]" ook de tests
```

Controleer de installatie met de demo. Die rendert een synthetische scan van een beugel en verwerkt die, zonder camera:

```bash
camtocad demo --uit camtocad-demo
```

## Gebruiken

1. **Mat printen.** `camtocad mat --formaat A4` (of `A3`, `Letter`) schrijft `kalibratiemat_A4.pdf`.
   - Print op 100% (werkelijke grootte), op mat papier, en leg de mat vlak op een stijve ondergrond.
   - Meet beide meetlijnen van 100,0 mm na: **X** onder de mat, **Y** links. Wijken ze af, geef ze dan op, bijvoorbeeld `--meetlijn 99.6 99.8`, of in de velden op de telefoonpagina. De printschaal wordt dan per richting gecorrigeerd.
   - Dit is mat v2: in elk zwart vak een raster witte stippen, zodat ook een donker onderdeel op een zwart vak te zien is. Een eerder geprinte mat (v1) werkt nog en wordt automatisch herkend, net als het formaat.
2. **Foto's maken.** Leg het onderdeel plat in het midden van de mat, bij diffuus licht. Maak 30–60 foto's met de gewone camera-app van je telefoon (JPG):
   - **raak het onderdeel niet aan tot de laatste foto**: één scan is één vaste ligging. Wil je de mat draaien, draai dan mat en onderdeel samen. Een andere kant (omgedraaid, op zijn kant) is een aparte scan;
   - rondom, op twee à drie hoogtes (ongeveer 35°, 55° en 70° boven de mat), op zo'n 25–35 cm afstand met de hele mat in beeld;
   - **4–6 foto's recht van boven**: nodig voor de contour en om door gaten heen te kijken;
   - de mat steeds grotendeels in beeld, niet inzoomen, zelfde lens (geen groothoek/tele wisselen).
3. **Controleren en verwerken**, op één van twee manieren:
   - **Via de browser van je telefoon:** start `camtocad server` op de pc en open op je telefoon (zelfde wifi) de link die in de terminal verschijnt, inclusief `?token=…`. Kies of maak daar de foto's.
     - Elke foto wordt direct gecontroleerd: is de mat gevonden, is de foto scherp en goed belicht.
     - Een dekkingskaart toont in het rood welke richtingen nog ontbreken, met aanwijzingen zoals "nog 2 foto's recht boven het onderdeel".
     - Slechte foto's verwijder je met ×. Ontbrekende foto's voeg je toe, ook later aan een mislukte scan.
     - Daarna tik je op **Verwerken**. Rapport, STEP, STL en script staan daarna klaar om te downloaden, bij een fout met de debugbeelden erbij.
   - **Via de opdrachtregel:** kopieer de foto's naar een map. `camtocad controleer <map>` beoordeelt de set in een paar seconden. `camtocad scan <map>` verwerkt hem; de uitvoer komt in `<map>_cad/`.

Opties voor `scan`:

- `--mat`: `A4`, `A3`, `Letter`, `A4-v1` of `A3-v1`. Standaard wordt de mat herkend aan de markers.
- `--meetlijn X [Y]`: gemeten lengte van de 100 mm-lijnen.
- `--max-zijde 2000`: werkresolutie.
- `--snapdrempel 0.8`.
- `--inch`: snappen naar inchmaten.

Om te meten hoe nauwkeurig scans van je eigen onderdelen zijn, vergelijk je ze met schuifmaatmetingen: `camtocad valideer`, zie [FASE-0.md](FASE-0.md).

## Als het niet lukt

Elke scan schrijft diagnosebeelden naar `<uitvoer>/debug/`, ook als de verwerking halverwege stopt:

| Bestand | Wat je ziet |
|---|---|
| `masker_<foto>.jpg` | Het objectmasker over de foto: **oranje** = object, **blauw** = zekere mat, **paars** = object en mat zijn daar even donker of licht, dus de foto zegt daar niets (binnen het object opgevuld voor de startcontour; de fit negeert het). Alle gebruikte bovenaanzichten en een paar schuine foto's |
| `lokalisatie.png` | De grove visual hull van boven over de mat (lichter = hoger), met het zoekgebied |
| `bovenaanzicht.png` | Hoe vaak de bovenaanzichten "object" zeggen op de gekozen hoogte (geel = allemaal), met de startcontour in cyaan |
| `diagnose.json` | Per foto: kijkhoek, onscherpte, objectaandeel, ruis en mathoeken; de dekking rond het object; welke foto's bij welke ligging horen (`ligging`); de hoogtezoektocht; de terugvalopties die zijn geprobeerd |

Meest voorkomende meldingen:

- **"Geen objectcontour gevonden in de foto's recht van boven"**
  - Kijk in `masker_top*.jpg` of het onderdeel oranje is.
  - Is het grotendeels grijs, dan steekt het te weinig af tegen de mat. Paars is geen probleem: daar is het onderdeel even donker (of licht) als de mat, en dat wordt opgevangen. Grote paarse stukken ontstaan bij een zwart onderdeel op een mat v1: print dan mat v2.
  - Oplossing: diffuus licht, het onderdeel midden op de mat, en 4–6 foto's recht boven het onderdeel met de hele mat in beeld.
  - Achter de melding staat welke foto's er rond het object nog ontbreken.
- **"Het onderdeel ligt niet in alle foto's op dezelfde plek"**
  - De foto's spreken elkaar tegen: in de ene foto ligt het onderdeel ergens waar een andere foto gewoon mat ziet. Meestal is het tussendoor verschoven, gedraaid of op een andere kant gelegd. De melding noemt welke foto's bij elkaar horen.
  - Oplossing: maak de scan opnieuw zonder het onderdeel aan te raken, en scan elke ligging apart.
  - Passen maar een paar foto's niet (minder dan een kwart), dan gaat de verwerking door zonder die foto's. Het rapport noemt ze bij de waarschuwingen: "foto('s) niet gebruikt omdat ze niet bij de rest passen". Dat kan ook een hand of ander voorwerp in beeld zijn, of een mislukt masker.
  - Een klein duwtje (een paar millimeter) valt hier niet op. Dat zie je aan "silhouetten passen matig" in het rapport.
- **"gekozen mat A4, maar de foto's tonen mat A3"**: de mat op de foto's is gebruikt. Controleer of dat de mat is die je bedoelde.
- **"betrouwbaarheid: laag"** in het rapport. Het model is gemaakt, maar iets klopt niet; de reden staat erbij. Bijvoorbeeld een gat dat als niet-ronde uitsparing is herkend, of een model dat in sommige foto's slecht past. Controleer die maten.
- **Foto's "niet gebruikt (afwijkend formaat)"**: andere lens, zoom of bijgesneden. Staand opgeslagen foto's worden automatisch teruggedraaid.
- **"foto's onscherp (σ > 1,8 px)"**: bewogen of niet scherp gesteld. Maak ze opnieuw; `camtocad controleer` laat per foto de onscherpte zien.

## Hoe het werkt

| Stap | Module | Wat er gebeurt | Architectuur |
|---|---|---|---|
| Mat | `mat.py`, `pdf.py` | ChArUco-mat als vector-PDF op exacte schaal, met meetlijnen X en Y en een stippenraster in de zwarte vakken. Eigen marker-ID's per formaat: de mat wordt herkend | §4.5 |
| Fotocontrole | `preflight.py` | Per foto: mat, onscherpte (gemeten aan de matranden), belichting, kijkhoek. Per scan: waar ligt het object, dekking per richting en hoogte, aanwijzingen | — |
| Camera | `calib.py` | Mat herkennen, camera zelf kalibreren (Zhang), per foto een metrische pose: de mat is de tracker. De printschaal (X en Y) zit in de matgeometrie. De hoekverschuiving van de geïnstalleerde OpenCV-versie wordt gemeten en gecorrigeerd; staand opgeslagen foto's worden teruggedraaid | §4.5, OPEN-SOURCE-LOKAAL §2.1 |
| Maskers | `masks.py` | Voorspel per foto hoe de mat eruitziet; afwijkingen zijn object. "Zekere mat" alleen waar het lokale patroon de mat herhaalt (correlatie), zodat donkere en witte onderdelen niet wegvallen en schaduwen mat blijven. Randpixels volgens de 50%-regel. Waar object en mat even donker of licht zijn (zwart op zwart), is een pixel geen bewijs: binnen het object wordt zo'n stuk opgevuld, de fit negeert het. Per foto aangepast aan onscherpte (de voorspelling even vaag), posefout (gemeten aan de matranden) en glans (zwart lichter dan verwacht) | §5.3 [R3c] |
| Ligging | `placement.py` | Ligt het onderdeel in alle foto's op dezelfde plek? Per voxel (3 mm) hoeveel foto's hem op het object zien en hoeveel op de mat; per foto of de andere foto's zijn objectpixels steunen en of hij de gezamenlijke hull niet op de mat ziet. Foto's met verschillende liggingen vallen zo in groepen uiteen | — |
| Lokaliseren | `hull.py` | Grove visual hull (2 mm): waar ligt het object, en een bovengrens voor de hoogte | §5.3 [R3c] |
| Startmodel | `initial.py`, `profile.py` | Hoogte zoeken waarop de bovenaanzichten samenvallen, terugprojecteren → contour → randen, afrondingen, gaten; met terugvalopties en een uitgelegde fout | §6.3–6.5 |
| Model fitten | `silhouette.py`, `pipeline.py` | Analysis-by-synthesis: model-silhouet renderen in élke foto, pixelverschil minimaliseren (hoogte, randen, afrondingen, gaten). Daarna per hoek grote stappen in de afronding proberen, en hoekpunten weghalen waar het model zonder even goed past (een knik of een korte schuine rand zonder bewijs) | §6.8, §7.3 |
| Ontwerpintentie | `snapping.py`, `cadmodel.py` | Werkassenstelsel met datum, randen exact haaks, Bayesiaans snappen (hele mm, ISO 273, tapboormaten, standaardstralen), steekcirkels | §6.6 |
| CAD | `cadmodel.py`, `cadhelpers.py` | OpenCascade-solid via CadQuery, STEP/STL-export, leesbaar script met benoemde maten | §6.9 |
| Kwaliteit | `pipeline.py`, `debug.py` | Kwaliteitspoort (past het model bij de foto's?) en diagnosebeelden in `debug/` | §4.6 |
| Rapport | `report.py` | Maten met U95 en snapreden, betrouwbaarheid, bovenaanzicht, samenvatting, waarschuwingen; de geometrie staat ook in `report.json` | §4.6 |
| Validatie | `validate.py` | Scans vergelijken met schuifmaatmetingen (`maten.json`): fout per maat, binnen U95 ja/nee, bias per soort, printschaalcontrole | §8 |

**Waarom silhouetten en geen MVS?** Voor deze objectklasse zijn silhouetten nauwkeuriger en robuuster: ze hebben geen textuur op het object nodig en zijn ongevoelig voor glans. Een visual hull alléén is niet genoeg: van bovenaf is het bovenvlak niet te zien, en de wanden worden te ruim. De maten komen daarom uit het fitten van het parametrische model op de randen in alle foto's — de "model-gebaseerde verfijning" uit het architectuurdocument. MVS (COLMAP/OpenMVS) volgt voor vrije vormen.

## Resultaten op synthetische scans

Gerenderde scans van 46 foto's (1600 × 1200 pixels, ~0,25 mm/pixel op het object), met lensvervorming, ruis en belichtingsverloop. De camera en de poses worden uit de foto's zelf bepaald.

| Onderdeel | Maat | Waarheid | Gefit | Na snappen |
|---|---|---|---|---|
| Beugel (op 17° gedraaid) | lengte × breedte × hoogte | 80 × 40 × 12 | 79,95 × 39,95 × 12,05 | 80 × 40 × 12 |
| | afrondingen (4x) | R3 | R2,85–R3,17 | R3 |
| | gaten (2x) | Ø 6,6 op (10, 20) en (70, 20) | Ø 6,63 en Ø 6,61 op (9,93, 19,94) en (69,98, 20,00) | Ø 6,62 (niet gesnapt: 6,5 en 6,6 beide plausibel); posities exact |
| L-vorm (niet-convex) | maten | 60 / 30 / 25 / 50, hoogte 8 | 59,95 / 29,96 / 24,91 / 49,95, hoogte 8,07 | exact |
| | afrondingen | 4 × R2, 2 scherpe hoeken | 4 × R2,24, 2 × scherp | R2,24 (niet gesnapt, binnen U95) |
| | gat | Ø 5,5 op (15, 12) | Ø 5,43 op (14,98, 11,96) | Ø 5,43 (niet gesnapt: 5,3 en 5,5 beide plausibel) op (15, 12) |
| Ronde flens | diameter × hoogte | Ø 50 × 10 | Ø 49,96 × 10,02 | Ø 50 × 10 |
| | gatenpatroon | 4 × Ø 4,5 op steekcirkel Ø 35 | 4 × Ø 4,43 op Ø 35,01 | 4 × Ø 4,5 (ISO 273 M4) op Ø 35, parametrisch patroon |

Over alle stresstests die geen waarschuwing geven (ook een zwart en een wit onderdeel, zie [ROUTE-A-VERBETERPUNTEN.md §2b](ROUTE-A-VERBETERPUNTEN.md#2b-stresstests-v04)):

- buitenmaten binnen ±0,08 mm van de waarheid, hoogtes 0 tot +0,09 mm;
- gaten tot 0,07 mm te klein; bij het zwarte onderdeel één gat 0,2 mm;
- afrondingen −0,1 tot +0,4 mm.

Dat is binnen het Precisie-doel van ±(0,2 mm + 0,1% · L). **Let op:** dit zijn synthetische scans met mat v2. Echte foto's hebben schaduwen, autofocus, compressie en een niet perfect vlakke mat. Hoe dicht v0.4 daarbij in de buurt komt, moet Fase 0 uitwijzen ([FASE-0.md](FASE-0.md)).

De eerste echte fotoset (een zwarte accu op mat v1) is niet gelukt: het onderdeel lag in minstens vier verschillende liggingen, en een zwart onderdeel op mat v1 is op de zwarte vakken onzichtbaar. Wat dat opleverde staat in [ROUTE-A-VERBETERPUNTEN.md §2c](ROUTE-A-VERBETERPUNTEN.md#2c-de-eerste-echte-fotoset-september-2026).

## Beperkingen van v0.4.1

- **Objectklasse:** alleen 2,5D-onderdelen plat op de mat, met doorgaande gaten. Geen blinde gaten, kamers, treden in de hoogte, afschuiningen op de bovenrand, schroefdraad of vrije vormen. Afschuiningen op verticale hoeken worden als afronding benaderd; buitencontouren met bogen groter dan een hoekafronding (bijv. een sleufvorm) worden met rechte randen benaderd.
- **Bovenaanzichten zijn verplicht**: zonder foto's recht van boven stopt de verwerking met een duidelijke melding.
- **Eén ligging per scan.** Een verplaatst of omgedraaid onderdeel wordt herkend en gemeld; een klein duwtje van een paar millimeter niet (zie "Als het niet lukt").
- **Belichting:**
  - Schaduwen op de gestructureerde delen van de mat worden herkend.
  - Een harde slagschaduw over een egaal vak kan nog als object meetellen. De kwaliteitspoort markeert het resultaat dan als onbetrouwbaar (in de stresstests: altijd).
  - Gebruik diffuus licht, en mat papier voor de mat.
- **Zwarte en witte onderdelen:** op mat v2 lukken ze in de stresstests. Waar het onderdeel even donker (of licht) is als de mat, telt de foto niet mee; binnen het onderdeel wordt zo'n stuk opgevuld, en de fit gebruikt alleen echt bewijs. Het zwakste punt is een gat boven een groot zwart vlak van een marker: de rand is daar in de bovenaanzichten niet te zien, en in de stresstest kwam zo'n gat 0,2 mm te klein uit. Op een mat v1 (zonder stippen) mislukt een zwart onderdeel nog; het resultaat wordt dan gemarkeerd. Print mat v2.
- **Onzekerheid (U95)** is een indicatie op basis van resolutie en aantal foto's, nog niet gekalibreerd op echte metingen. Meet het zelf met `camtocad valideer` ([FASE-0.md](FASE-0.md)).
- **Afrondingen** zijn het minst nauwkeurig (−0,1 tot +0,4 mm): ze bepalen maar een klein stukje van het silhouet. Gelijke afrondingen worden gegroepeerd en alleen gesnapt als dat zeker is.
- **Afrondingen kleiner dan ~3 pixels** (bij de demo ~0,8 mm) zijn niet te onderscheiden van scherpe hoeken en worden als scherp gemodelleerd; het rapport meldt dat.
- **Rekentijd:** ~55 s per scan van 46 foto's (1600 × 1200) op een gewone CPU met 4 kernen, waarvan ~4 s voor de controle op verplaatsing; foto's worden standaard teruggeschaald naar 2000 pixels. Tot ~2,5 minuten als er overbodige hoekpunten uit de contour moeten (zwart onderdeel, schaduw). De fotocontrole kost ~0,1–0,3 s per foto.
- **Nog geen native app:** de telefoon gebruikt de browser. De geleide AR-opname uit het architectuurdocument volgt met de Android-app.

## Licenties

Alle gebruikte bibliotheken zijn open source met een permissieve licentie: NumPy en SciPy (BSD), OpenCV (Apache-2.0), CadQuery (Apache-2.0) op OpenCascade (LGPL-2.1 met uitzondering), en FastAPI/Uvicorn (MIT/BSD). Welke licentie Cam-to-CAD zelf krijgt, is nog een open beslissing; OPEN-SOURCE-LOKAAL.md §1 adviseert AGPL-3.0-or-later.

## Volgende stappen

Een uitgebreid, geprioriteerd overzicht staat in [ROUTE-A-VERBETERPUNTEN.md](ROUTE-A-VERBETERPUNTEN.md). Het is gebaseerd op stresstests, een codereview en onderzoek naar de stand van de techniek. Kort:

1. **Fase 0 met echte foto's:** 10–20 onderdelen met bekende maten scannen, afwijkingen en U95-kalibratie meten, drempels bijstellen (debugbeelden in `resultaat/debug/`).
2. **Gatranden en de hoogte uit beeldranden** (subpixel) als extra term in de fit — nauwkeuriger bij weinig bovenaanzichten.
3. **Uitbreiding van de objectklasse:** treden en kamers (meerdere hoogteniveaus), afschuiningen, sleufvormen, gatenpatronen op rechthoekige delen.
4. **Vrije vormen:** COLMAP + OpenMVS met de matposes, en `camtocad cad` voor puntenwolken.
5. **Android-app** (GPL, zonder Google Play Services) met geleide opname: dekkingskoepel, automatisch afdrukken, stills op volle resolutie.
6. **FreeCAD-export** (`.FCStd` met PartDesign-feature tree) naast het CadQuery-script.
