# Route A — verbeterpunten (onderzoek, september 2026)

Dit document beschrijft wat er aan de implementatie van route A ([ROUTE-A.md](ROUTE-A.md)) beter kan. Een deel van de bevindingen zit al in v0.2 (§3), het gereedschap voor Fase 0 in v0.3 (§3b), zwarte en witte onderdelen plus een betere fit in v0.4 (§3c), en wat de eerste echte fotoset leerde in v0.4.1 (§2c, §3d); de rest staat hier als geprioriteerde lijst (§4).

Het onderzoek bestond uit drie delen:

1. **Eigen stresstests.** Synthetische scans met storingen zoals in echte foto's, door de volledige pipeline heen: schaduw, JPEG, vignettering, een scheve of bol liggende mat, een handschaduw, donkere en witte onderdelen, verspreide of weinig bovenaanzichten, kleine onderdelen, en OpenCV 4.x naast 5.0.
2. **Een codereview** van alle modules. Elke bewering is waar mogelijk met een klein experiment nagebootst.
3. **Een literatuur- en softwareonderzoek** naar de stand van de techniek in 2025–2026. De nadruk lag op open source dat lokaal draait, met een licentie die past bij AGPL-3.0.

Aanleiding was de melding "Geen objectcontour gevonden" van de eerste gebruiker met echte foto's.

## 1. Kernbevindingen

- **De oorzaak van "Geen objectcontour gevonden" was waarschijnlijk de starthoogte.**
  - v0.1 projecteerde de bovenaanzichten terug op 0,6 × de hoogte van de grove visual hull.
  - Bij een scan met weinig schuine foto's is die hull vaak vele malen te hoog: 33–65 mm voor een beugel van 12 mm, 113 mm voor een ring van 8 mm.
  - Op die verkeerde hoogte vallen de bovenaanzichten niet meer over elkaar, en de contour wordt leeg of onzin.
  - *Opgelost in v0.2* (§3).
- **Een tweede, sluipende oorzaak zat in de kalibratie.**
  - Keurde de tweede uitschieterronde een foto af, dan kregen alle volgende foto's de pose van hun buurfoto.
  - Twee onscherpe of verkeerd herkende foto's zijn genoeg; daarna zijn maskers en hull onzin.
  - *Opgelost.*
- **De maskers vertrouwden te veel op grijswaarden.**
  - Een donker object op een zwart vak lijkt per pixel op de mat. v0.1 noemde tot 41 % (donker) en 11 % (wit) van de objectpixels "zekere mat". Die pixels sneden het object weg uit de hull.
  - Schaduwen telden als object.
  - v0.2 kijkt naar de lokale textuur (correlatie met de voorspelde mat) en corrigeert de lokale belichting.
- **Het resultaat hing af van de OpenCV-versie.**
  - OpenCV vóór 4.14 legt ChArUco-hoeken systematisch 0,5 px verschoven ([opencv#25539](https://github.com/opencv/opencv/issues/25539), opgelost in 4.14/5.x). De voorspelde mat viel dan een halve pixel naast de foto; met OpenCV 4.12 werd de beugel 12,31 mm hoog in plaats van 12,0.
  - De eigen rastermat had nog een tweede halve-pixelfout.
  - v0.2 meet de verschuiving bij de start en corrigeert die, voor elke versie.
- **Stille fouten waren het grootste risico.**
  - In de stresstests (§2) ging v0.1 in 10 van de 13 situaties onderuit. Drie keer was dat een crash. Zeven keer kwam er zonder waarschuwing een fout model uit: verkeerde maten (onder OpenCV 4.12 0,3 mm te hoog), spookgaten, ontbrekende gaten.
  - v0.2 heeft een kwaliteitspoort: twijfelgevallen krijgen "betrouwbaarheid: laag" met de reden, onzin wordt een foutmelding met uitleg.
- **De eerste echte fotoset mislukte om een andere reden: het onderdeel lag niet stil** (§2c).
  - Het lag in minstens vier liggingen: plat, omgedraaid, op zijn kop en op zijn kant. Een visual hull en een silhouetfit gaan uit van één vaste ligging.
  - Daarbij: een zwart onderdeel op mat v1, glans, harde schaduwen en veel onscherpe foto's.
  - *v0.4.1* herkent een verplaatst onderdeel en zegt welke foto's bij elkaar horen; de maskers passen zich per foto aan onscherpte, posefout en glans aan (§3d).
- **Grootste open punten:**
  - nauwkeurigheid en een eerlijke U95. De fit telt pixels, en afrondingen komen niet beter dan ±0,3 mm.
  - zwarte en witte onderdelen. Sinds v0.4 lukken ze op mat v2 in de stresstests (§2b); echte foto's moeten dat bevestigen. Een harde slagschaduw wordt gemarkeerd, maar niet opgelost: diffuus licht blijft nodig.
  - de objectklasse: treden, verzinkingen, afschuiningen.
  - begeleiding tijdens het fotograferen. Sinds v0.3 volgt na elke foto een controle met aanwijzingen; live in de camera nog niet (V25).

## 2. Stresstests: v0.1, v0.2 en v0.3

Proefstukken:

- **Beugel:** 80 × 40 × 12 mm, R3-hoeken, 2 gaten Ø 6,6.
- **Plaatje:** 30 × 20 × 5 mm, R2-hoeken, gat Ø 4,5.
- **Ring:** Ø 25 × 8 mm, gat Ø 8.

Ruis, JPEG-compressie, vignettering en verscherping zitten in alle scenario's behalve "basis". Maten zijn de gefitte waarden vóór het snappen. **Stil fout** betekent: een fout model zonder waarschuwing. v0.1 en v0.2 draaiden met mat v1, v0.3 met mat v2.

| Scenario | Situatie | v0.1 | v0.2 | v0.3 (mat v2) |
|---|---|---|---|---|
| Basis | Beugel, 46 foto's, schone render | Goed: 79,96 × 39,95 × 12,02 | Goed: 79,86 × 39,93 × 12,07 | Goed: 80,01 × 39,95 × 12,04, 2 × Ø 6,59 |
| Realistisch | Ruis, JPEG, vignettering, verscherping | Goed: 79,92 × 39,89 × 12,02 | Goed: 79,94 × 39,91 × 12,03 | Goed: 79,98 × 39,94 × 12,03, 2 × Ø 6,59 |
| Harde schaduw | Slagschaduw van 55 % | **Stil fout**: IoU 0,78, gaten op de verkeerde plek | 79,92 × 39,96 × 12,00, maar één gat als uitsparing: **gemarkeerd als onbetrouwbaar** | 80,03 × 39,97 × 12,00, 2 × Ø 6,53, maar een inhammetje bij één hoek (korte randen): **gemarkeerd als onbetrouwbaar** |
| Donker onderdeel | Albedo 0,08 (zwart kunststof) | **Crash** (deling door nul) | **Duidelijke foutmelding**: het silhouet is maar half zichtbaar (V5) | Rommelig model (50 randen): **gemarkeerd als onbetrouwbaar**, na ~3 minuten. Mat v2 maakt het onderdeel grotendeels zichtbaar, maar in de stipvrije stroken blijven strepen "onbekend" (V8) |
| Wit onderdeel | Albedo 0,95 | **Stil fout**: 12 spookgaten, IoU 0,78 | 79,93 × 39,88 × 11,99, maar twee spookgaatjes (Ø 1,2) en een uitsparing: **gemarkeerd als onbetrouwbaar**¹ | 79,91 × 39,90 × 12,13 met de twee echte gaten (Ø 6,46), maar 12 spookgaatjes (Ø 1,5–3): **gemarkeerd als onbetrouwbaar** (gaten waardoor geen mat te zien is) |
| Handschaduw | Schaduw van hand en telefoon in de bovenaanzichten | **Stil fout** in de codereview: 87,5 × 64,1 × 10,8 | Goed: 79,92 × 39,91 × 12,02 | Goed: 79,98 × 39,92 × 12,03, 2 × Ø 6,60² |
| Verspreide bovenaanzichten | Tot 45 mm naast het onderdeel, 10° scheef | Goed, maar één gat Ø 6,46 | Goed: 79,89 × 39,92 × 12,05 | Goed: 79,97 × 39,97 × 12,01, 2 × Ø 6,58² |
| Weinig lage foto's | Alleen ringen op 60° en 75°, 5 bovenaanzichten | **Stil fout**: h 12,49, 79,29 × 39,62, geen gaten | 79,93 × 39,92 × 12,01, maar één gat als uitsparing: **gemarkeerd als onbetrouwbaar**¹ | Goed: 79,95 × 39,96 × 12,02, 2 × Ø 6,55² |
| Klein onderdeel | Plaatje 30 × 20 × 5, R2, Ø 4,5 | **Crash** (singuliere matrix) | Goed: 29,92 × 19,93 × 5,01, Ø 4,48 | Goed: 29,90 × 19,92 × 5,07, Ø 4,47² |
| Klein, weinig lage foto's | Idem, bovenaanzichten 6° scheef | **Crash** | Goed: 29,84 × 19,88 × 5,12, Ø 4,55. Afrondingen onzeker | Goed: 29,93 × 19,94 × 5,10, Ø 4,43² |
| Ring | Ø 25 × 8, gat Ø 8, weinig lage foto's | **Stil fout**: spookobject 56 × 154 mm, IoU 0,52 | Goed: Ø 24,88 × 8,08, gat Ø 7,99 | Goed: Ø 24,88 × 8,08, gat Ø 8,00² |
| Zwaar | Schaduw, bolle mat, handschaduw en verspreide bovenaanzichten samen | **Stil fout**: h 11,89, geen gaten, IoU 0,93 | Goed: 79,91 × 39,94 × 12,05, 2 × Ø 6,58 (één extra korte rand) | 79,95 × 39,90 × 12,07, maar een knik van 3,8° in de bovenrand en één gat Ø 6,38: **gemarkeerd als onbetrouwbaar**³ |
| OpenCV 4.12 | Basisscan met de oudere OpenCV | **Stil fout**: h 12,31; 79,80 × 39,82 | 12,02; 79,92 × 39,93 (hoekverschuiving gemeten en gecorrigeerd) | Goed: 79,91 × 39,96 × 12,07 (hoekverschuiving +0,50 px gemeten en gecorrigeerd) |

¹ Gedraaid vóór de laatste reparatie van de gatherkenning. Die reparatie lostte hetzelfde probleem op in "realistisch" (daar nu beide gaten Ø 6,59); deze twee scenario's zijn daarna niet opnieuw gedraaid.

² Gedraaid met de definitieve maskers van v0.3, maar vóór de laatste aanpassingen aan de fit (eerder stoppen, gladgestreken startcontour). Bij "basis" en "realistisch", die opnieuw gedraaid zijn, veranderden lengtes, hoogte en gaten daardoor minder dan 0,02 mm en de afrondingen tot 0,2 mm.

³ De markering komt van de controles op een knik in een rand en een te grote afronding. Die zijn na deze run toegevoegd en nagerekend op het bewaarde model; bij de goede scans in deze tabel slaan ze niet aan.

Wat opvalt in v0.3 (mat v2):

- **Goede scans worden nauwkeuriger.** Buitenmaten en hoogtes liggen binnen ±0,08 mm (v0.2: lengtes ~0,1 mm te klein). De maskerrand heeft geen bias meer die met het matpatroon meeverandert (§3b). Gaten komen 0,01–0,1 mm te klein uit en afrondingen tot 0,3 mm te groot: dat moet Fase 0 op echte foto's bevestigen of corrigeren (V13).
- **"Weinig lage foto's" is nu goed** (v0.2: een gat werd een uitsparing).
- **Harde schaduw en "zwaar" worden als onbetrouwbaar gemarkeerd.** De hoofdmaten kloppen, maar er zitten kleine fouten langs de schaduwkant, en in "zwaar" is één gat 0,2 mm te klein. Diffuus licht blijft nodig.
- **Spookgaten worden herkend.** Een gat waardoor in geen enkel bovenaanzicht zekere mat te zien is, maakt het resultaat onbetrouwbaar. Het witte onderdeel had er 12; de twee echte gaten tonen de stippen van de mat en slaan niet aan.
- **Een zwart onderdeel lukt nog niet.** Met mat v2 is het grotendeels zichtbaar, maar in de stipvrije stroken langs de vakranden en rond de hoeken blijft het "onbekend". De startcontour wordt dan rommelig en het resultaat wordt terecht als onbetrouwbaar gemarkeerd. De volgende stap staat in §4.2 (V8).
- **Rekentijd.** Faalt een scan, dan kost dat geen tientallen minuten meer: de fit stopt bij stilstand of bij een matige pasvorm, en een rommelige startcontour krijgt maar een korte verfijning (in v0.2 kostte het zwarte onderdeel 12 minuten).

Wat opvalt in v0.2:

- **Systematische afwijking.** Lengtes en diameters komen systematisch ongeveer 0,1 mm te klein uit (0,05 mm per rand), hoogtes ongeveer 0,05 mm te groot. Dat is binnen het precisiedoel en zit nu in de U95.
- **Oorzaak.** De maskerrand ligt ~0,09 px naar binnen; de rest komt uit de fit. Oplossen via V2 en V13.

## 2b. Stresstests v0.4

Dezelfde scenario's met mat v2 en de code van v0.4, plus een zwart en een wit onderdeel op mat v1. **Goed** betekent: geen waarschuwing. De maten zijn gefit, vóór het snappen.

| Scenario | v0.4 | v0.3 |
|---|---|---|
| Basis | Goed: 79,94 × 39,94 × 12,08, R3,03–3,11, Ø 6,59 en 6,58 | Goed |
| Realistisch | Goed: 79,97 × 39,94 × 12,01, R3,04–3,30, Ø 6,56 en 6,62 | Goed |
| Harde schaduw | **Gemarkeerd als onbetrouwbaar**: uitsteeksels aan de schaduwkant (korte randen, een knik, een te grote afronding) | Gemarkeerd |
| Donker onderdeel | **Goed**: 79,96 × 39,96 × 12,01, R3,02–3,28, Ø 6,49 en 6,39 | Rommelig model, gemarkeerd |
| Wit onderdeel | **Goed**: 80,00 × 39,99 × 12,03, R3,00–3,41, Ø 6,56 en 6,54 | 12 spookgaatjes, gemarkeerd |
| Handschaduw | Goed: 80,01 × 39,93 × 12,01, Ø 6,62 en 6,62 | Goed |
| Verspreide bovenaanzichten | Goed: 79,94 × 39,96 × 12,06, Ø 6,60 en 6,59 | Goed |
| Weinig lage foto's | Goed: 79,93 × 39,94 × 12,04, Ø 6,59 en 6,60 | Goed |
| Klein onderdeel | Goed: 29,92 × 19,94 × 5,06, R1,94–2,17, Ø 4,48 | Goed |
| Klein, weinig lage foto's | Goed: 29,95 × 19,94 × 5,04, R2,08–2,21, Ø 4,43 | Goed |
| Ring | Goed: Ø 24,88 × 8,09, gat Ø 7,99 | Goed |
| Zwaar | **Gemarkeerd als onbetrouwbaar**: een schaduwbult (twee korte randen); Ø 6,37 en 6,54 | Gemarkeerd |
| OpenCV 4.12 | Goed: 79,95 × 39,97 × 12,07, Ø 6,61 en 6,61 | Goed |
| Donker, mat v1 | **Gemarkeerd als onbetrouwbaar**: rommelig model (zonder stippen is het onderdeel op de zwarte vakken niet te zien) | — |
| Wit, mat v1 | Goed: 79,97 × 39,95 × 12,07, Ø 6,58 en 6,55 | — |

Wat opvalt in v0.4:

- **Zwart en wit lukken op mat v2.** Het zwarte onderdeel was in v0.3 een rommelig model met 50 randen, het witte had 12 spookgaatjes.
- **Geen stille fouten.** Elk scenario is goed of gemarkeerd. Tijdens de ontwikkeling gaf de harde schaduw nog een fout model zonder waarschuwing: schaduwvlekjes die bij het object werden getrokken. De toets per gebied en een sluiting van alleen het object zelf (§3c) hebben dat verholpen.
- **Afrondingen:** de afrondingsproef vindt nu alle vier de hoeken (R3 gefit tussen 2,98 en 3,41). Daarvoor kwamen soms één of twee hoeken scherp uit.
- **Gaten** komen tot 0,07 mm te klein uit (V13). Bij het zwarte onderdeel komt één gat 0,2 mm te klein uit: het ligt boven een groot zwart vlak van een marker (V5).
- **Rekentijd:** 21–148 s per scan (twee scans tegelijk op 4 kernen). Het weghalen van overbodige hoekpunten kost tot ~80 s, alleen als er kandidaten zijn.

**v0.4.1** (dezelfde scenario's, met de maskers voor echte foto's en de controle op verplaatsing, §3d): hetzelfde beeld.

- De controle op verplaatsing hield in geen enkel scenario foto's apart; harde schaduw en zwaar blijven gemarkeerd.
- Goed: buitenmaten 79,92–79,97 × 39,92–39,98, hoogtes 12,00–12,08, afrondingen R2,96–3,20, gaten Ø 6,53–6,60; klein onderdeel 29,92–29,95 × 19,93–19,97 met Ø 4,43–4,47; ring Ø 24,89 met gat Ø 8,00.
- Het donkere onderdeel is beter: gaten Ø 6,53 en 6,53 (v0.4: 6,49 en 6,39).
- Zwaar: één gat Ø 6,23 (v0.4: 6,37), gemarkeerd. De maskers in dat gat zijn gelijk aan v0.4; het verschil zit in de fit van dit instabiele scenario.

## 2c. De eerste echte fotoset (september 2026)

57 foto's van een zwarte accu op een geprinte A4-mat v1, met een telefoon (2992 × 2992 pixels). De verwerking stopte met "Geen objectcontour gevonden". Wat de analyse liet zien, van groot naar klein:

- **Het onderdeel lag niet stil.** De set bevat minstens vier liggingen: plat met het etiket boven, omgedraaid, op zijn kop (je ziet de contacten) en op zijn kant. Ook binnen één reeks is het verplaatst: in de twee foto's recht van boven van de tweede reeks ligt het de ene keer plat en de andere keer op zijn kop, met de contacten boven. Een visual hull en een silhouetfit gaan uit van één vaste ligging. Met silhouetten van verschillende liggingen blijft er van de hull een grillig restje over, en vallen de bovenaanzichten nergens samen.
- **Een zwart onderdeel op mat v1.** Op de zwarte vakken (zonder stippen) is het onderdeel onzichtbaar. Zekere mat is er alleen langs de vakranden, dus er wordt weinig weggesneden: de grove hull reikte tot 121 mm hoogte.
- **Glans en harde schaduwen.** Een lamp maakte de zwarte vakken lichter (glans op de toner), en schaduwen van hand en telefoon vielen over de mat.
- **Onscherpte en afstand.** 25 van de 53 bruikbare foto's hebben σ > 1,8 px (5 zelfs > 3 px). De camera hing vaak maar 9–15 cm boven de mat (mediaan 14 cm): grote parallax, weinig scherptediepte en een deel van de mat buiten beeld. De kalibratie haalt daardoor 1,6 px reprojectiefout; op synthetische scans is dat 0,1 px.
- **Weinig bovenaanzichten per ligging.** Er zijn er 17, maar verdeeld over de liggingen; per ligging twee à drie.

Wat v0.4.1 ermee doet (§3d):

- De nieuwe controle op verplaatsing meldt het direct, met de groepen foto's die bij elkaar horen. Op de hele set (53 bruikbare foto's): vier groepen, de grootste met maar een kwart van de foto's. Ook de deelreeksen worden gestopt:
  - Reeks 2b (14 foto's): de accu is halverwege omgedraaid, eerst met de zwarte kant boven en daarna met het etiket boven. Precies die helft past niet bij de rest.
  - Reeks 2a (11 foto's, plat en op zijn kop): de controle stopt ook hier (maar 8 van de 11 passen bij elkaar). De maskers zijn te slecht om de liggingen netjes te scheiden: die 8 horen niet allemaal bij dezelfde ligging. De groepen in de melding zijn bij zulke maskers dus een aanwijzing, geen exacte indeling.
  - Reeks 1a en 1b (16 en 12 foto's) komen door de controle, zonder 3 foto's die er niet bij passen (in 1b drie opeenvolgende: 202809–202810). Daarna lopen ze vast op "Het gevonden model past niet bij de foto's" (silhouet-IoU mediaan 0,63 en 0,51): een zwart onderdeel op mat v1, en in 1a een reprojectiefout van 3 px.
- De maskers houden meer bewijs over. In v0.4 was op deze foto's gemiddeld een kwart van de mat "dubbelzinnig" (paars; in sommige foto's driekwart): door glans en ruis leek bijna alles even donker als het onderdeel. Nu is dat 3%, en wordt meer van het onderdeel gevonden (reeks 2b: 2,0% van de mat als object tegen 1,0%). Er komen wel wat meer losse vlekken buiten het onderdeel bij: 0,8% van de mat tegen 0,4%, vooral in glans en harde schaduw.
- Een model van de accu levert deze set niet op. Daarvoor is een nieuwe scan nodig: mat v2, één ligging per scan, diffuus licht, telefoon stil en op 25–35 cm.

## 3. Opgelost in v0.2

| Probleem | Oplossing | Waar |
|---|---|---|
| Startcontour leeg bij een te hoge hull ("Geen objectcontour gevonden") | Hoogtezoektocht: de hoogte waarop de teruggeprojecteerde bovenaanzichten samenvallen. Alleen foto's waarin een punt op de mat valt stemmen mee, met een adaptieve stemdrempel. Terugval: lagere drempel → schuinere bovenaanzichten → hele mat → visual hull, daarna een uitgelegde fout | `initial.py` |
| Bovenaanzichten gekozen op de stand van de camera | Kijkhoek naar het object zelf: recht omlaag maar 60 mm ernaast is ~10° schuin | `initial.py` |
| Poses bij de buurfoto na de tweede uitschieterronde | Opnieuw kalibreren tot er geen uitschieters meer zijn; stoppen zolang poses en foto's bij elkaar horen | `calib.py` |
| 0,5 px verschuiving in OpenCV < 4.14; halve-pixelfout in de rastermat | Verschuiving gemeten op een synthetisch matbeeld en afgetrokken; rastermat volgt de OpenCV-pixelconventie | `calib.py`, `mat.py` |
| "Zekere mat" op donkere en witte objecten; schaduw als object | Zekere mat alleen als het venster het matpatroon herhaalt. Lokale versterking uit die vensters voor schaduw en glans. "Textuur verwacht maar afwezig" telt als object | `masks.py` |
| Crashes op evenwijdige, teruglopende of piepkleine randen | Randen samenvoegen, afrondingen begrenzen, randen korter dan 3 px opruimen; `is_valid` vangt rekenfouten af | `profile.py`, `cadhelpers.py` |
| Afschuining trekt het werkassenstelsel scheef (15,8° in plaats van 17°) | Hoofdrichting = lengtegewogen modus, verfijnd over de randen binnen ±3° | `profile.py` |
| Ronde delen 0,4 % te groot (cirkel als 30-hoek gefit) | Veelhoeken met dezelfde oppervlakte als de cirkel | `profile.py`, `silhouette.py` |
| Gat met een kleine maskerfout werd een "uitsparing" | Grootste ingeschreven cirkel, daarna een cirkelfit op alleen de passende contourpunten | `profile.py` |
| Optimalisatie bleef steken in smalle valleien (hoogte tegen randen) | Patroonstappen (Hooke-Jeeves) en herstarts | `silhouette.py` |
| Stille foute modellen | Kwaliteitspoort: IoU, convergentie, aantal randen, niet-ronde uitsparingen. `betrouwbaarheid` staat in het rapport | `pipeline.py` |
| Geen zicht op wat er misgaat | `debug/`: maskers van de bovenaanzichten, lokalisatie, stemmenbeeld, `diagnose.json` per foto | `debug.py` |
| Verkeerde matmaat en gedraaid opgeslagen foto's gaven misleidende meldingen | Andere matmaat wordt herkend en genoemd; staand opgeslagen foto's worden automatisch teruggedraaid | `pipeline.py` |
| Printschaal niet te corrigeren (bij "passend op pagina" 3–6 %) | `--meetlijn` op de opdrachtregel en een veld op de telefoonpagina | `pipeline.py`, `server/` |
| U95 van afrondingen te optimistisch (±0,21 mm, fouten tot 0,47 mm) | ±0,8 px per hoek, zonder de aanname dat die met meer foto's afneemt | `cadmodel.py` |
| Upload zonder token werd eerst helemaal ingelezen (tot 12 GB); de worker kon stoppen; halve scans bleven staan | Token- en groottecontrole vóór het lezen, worker vangt fouten af, opruimen bij een mislukte upload | `server/app.py` |
| Scannaam kon code in `model.py` zetten; Windows-console crashte op "≤" | Naam opschonen; console met `errors="replace"` | `cadmodel.py`, `cli.py` |
| OpenCV 4.8 laadt niet met numpy 2 | Minimaal `opencv-python-headless>=4.10` | `pyproject.toml` |
| Brede belichtingscorrectie traag (~0,2 s per foto) | Op 1/8 resolutie (7 ms), zelfde uitkomst | `masks.py` |

## 3b. Opgelost in v0.3 (Fase 0)

| # | Wat | Hoe | Waar |
|---|---|---|---|
| V1 | Gereedschap voor de meetset met echte onderdelen | `camtocad valideer`: per onderdeel een map met foto's en een `maten.json` met schuifmaatmetingen. Koppelt elke maat aan het model en geeft per maat de fout en of die binnen U95 valt. Per soort maat volgen bias en spreiding, plus een printschaalcontrole (alle lengtes procentueel te groot of te klein). Het protocol staat in [FASE-0.md](FASE-0.md). De demo levert ook zo'n map op. **De fotoset zelf moet nog gemaakt worden** | `validate.py` |
| V5 | Mat v2 | Een raster witte stippen (1 mm, steek 2,5 mm) in elk zwart vak. De zone rond de schaakbordhoeken (4 mm) en een strook langs de randen blijven vrij, zodat de hoekdetectie en de onscherptemeting niet gestoord worden. Elk formaat heeft een eigen marker-ID-bereik (DICT_5X5_1000: A4 250–297, Letter 300–343, A3 350–457): formaat én versie worden aan de foto's herkend (`--mat` is standaard `auto`). Letter-formaat toegevoegd. De v1-matten blijven werken | `mat.py`, `calib.py` |
| V6 | Fotocontrole vooraf | Per foto ~0,1 s: mat gevonden, onscherpte, belichting en kijkhoek, met een oordeel goed, matig of onbruikbaar. De onscherpte is σ in pixels, gemeten door de voorspelde matpatches te vervagen tot ze op de foto passen (nauwkeurig tot ~0,05 px). Per scan: waar het object ligt (hoeken en markers die steeds ontbreken), een dekkingskaart per richting en hoogte, en concrete aanwijzingen. De telefoonpagina uploadt per foto met direct oordeel, toont de dekkingskaart, laat slechte foto's verwijderen en foto's toevoegen aan een bestaande of mislukte scan, en toont de debugbeelden. Verkleinen op de telefoon maakt de upload ~5× kleiner. Op de opdrachtregel: `camtocad controleer` | `preflight.py`, `server/` |
| V7 | Printschaal per richting | Meetlijnen X en Y (`--meetlijn X Y`, twee velden op de telefoonpagina). De schaal zit in de matgeometrie van kalibratie, poses en maskers, dus ook een ongelijke schaal wordt goed verwerkt, en de hoogte schaalt mee | `mat.py`, `calib.py`, `pipeline.py` |
| — | Onscherpte en dekking in de verwerking | Per foto de onscherpte in `diagnose.json`, met een waarschuwing bij σ > 1,8 px. Bij "geen bovenaanzichten" of "geen objectcontour" staat in de melding welke foto's rond het object ontbreken; bij een onvolledige set volgt een waarschuwing in het rapport | `pipeline.py` |
| — | Maskers rond het object | Drie reparaties, gevonden bij het testen van mat v2 en ook nuttig met v1. (1) De lokale versterking (schaduw) gebruikt alleen vensters waar niveau en contrast dezelfde versterking geven, en niet vlak naast ontbrekende textuur (het object). Anders liet een gatrand die samenvalt met een vakrand het object ernaast als "beschaduwd wit" wegvallen. (2) De randstrook "zekere mat" voor de fit bevat alleen pixels die duidelijk op de mat lijken. (3) Randpixels zonder bruikbaar contrast volgen de meerderheid van hun buren. De maskerrand heeft daardoor geen bias meer die met het matpatroon meeverandert | `masks.py` |
| — | Startcontour en kwaliteitspoort | Geeft een rafelige rand een ongeldige omtrek, dan eerst gladgestreken opnieuw proberen, in plaats van terug te vallen op een lagere stemdrempel die juist meer schaduw meeneemt. Nieuwe signalen voor "onbetrouwbaar": twee of meer zeer korte randen (< 2,5 mm; een uitstulping of inham die er niet is), een knik van minder dan 10° in een rand (vaak een schaduw langs die rand), een afronding die groter is dan de randen eromheen, en gaten waardoor in geen enkel bovenaanzicht zekere mat te zien is (spookgaten) | `initial.py`, `pipeline.py` |
| V21 (deels) | Sneller | De maskers per foto lopen parallel in threads (OpenCV en numpy geven de GIL vrij bij grote beelden): 2,4× sneller op 4 kernen. De silhouetenergie heeft minder Python-overhead per foto: 81 in plaats van 229 ms per evaluatie. Threads maakten die juist trager. De fit stopt als de laatste 200 evaluaties samen < 0,1% opleveren, of na hooguit één extra blok als het model matig past (IoU < 0,95). Een rommelige startcontour (> 20 randen, gaten en uitsparingen) krijgt maar een korte verfijning. De demoscan (46 foto's van 1600 × 1200) kost daarmee ~70 s op 4 kernen; een mislukte scan kost geen tientallen minuten meer | `pipeline.py`, `silhouette.py` |

## 3c. Opgelost in v0.4 (V8, eerste stap)

Het zwarte onderdeel mislukte in v0.3 ook op mat v2 (§2). De analyse met de echte objectmaskers van de gerenderde scan liet drie oorzaken zien:

- **12% van het object was "onbekend".** Dat zijn de zwarte stukken mat zonder stippen: de stroken langs de vakranden, de hoeken en de zwarte vlakken van de markers. Die gaten zitten aan de mat vast (z = 0), dus in elk bovenaanzicht op dezelfde plek. Gevolg: nepgaten in de startcontour, en een hoogtezoektocht die op 2 mm uitkwam in plaats van 12.
- **"Zekere mat" lekte tot 4 pixels het object in.** Het textuurvenster (7 × 7) van een objectpixel vlak bij de rand ziet de stippen van de mat ernaast nog. Omgekeerd lekt "textuur ontbreekt" even ver het gat in. Samen duwden ze gaten groter of kleiner en de buitenrand naar binnen.
- **De fit vond geen afrondingen vanuit een scherpe hoek.** Een kleine afronding levert bijna niets op (het weggesneden stukje groeit met R²), en de fit stopte eerder. De energie had wél een duidelijk minimum bij R3: samen bijna de helft van de totale energie.

| Wat | Hoe | Waar |
|---|---|---|
| Dubbelzinnige pixels | Per foto de lokale grijswaarde van het object. Waar de voorspelde mat binnen 2τ gelijk is (zwart op zwart, wit op wit), is een pixel geen bewijs. Een nieuwe klasse `amb`: de fit negeert die pixels, ook als ze voor de startcontour zijn opgevuld | `masks.py`, `silhouette.py` |
| Geen lek meer | In zulke zones telt alleen de kern van een geverifieerd matgebied als zekere mat, en de strook "textuur ontbreekt" langs de rand telt niet als object. De rand ligt ertussen; de fit bepaalt hem uit het bewijs rondom. Valse zekere mat in het object: van 20 370 naar ~700 pixels over 46 foto's | `masks.py` |
| Opvullen (V8, stap 1) | Een dubbelzinnig stuk binnen de sluiting van het object (schijf van 3,5 mm) wordt object, tenzij het daarbinnen aan matbewijs grenst: zekere mat, of een pixel die duidelijk op de mat lijkt waar het object wél zichtbaar zou zijn. Afgedekte witte stippen tellen mee als object. "Onbekend" in het object: van 12% naar 0,2% | `masks.py` |
| Toets per gebied | Per pixel is een slagschaduw op wit vaak even grijs als een grijs object; over het hele stuk is het verschil duidelijk. Een stuk dat als geheel op de mat lijkt, wordt niet opgevuld, als object en mat daar minstens 10 grijswaarden verschillen. Bij zwart op zwart (~5) kan dat niet, daar beslist de opvulling | `masks.py` |
| Gaten naast een dubbelzinnig vlak | Grenst zo'n vlak aan een echt gat, dan gaat elke pixel naar het dichtstbijzijnde bewijs, object of mat. In de stemkaart komt een laag "weet niet", en de cirkel van een gat wordt gefit op alleen de zichtbare rand. Daardoor geen vierkante of gestaarte gaten meer | `masks.py`, `initial.py`, `profile.py` |
| Afrondingsproef | Na de fit per hoek R = 0, 0,5, 1 … 10 mm proberen, daarna een korte verfijning. Energie van het zwarte onderdeel: 6360 → 3432 | `silhouette.py`, `pipeline.py` |
| Hoekpunten weghalen (V4, deels) | De fit kan geen hoekpunten weghalen. Voor elke knik < 10° en elke rand < 5 mm volgt een korte fit zonder dat hoekpunt; past het model even goed (energie +0,5% of minder, of één pixel per foto), dan blijft het weg. Een echt kenmerk of een schaduw heeft bewijs in de foto's en blijft staan, en de kwaliteitspoort markeert een schaduw | `pipeline.py`, `profile.py` |
| Debugbeelden | Dubbelzinnige pixels paars in `masker_*.jpg` | `debug.py` |
| Rekentijd | De extra stappen alleen in een uitsnede rond het object: +~15% per foto. Het weghalen van hoekpunten kost alleen tijd als er kandidaten zijn (~15 s per poging) | `masks.py` |

## 3d. Opgelost in v0.4.1 (eerste echte fotoset)

| Wat | Hoe | Waar |
|---|---|---|
| Verplaatst onderdeel herkennen | Na de maskers, vóór de hull. Per voxel van 3 mm: in hoeveel foto's valt hij op het object, en in hoeveel duidelijk op de mat (ook egale stukken, niet waar het object onzichtbaar zou zijn). Per foto twee toetsen tegen de consensus van de andere: langs de kijkstraal van elke objectpixel moet de consensus hoog zijn (25e percentiel, per maskerdeel; een losse vlek telt niet), en de foto mag de gezamenlijke hull niet op de mat zien. Groeperen: houd de foto's die het best passen (Otsu, of de slechtste 10% afpellen bij een gelijke stand) tot de groep klopt, en herhaal voor de rest. Past minder dan 75% van de foto's bij de grootste groep, dan een melding met de groepen (bij één groep met de kanttekening dat het ook aan de maskers kan liggen); anders gaan de paar afwijkers er met een waarschuwing uit. Kost ~3–5 s. Een klein duwtje (10 mm bij een onderdeel van 80 mm) valt niet op: de liggingen overlappen dan grotendeels. De kwaliteitspoort markeert zo'n model wel (IoU 0,69 in de slechtste foto) | `placement.py`, `pipeline.py` |
| Onscherpte in het masker (V13, deels) | De voorspelde mat wordt vervaagd tot de gemeten onscherpte van de foto (`preflight.py`). Anders geeft elke zwart-witrand van de mat in een bewogen foto aan weerszijden een afwijking die op object lijkt | `masks.py` |
| Posefout per foto | De tolerantie aan patroonranden wordt per foto gemeten aan de mat zelf: het 75e percentiel van afwijking gedeeld door helling op duidelijke randen, tussen 0,4 en 2,5 px. Synthetisch blijft het 0,4 px; bij de echte foto's 0,5–2 px | `masks.py` |
| Glans | Glans op de toner maakt zwart lichter en laat wit bijna gelijk; dat is geen versterking. Een aparte "zwart-optilling", gemeten binnen de zwarte vakken (fijn waar genoeg bronnen zijn, grof daartussen) | `masks.py` |
| Schaduw tot over de papierrand | Waar in de buurt geen bronnen voor de versterking zijn (de witte rand, midden in een groot vak) volgt de versterking het grove verloop in plaats van "geen schaduw". Niet binnen ~10 mm van ontbrekende textuur: daar ontbreken de bronnen door het object zelf, en een doorgetrokken schaduw liet in het zware stressscenario stukjes object als mat doorgaan (spookgaatjes van Ø 0,8 mm) | `masks.py` |
| Dubbelzinnig alleen bij het object | "Even donker als het object" alleen in de buurt van het object, en nooit meer dan een kwart van het zwart-witcontrast van de mat. Anders werd bij een ruisige foto de halve mat paars | `masks.py` |

Geprobeerd en teruggedraaid: de ruis alleen op de vlakke stukken mat schatten. Dat gaf een lagere drempel, maar in het zware stressscenario telde de rand van de slagschaduw dan als object: één gat 0,38 mm te klein, zonder waarschuwing. Op de echte foto's was het verschil met de oude schatting niet systematisch. Ook de grove versterking overal toepassen bleek slecht: op de echte foto's gaf dat juist meer losse vlekken (1,2% van de mat tegen 0,8%).

## 3e. Opgelost in v0.5 (V2, V3, V4)

| Wat | Hoe | Waar |
|---|---|---|
| Randfit (V2) | Na de pixelfit een kleinste-kwadratenfit op subpixelafstanden tot de silhouetrand. Modelpunten (~1 per mm) op de onder- en bovenrand van de contour en de gaten; per foto telt de onderrand als de wand naar de camera kijkt, anders de bovenrand, en alleen punten op het modelsilhouet. Residu: afstand tot de rand van het objectmasker (bilineair), min 0,2 × de strook zonder bewijs tussen object en zekere mat (tot 1,5 px; breder is gebrek aan bewijs, niet menging), gewogen naar hoeveel zekere mat er vlakbij is. Cauchy-verlies (0,5 px), dus een uitschieter telt nauwelijks. Alleen binnen ±0,5 mm en ±0,3° van de pixelfit, anders blijft die staan. Een parameter die de contour bij een stap van 0,1 mm ongeldig maakt (een afronding die net past, de randjes van een hap uit het masker), blijft vast: zijn afgeleide is een sprong naar de strafwaarde en zette in het zware stressscenario de hele fit vast | `edgefit.py`, `pipeline.py` |
| U95 per maat (V3) | Toevallig deel: jackknife over 6 groepen foto's (per groep opnieuw oplossen zonder die foto's). De formele covariantie uit de Jacobiaan is veel te optimistisch (honderden randpunten per foto zijn niet onafhankelijk), en één Gauss-Newtonstap per groep onderschat de spreiding tot 20× (afrondingen). Per maat σ² = gᵀCg, met g de numerieke gradiënt van de maat in het werkassenstelsel (randposities t.o.v. de datumrand, gatposities t.o.v. de datum). Systematisch deel per soort maat, in pixels op het object (tabel hieronder). Printschaal: 0,05% met gemeten meetlijnen, anders 0,3%. Snappen gebeurt zonder de printschaal (een schaalfout verschuift alle maten samen), het rapport telt hem wel mee. Staat de randfit niet, dan blijft de indicatieve U95 van v0.4 (met dezelfde printschaalterm) | `uncertainty.py`, `edgefit.py`, `cadmodel.py`, `pipeline.py` |
| Gemiste gaten (V4) | Het bovenvlak (z = hoogte) wordt in rasters van 0,25 mm teruggeprojecteerd in de foto's recht van boven (≤ 25°). Waar ≥ 2 foto's én de meerderheid zekere mat zien, niet binnen 1 mm van een rand of bestaand gat, ligt een kandidaat; een cluster van ≥ Ø 1,5 mm wordt een cirkel. Eerst alleen dat gat op maat fitten, dan kort alles; het gat blijft als de energie duidelijk daalt (meer dan 0,2% + 20). Een (bijna) ronde uitsparing wordt een gat | `holes.py`, `pipeline.py` |
| Scherpe hoeken | De randfit ziet een scherpe hoek als een afronding van 3,5–4 px (onscherpte en masker ronden hem af). De grens voor "niet te onderscheiden van scherp" gaat daarom van 3 naar 4,5 px (~1,1 mm bij 0,25 mm/px). Zonder die stap kreeg de L-vorm twee afrondingen van R1 op scherpe hoeken | `pipeline.py` |
| Renderer van de stresstests | 3 × 3 supersampling in plaats van 2 × 2: met een even aantal was een half bedekte pixel "geen object", en de gerenderde objectmaskers lagen 0,14 px binnen de ware rand. De stressresultaten van v0.4 waren daardoor iets te klein. Alle v0.5-cijfers gebruiken opnieuw gerenderde scans; ter vergelijking draaide ook v0.4.1 daarop | `render.py` |

**IJking van de systematische termen.** De randfit met jackknife op tien gerenderde scans (beugel in zes stressscenario's, plaatje, ring, L-vorm en flens), 110 maten tegen de waarheid. Per soort de kleinste systematische term waarbij 95% van de fouten binnen U95 valt, en de gekozen waarde (met wat marge):

| Soort | n | \|fout\| mediaan / max (mm) | Binnen 2σ van alleen de jackknife | Nodig voor 95% | Gekozen (px) | Binnen U95 | U95 mediaan (mm) |
|---|---|---|---|---|---|---|---|
| lengte | 18 | 0,036 / 0,058 | 17% | 0,12 px | 0,12 | 100% | 0,062 |
| hoogte | 10 | 0,024 / 0,043 | 20% | 0,09 px | 0,10 | 100% | 0,052 |
| gat Ø | 19 | 0,035 / 0,309 | 42% | 0,27 px | 0,30 | 100% | 0,154 |
| positie | 31 | 0,027 / 0,154 | 52% | 0,12 px | 0,15 | 100% | 0,081 |
| afronding | 32 | 0,071 / 0,431 | 44% | 0,74 px | 0,80 | 100% | 0,411 |

De jackknife alleen is dus te optimistisch: op deze scans is de fout vooral systematisch. De maskerrand ligt per scène net anders (een donker of licht onderdeel: lengtes +0,04 tot +0,06 mm, een klein plaatje −0,05 mm), en kleine diepe gaten (Ø 4,5 in 10 mm) komen tot 0,14 mm te klein uit. Afrondingen blijven het zwakst (tot ±0,4 mm, vooral waar de pixelfit een extra randje in de contour liet). Dit is afgesteld op gerenderde scans; op echte foto's kan de systematiek groter zijn (V1).

**Stresstests v0.5.** De scenario's van §2b plus drie extra (vervormde mat, gamma, zwaar met een klein donker onderdeel), opnieuw gerenderd met de nieuwe renderer. Ter vergelijking draaide v0.4.1 op dezelfde scans. **Goed** betekent: geen waarschuwing. De maten zijn gefit, vóór het snappen.

| Scenario | v0.5 | v0.4.1 (zelfde scans) |
|---|---|---|
| Basis | Goed: 79,98 × 40,02 × 12,01, R3,07, Ø 6,59 | Goed: 79,95 × 39,94 × 12,06, R3,06, Ø 6,59 |
| Realistisch | Goed: 79,99 × 40,01 × 12,00, R3,03, Ø 6,60 | Goed: 79,98 × 39,94 × 12,04, R3,16, Ø 6,59 |
| Harde schaduw | Gemarkeerd (twee korte randen aan de schaduwkant): 80,06 × 40,06 × 11,96, Ø 6,56 | Gemarkeerd |
| Donker onderdeel | Goed: 80,04 × 40,05 × 11,98, R3,13, Ø 6,57 en **6,29 ± 1,16** | Goed: 79,95 × 39,93 × 12,03, Ø 6,43 en **5,99 ± 0,16, gesnapt naar 6,0** |
| Wit onderdeel | Goed: 80,04 × 40,05 × 12,02, R3,12, Ø 6,56 | Goed: 80,00 × 39,98 × 12,00, R3,19, Ø 6,56 |
| Handschaduw | Goed: 79,98 × 40,01 × 12,00, R3,04, Ø 6,61 | Goed: 79,98 × 39,92 × 12,06, R3,14, Ø 6,61 |
| Verspreide bovenaanzichten | Goed: 79,98 × 40,01 × 12,00, R3,03, Ø 6,58 | Goed: 79,94 × 39,97 × 12,06, R3,16, Ø 6,57 |
| Weinig lage foto's | Goed: 79,99 × 40,01 × 12,02, R3,07, Ø 6,59 | Goed: 79,97 × 39,99 × 12,04, R3,18, Ø 6,60 |
| Klein onderdeel | Goed: 29,95 × 19,97 × 5,03, R2,01, Ø 4,46 | Goed: 29,89 × 19,92 × 5,07, R2,00, Ø 4,45 |
| Klein, weinig lage foto's | Goed: 29,99 × 20,00 × 5,02, R2,18, Ø 4,43 | Goed: 29,94 × 19,94 × 5,05, R2,07, Ø 4,43 |
| Ring | Goed: Ø 24,95 × 8,04, gat Ø 7,97 | Goed: Ø 24,91 × 8,07, gat Ø 7,98 |
| Zwaar | Gemarkeerd (een schaduwbult): 79,97 × 40,03 × 12,02, Ø 6,56 | Gemarkeerd, Ø 6,25 en 6,60 |
| Vervormde mat | Goed: 79,97 × 40,01 × 12,01, R3,04, Ø 6,59 | Goed: 79,94 × 39,94 × 12,06, R3,17, Ø 6,61 |
| Gamma | Goed: 80,02 × 40,03 × 12,01, R3,09, Ø 6,57 | Goed: 80,00 × 40,00 × 12,03, R3,26, Ø 6,56 |
| Zwaar, klein en donker | Gemarkeerd: rommelige contour (16 randen, R tot 9,5, gat Ø 3,95); de randfit loopt tegen zijn vertrouwensgebied en blijft uit | Gemarkeerd, zelfde model |

Over de 101 maten van de scans zonder waarschuwing:

| Soort | \|fout\| mediaan / max v0.5 | v0.4.1 | U95 mediaan v0.5 (zonder printschaal) | v0.4.1 (met 0,05% printschaal) |
|---|---|---|---|---|
| hoogte | 0,015 / 0,040 | 0,055 / 0,073 | 0,051 | 0,105 |
| lengtes | 0,017 / 0,052 | 0,055 / 0,110 | 0,062 | 0,154 |
| gaten Ø | 0,027 / 0,309 | 0,029 / 0,607 | 0,154 | 0,123 |
| gatposities | 0,027 / 0,154 | 0,034 / 0,326 | 0,079 | 0,111 |
| afrondingen | 0,068 / 0,183 | 0,161 / 0,261 | 0,408 | 0,286 |

- **Binnen U95:** 100% in v0.5 (101 van 101); in v0.4.1 95%, maar de grootste fouten vielen er ver buiten. Het gat van het donkere onderdeel boven het zwarte markervlak was in v0.4.1 Ø 5,99 ± 0,16 en werd naar 6,0 gesnapt (waar: 6,6). In v0.5 is het Ø 6,29 ± 1,16: de jackknife ziet dat dit gat slecht bepaald is, en snappen gebeurt dan niet.
- **Snappen:** 84 maten gesnapt, alle goed (v0.4.1: 79, waarvan 1 fout).
- **Hoogte en lengtes ~3× nauwkeuriger.** De pixelfit kwam ~0,05 mm te klein uit door de modelrenderer (V29); de randfit vergelijkt direct met de maskers.
- **Gemarkeerde scans** blijven gemarkeerd: geen stille fouten. Hun maten vallen niet altijd binnen U95 (in het zware kleine scenario is het model zelf fout); de waarschuwing zegt dat.
- **Rekentijd:** 16–121 s per scan (v0.4.1 op dezelfde scans: 23–141 s, twee scans tegelijk op 4 kernen). De randfit kost 0–9 s, de jackknife 0–11 s.

Nog gevonden, niet opgelost:

- De snelle modelrenderer van de pixelfit (`silhouette.render`) tekent in schuine aanzichten ~0,15–0,18 px te ruim (gemeten tegen 5 × 5 supersampling; vermoedelijk dunne wandvierhoeken die na het krimpen van 0,5 px omklappen). Daardoor komt de pixelfit ~0,05 mm te klein uit. De randfit heeft er geen last van: die vergelijkt direct met de maskers. Zie V29.
- De startcontour kan bij een afgeronde hoek een knikje van een paar graden krijgen (een koorde die bijna in het verlengde van de rand ligt). `_simplify_outline` haalt het weg, maar het maakt de startcontour ~2% te groot.
- Wie de meetlijnen precies 100,0 mm meet, is niet te onderscheiden van wie ze niet opgeeft; beide krijgen 0,3% printschaal in de U95.

## 4. Open verbeterpunten, op prioriteit

Impact en moeite: **H**oog, **M**iddel, **L**aag. Moeite S/M/L staat voor dagen, een week, of meerdere weken.

### 4.1 Eerst (Fase 0 en v0.5)

V5, V6 en V7 zijn in v0.3 gedaan, en voor V1 staat het gereedschap klaar (§3b). Hieronder staat per punt wat er nog open is.

| # | Verbetering | Waarom | Aanpak | Impact | Moeite |
|---|---|---|---|---|---|
| V1 | **Echte fotoset met schuifmaatmetingen** (gereedschap klaar in v0.3) | Alle drempels en U95 zijn nu op synthetische scans afgesteld | 10–20 onderdelen: metaal, zwart, wit, kunststof. Referentie met eindmaten en een ringkaliber, in de stijl van [ISO 10360-13 / VDI 2634](https://www.nist.gov/publications/vdivde-2634-2-and-iso-10360-13-performance-evaluation-tests-and-systematic-errors). Draaien als regressiesuite. **Open:** de set zelf maken volgens [FASE-0.md](FASE-0.md) en daarmee U95 en de drempels afstellen | H | M |
| V2 | ~~Fit op randafstanden in plaats van pixeltelling~~ (gedaan in v0.5) | Pixeltelling is een trapfunctie: traag, geen covariantie, en afrondingen dwalen ±0,3 mm | Gedaan: `edgefit.py`, zie §3e. **Open:** subpixelranden uit de grijswaarden zelf ([Devernay, IPOL](https://www.ipol.im/pub/art/2017/216/)) in plaats van de maskerrand; dan hoeft de strook zonder bewijs niet meer met een vaste fractie (0,2) te worden verdeeld | H | M |
| V3 | ~~U95 die klopt~~ (gedaan in v0.5, op gerenderde scans) | U95 volgde alleen uit de resolutie | Gedaan: jackknife over groepen foto's, systematiek per soort maat en printschaal, afgesteld op 95% dekking (§3e). **Open:** toetsen en bijstellen op echte foto's met V1; kalibratie-σ ([mrcal](https://mrcal.secretsauce.net/uncertainty.html)) als aparte term | H | M |
| V4 | ~~Topologie bijwerken na de fit~~ (gedaan in v0.4 en v0.5) | De fit kan geen gaten of randen toevoegen: een gemist gat blijft gemist | Gedaan: knikken en korte randen weg als het model zonder even goed past (§3c); gemiste gaten toevoegen en ronde uitsparingen als gat (§3e). **Open:** een gemiste uitstulping of inham in de buitencontour (residuclusters langs de rand, V14) | H | M |
| V5 | ~~Mat v2~~ (gedaan in v0.3) | Een donker onderdeel op een zwart vak is onzichtbaar: in de stresstest werd maar ~50 % van het silhouet gezien | Gedaan: stippenraster in de zwarte vakken, eigen marker-ID's per formaat, Letter. **Open:** stippen in de zwarte vlakken van de markers en donkere stippen in de witte marges eromheen (de markerdetectie mag er niet onder lijden). Een gat van een zwart onderdeel boven een groot zwart markervlak is nu het zwakste punt: de rand is daar in de bovenaanzichten niet te zien. Verder middengrijze vakken, en controleren of de stippen op gewone printers goed uitkomen (V1) | H | M |
| V6 | ~~Preflight bij het uploaden~~ (gedaan in v0.3) | Een slechte fotoset blijkt nu pas na de verwerking | Gedaan: zie §3b. **Open:** een live camerabeeld met dezelfde controle (V25) | H | M |

### 4.2 Robuustheid op echte foto's

| # | Verbetering | Aanpak | Impact | Moeite |
|---|---|---|---|---|
| V7 | ~~Anisotrope printschaal (sx ≠ sy)~~ (gedaan in v0.3) | Gedaan: meetlijnen X en Y, de schaal zit in de matgeometrie. De printschaalcontrole zit in `camtocad valideer` (alle lengtes procentueel te groot of te klein). **Open:** de meetlijnen zijn in de foto's zelf niet te meten, want ze schalen mee met de print; alleen een onafhankelijk object met bekende maat (eindmaat, bankpas 85,60 × 53,98 mm) kan de schaal controleren | H | S–M |
| V8 | Segmentatiecascade (eerste stap gedaan in v0.4) | Gedaan: dubbelzinnige stukken binnen het object opvullen, zonder lek van het textuurvenster, met een toets per gebied (§3c). **Open:** GrabCut met de driedeling object/mat/onbekend. Kleur (afstand tot de zwart-witte mat) als extra objectbewijs. [PyMatting](https://github.com/pymatting/pymatting) (MIT) voor een subpixel-alfarand. Schaduwdetectie op regioniveau: een harde slagschaduw wordt nu gemarkeerd, maar nog niet weggewerkt ([overzicht](https://arxiv.org/abs/1304.1233)) | H | M |
| V9 | Optioneel een geleerd masker, alleen met Apache-2.0-code en -gewichten | [SAM 2.1](https://github.com/facebookresearch/sam2), [HQ-SAM 2](https://github.com/SysCV/sam-hq) of [EfficientViT-SAM](https://huggingface.co/mit-han-lab/efficientvit-sam) via ONNX Runtime op de CPU, geprompt met een kader en punten uit het matmasker. Altijd combineren met het matresidu en met meerdere foto's: SAM faalt op spiegelend metaal en lage contrasten. De CPU-snelheid is nog niet gemeten | M–H | M |
| V10 | Foto's zoals telefoons ze maken | EXIF lezen: oriëntatie, lens, brandpunt, digitale zoom. HEIC via pillow-heif. Groeperen per camera of lens, en een cameramodel per toestel bewaren (voor kleine scans). Waarschuwen bij σ(f)/f > 0,3 % | H | M |
| V11 | Mat niet vlak | Een residukaart per hoek over alle foto's toont krul. Eventueel een bundelaanpassing met een laag-orde matoppervlak | M | M |
| V12 | Meer dan één ding op de mat, of het object deels ernaast | Waarschuwen, en het object kiezen dat in de bovenaanzichten steun heeft. Een liniaal of munt kan groter zijn dan het onderdeel. "Ligt deels naast de mat" als expliciete melding | M | S |
| V13 | Maskerrand-bias en onscherpte per foto (deels gedaan in v0.3, v0.4.1 en v0.5) | Gedaan: de onscherpte per foto wordt aan de mat gemeten (`preflight.py`), staat in `diagnose.json`, en onscherpe foto's worden gemeld. De randbias hangt niet meer van het matpatroon af (§3b). De voorspelde mat wordt vervaagd tot de gemeten σ (§3d). De randfit legt de rand op 0,2 van de strook zonder bewijs (§3e). **Open:** de kleinste herkenbare afronding per scan uit de gemeten onscherpte afleiden (nu vast 4,5 px, afgesteld op gerenderde scans met weinig onscherpte) | M | M |
| V14 | Kwaliteitspoort verfijnen (korte randen: gedaan in v0.3) | Residuclusters per foto: een gemist gat, een extra uitstulping. Snappen beoordelen op het energieverschil in plaats van het IoU-verschil, want 0,5 mm fout verandert de IoU maar ~0,005 | M | S |
| V27 | Een klein duwtje herkennen (grote verplaatsing: gedaan in v0.4.1) | Een verschuiving van een paar millimeter valt in de consensus niet op (§3d). Na de fit per foto de verschuiving van het silhouet t.o.v. het model schatten; een groep opeenvolgende foto's met dezelfde verschuiving is een duwtje. Melden, of die groep apart fitten en de verschuiving meenemen | M | S |
| V28 | Kalibratie op echte foto's | De eerste echte set haalde 1,6 px reprojectiefout, vooral door bewogen foto's en hoekruis. Foto's met σ > 3 px niet voor de kalibratie gebruiken (wel voor de maskers, als ze scherp genoeg zijn), hoeken wegen naar hun onscherpte, en in de fotocontrole waarschuwen als de camera dichter dan ~15 cm bij de mat is | M | S |
| V29 | Exacte modelrenderer voor de pixelfit | `silhouette.render` krimpt elke polygoon 0,5 px om `cv2.fillPoly` exact te maken, maar in schuine aanzichten wordt het silhouet toch ~0,15 px te ruim (vermoedelijk klappen dunne wandvierhoeken om) en de pixelfit ~0,05 mm te klein (§3e). Wanden als één band tekenen, of dunne wanden overslaan als de boven- en onderrand het silhouet al dekken. Daarna de drempels van de pixelfit opnieuw toetsen | M | S |

### 4.3 Grotere objectklasse

| # | Verbetering | Aanpak | Impact | Moeite |
|---|---|---|---|---|
| V15 | Sleuven en rechthoekige uitsparingen parametrisch | Nu zijn het ruwe polygonen, door parallax te klein. Een sleuf- en rechthoekmodel dat ook gefit wordt | H | S–M |
| V16 | Randen uit beeldgradiënten van de bovenaanzichten | Gatranden, verzinkingen, treden en afschuiningen zijn in silhouetten onzichtbaar maar in de foto wel te zien | H | M |
| V17 | Getrapte prisma's en afschuiningen op de bovenrand | Hoogtezoektocht per pixel geeft niveaus; een afschuining als gelaagd prisma | H | L |
| V18 | Draaidelen die liggen | Een revolve-model, plus een generieke mesh-silhouetrenderer waarmee elk parametrisch CadQuery-sjabloon te fitten is | M–H | L |
| V19 | "Niet 2,5D" herkennen | Melden in plaats van een fout model. Blinde gaten als vraag aan de gebruiker | M | M |
| V20 | Vrije vormen | [OpenMVS 2.4](https://github.com/cdcseacave/openMVS) (AGPL-3.0, dieptekaarten op de CPU, maskers, COLMAP-import) met de matposes. Daarna primitieven fitten (pyransac3d, Open3D, CGAL) en naar CadQuery. Geleerde CAD-voorstellen ([CADReasoner](https://github.com/zhemdi/CADReasoner) Apache-2.0, [CADENA](https://arxiv.org/abs/2608.00799) MIT) hooguit als optionele GPU-plugin, met de parameters altijd opnieuw gefit op de foto's. De auteurs van CADENA melden zelf een flinke kwaliteitsdaling op echte onderdelen | M | L |

### 4.4 Software, prestaties, distributie

| # | Verbetering | Aanpak | Impact | Moeite |
|---|---|---|---|---|
| V21 | Nog sneller (deels gedaan in v0.3: threads, eerder stoppen) | Carving per z-vlak met één homografie, undistort-maps één keer berekenen, grote JPEG's verkleind decoderen, gatringen vectoriseren | M | S |
| V22 | Structuur | Stapfuncties met gecachte tussenresultaten (`--vanaf`). Alle drempels in één `Tuning`-dataclass, ook in `report.json`. Een `ScanResult`, dode code weg (fijne hull, `surface_points`). Maskers als uitsnede: nu ~3,6 GB bij 300 foto's | M | M |
| V23 | Installatie | Lockfile (uv of pixi). CadQuery 2.8 met cadquery-ocp 7.9.3.x: 8.0.1 breekt CadQuery. VTK-vrije OCP. `camtocad doctor`: één cv2-wheel, versies, hoekverschuiving. ~~Paden met niet-ASCII-tekens op Windows~~ (gedaan in v0.3: `imgio.py`). Installer via [PyApp](https://github.com/ofek/pyapp) of pixi-pack | M | M |
| V24 | Server | HTTPS op het LAN, nodig voor een live camera in de browser ([getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)). Quota en opruimen, PDF atomisch schrijven, `--lan` expliciet, hervatten van een afgebroken upload, magic bytes controleren (verwijderen en per foto uploaden: gedaan in v0.3). Bij AGPL een bronlink in de pagina (§13) | M | S–M |
| V25 | Live begeleiding in de browser | [OpenCV.js](https://github.com/opencv/opencv/blob/4.x/platforms/js/opencv_js.config.py) herkent de mat live in het voorbeeldbeeld en geeft een dekkingskaart, scherpte en glans. Vereist V24 (HTTPS). WebXR werkt niet op iOS, dus daar niet op leunen | H | M |
| V26 | CI | Stresstests in klein formaat als regressietests. Matrix: OS × OpenCV 4.10/5.x × numpy 1/2 | M | M |

## 5. Bewust níet gebruiken (licentie)

Voor een project dat AGPL-3.0 wil worden, telt ook de licentie van de **gewichten**, niet alleen die van de code.

| Kandidaat | Licentie | Alternatief |
|---|---|---|
| RMBG-2.0 (ook het standaardmodel van rembg!) | CC BY-NC | BiRefNet, BEN2-base (MIT), U²-Net/IS-Net (Apache-2.0), altijd met een expliciet gekozen model |
| EdgeSAM | S-Lab (niet-commercieel) | EfficientViT-SAM, MobileSAM (Apache-2.0) |
| SAM 3 / 3.1 | Eigen SAM-licentie: gebruiksuitsluitingen, eenzijdig wijzigbaar | SAM 2.1 (Apache-2.0) |
| nvdiffrast, NeuS2, Neuralangelo | NVIDIA, niet-commercieel | PyTorch3D, Mitsuba 3 (BSD); instant-nsr-pl (MIT) |
| 2DGS, PGSR, GOF, MILo | Inria/Gaussian-Splatting-licentie (niet-commercieel) | gsplat (Apache-2.0) met Open3D-TSDF (MIT) |
| DUSt3R, MASt3R, CUT3R, π³, VGGT (origineel) | Niet-commercieel of op aanvraag | MapAnything (Apache-checkpoint), Depth Anything 3 Small/Base (Apache-2.0) |
| CAD-Recode, cadrille, Point2CAD, Text2CAD | Gewichten niet-commercieel | CADReasoner, CADENA (optioneel) |
| Img2CAD | Basismodel Llama 3.2 Vision sluit ingezetenen van de EU uit | — |

## 6. Voorgestelde volgorde

1. **Fase 0 met echte foto's.** Het gereedschap is klaar in v0.3: V1-harnas, V5 (mat v2), V6 (fotocontrole), V7 (printschaal). Sinds v0.4 horen ook zwarte en witte onderdelen in de meetset. Nu de meetset zelf maken en meten volgens [FASE-0.md](FASE-0.md): meten is weten.
2. **v0.4 (gedaan):** de eerste stap van V8 (zwart op zwart, wit op wit), hoekpunten weghalen uit V4, en de afrondingsproef in de fit (§3c).
   **v0.4.1 (gedaan):** na de eerste echte fotoset: een verplaatst onderdeel herkennen, en maskers die zich per foto aanpassen aan onscherpte, posefout en glans (§2c, §3d).
3. **v0.5 (gedaan):** V2 (fit op randen), V3 (U95 per maat) en de rest van V4 (gaten toevoegen), afgesteld op gerenderde scans (§3e). Het bijstellen op de Fase 0-metingen volgt met V1.
4. **v0.6:** de objectklasse (V15–V19) en de rest van de segmentatiecascade (V8: GrabCut, kleur, schaduw; V10).
5. **v0.7 en verder:** vrije vormen (V20), live begeleiding (V24–V25) en een installer (V23).

## 7. Verantwoording

- De stresstests draaien op gerenderde scans. Ze laten zien wáár het breekt, niet hoe nauwkeurig echte foto's zijn. Dat moet V1 uitwijzen.
- De bronnen zijn in september 2026 verzameld. Niet alles is door ons nagemeten. Licenties kunnen veranderen: controleer ze bij adoptie opnieuw.
- Belangrijkste bronnen:
  - OpenCV: [opencv#25539](https://github.com/opencv/opencv/issues/25539), [opencv#23152](https://github.com/opencv/opencv/issues/23152) (legacy-patroon), [opencv#25850](https://github.com/opencv/opencv/issues/25850), [migratie 4→5](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration).
  - Kalibratiepraktijk: [calib.io](https://calib.io/blogs/knowledge-base/calibration-best-practices).
  - Nauwkeurigheid met telefoons: [D3Mobile](https://www.sciencedirect.com/science/article/pii/S0263224121003353).
  - Fotoprotocol kleine objecten: [Falkingham](https://peterfalkingham.com/2019/01/16/small-object-photogrammetry-how-to-take-photos/).
  - [Installatie van CadQuery](https://cadquery.readthedocs.io/en/latest/installation.html).
  - [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html).
