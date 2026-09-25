# Route A — verbeterpunten (onderzoek, september 2026)

Dit document beschrijft wat er aan de implementatie van route A ([ROUTE-A.md](ROUTE-A.md)) beter kan. Een deel van de bevindingen zit al in v0.2 (§3), het gereedschap voor Fase 0 in v0.3 (§3b); de rest staat hier als geprioriteerde lijst (§4).

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
- **Grootste open punten:**
  - nauwkeurigheid en een eerlijke U95. De fit telt pixels, en afrondingen komen niet beter dan ±0,3 mm.
  - donkere onderdelen op de zwarte vakken. Hier is een betere mat nodig.
  - de objectklasse: treden, verzinkingen, afschuiningen.
  - begeleiding tijdens het fotograferen.

## 2. Stresstests: v0.1 tegenover v0.2

Proefstukken:

- **Beugel:** 80 × 40 × 12 mm, R3-hoeken, 2 gaten Ø 6,6.
- **Plaatje:** 30 × 20 × 5 mm, R2-hoeken, gat Ø 4,5.
- **Ring:** Ø 25 × 8 mm, gat Ø 8.

Ruis, JPEG-compressie, vignettering en verscherping zitten in alle scenario's behalve "basis". Maten zijn de gefitte waarden vóór het snappen. **Stil fout** betekent: een fout model zonder waarschuwing.

| Scenario | Situatie | v0.1 | v0.2 |
|---|---|---|---|
| Basis | Beugel, 46 foto's, schone render | Goed: 79,96 × 39,95 × 12,02 | Goed: 79,86 × 39,93 × 12,07 |
| Realistisch | Ruis, JPEG, vignettering, verscherping | Goed: 79,92 × 39,89 × 12,02 | Goed: 79,94 × 39,91 × 12,03 |
| Harde schaduw | Slagschaduw van 55 % | **Stil fout**: IoU 0,78, gaten op de verkeerde plek | 79,92 × 39,96 × 12,00, maar één gat als uitsparing: **gemarkeerd als onbetrouwbaar** |
| Donker onderdeel | Albedo 0,08 (zwart kunststof) | **Crash** (deling door nul) | **Duidelijke foutmelding**: het silhouet is maar half zichtbaar (V5) |
| Wit onderdeel | Albedo 0,95 | **Stil fout**: 12 spookgaten, IoU 0,78 | 79,93 × 39,88 × 11,99, maar twee spookgaatjes (Ø 1,2) en een uitsparing: **gemarkeerd als onbetrouwbaar**¹ |
| Handschaduw | Schaduw van hand en telefoon in de bovenaanzichten | **Stil fout** in de codereview: 87,5 × 64,1 × 10,8 | Goed: 79,92 × 39,91 × 12,02 |
| Verspreide bovenaanzichten | Tot 45 mm naast het onderdeel, 10° scheef | Goed, maar één gat Ø 6,46 | Goed: 79,89 × 39,92 × 12,05 |
| Weinig lage foto's | Alleen ringen op 60° en 75°, 5 bovenaanzichten | **Stil fout**: h 12,49, 79,29 × 39,62, geen gaten | 79,93 × 39,92 × 12,01, maar één gat als uitsparing: **gemarkeerd als onbetrouwbaar**¹ |
| Klein onderdeel | Plaatje 30 × 20 × 5, R2, Ø 4,5 | **Crash** (singuliere matrix) | Goed: 29,92 × 19,93 × 5,01, Ø 4,48 |
| Klein, weinig lage foto's | Idem, bovenaanzichten 6° scheef | **Crash** | Goed: 29,84 × 19,88 × 5,12, Ø 4,55. Afrondingen onzeker |
| Ring | Ø 25 × 8, gat Ø 8, weinig lage foto's | **Stil fout**: spookobject 56 × 154 mm, IoU 0,52 | Goed: Ø 24,88 × 8,08, gat Ø 7,99 |
| Zwaar | Schaduw, bolle mat, handschaduw en verspreide bovenaanzichten samen | **Stil fout**: h 11,89, geen gaten, IoU 0,93 | Goed: 79,91 × 39,94 × 12,05, 2 × Ø 6,58 (één extra korte rand) |
| OpenCV 4.12 | Basisscan met de oudere OpenCV | **Stil fout**: h 12,31; 79,80 × 39,82 | 12,02; 79,92 × 39,93 (hoekverschuiving gemeten en gecorrigeerd) |

¹ Gedraaid vóór de laatste reparatie van de gatherkenning. Die reparatie lostte hetzelfde probleem op in "realistisch" (daar nu beide gaten Ø 6,59); deze twee scenario's zijn daarna niet opnieuw gedraaid.

Wat opvalt in v0.2:

- **Systematische afwijking.** Lengtes en diameters komen systematisch ongeveer 0,1 mm te klein uit (0,05 mm per rand), hoogtes ongeveer 0,05 mm te groot. Dat is binnen het precisiedoel en zit nu in de U95.
- **Oorzaak.** De maskerrand ligt ~0,09 px naar binnen; de rest komt uit de fit. Oplossen via V2 en V13.

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
| — | Maskers rond het object | Drie reparaties, gevonden bij het testen van mat v2 en ook nuttig met v1: (1) de lokale versterking (schaduw) gebruikt alleen vensters waar niveau en contrast dezelfde versterking geven, en niet vlak naast bewijs voor het object; anders liet een gatrand die samenvalt met een vakrand het object ernaast als "beschaduwd wit" wegvallen. (2) De randstrook "zekere mat" voor de fit bevat alleen pixels die duidelijk op de mat lijken. (3) Randpixels zonder bruikbaar contrast volgen de meerderheid van hun buren. De maskerrand heeft daardoor geen bias meer die met het matpatroon meeverandert | `masks.py` |

## 4. Open verbeterpunten, op prioriteit

Impact en moeite: **H**oog, **M**iddel, **L**aag. Moeite S/M/L staat voor dagen, een week, of meerdere weken.

### 4.1 Eerst (Fase 0 en v0.4)

V5, V6 en V7 zijn in v0.3 gedaan, en voor V1 staat het gereedschap klaar (§3b). Hieronder staat per punt wat er nog open is.

| # | Verbetering | Waarom | Aanpak | Impact | Moeite |
|---|---|---|---|---|---|
| V1 | **Echte fotoset met schuifmaatmetingen** (gereedschap klaar in v0.3) | Alle drempels en U95 zijn nu op synthetische scans afgesteld | 10–20 onderdelen: metaal, zwart, wit, kunststof. Referentie met eindmaten en een ringkaliber, in de stijl van [ISO 10360-13 / VDI 2634](https://www.nist.gov/publications/vdivde-2634-2-and-iso-10360-13-performance-evaluation-tests-and-systematic-errors). Draaien als regressiesuite. **Open:** de set zelf maken volgens [FASE-0.md](FASE-0.md) en daarmee U95 en de drempels afstellen | H | M |
| V2 | **Fit op randafstanden in plaats van pixeltelling** | Pixeltelling is een trapfunctie: traag, geen covariantie, en afrondingen dwalen ±0,3 mm. De rasterizer is alleen exact voor randen langs de assen (−0,09 px bij 30°) | Residuen tussen geprojecteerde modelranden en de afstandstransformatie van elk masker (subpixel), met `scipy.optimize.least_squares` (soft-L1). Eventueel subpixelranden met [Devernay (IPOL)](https://www.ipol.im/pub/art/2017/216/) | H | L |
| V3 | **U95 die klopt** | U95 volgt nu alleen uit de resolutie | Covariantie uit V2, plus leave-one-out of bootstrap over de foto's. Termen voor printschaal (0,3 % · L als er geen meetlijn is opgegeven) en kalibratie-σ ([mrcal](https://mrcal.secretsauce.net/uncertainty.html)). Toetsen op 95 % dekking met V1 | H | M |
| V4 | **Topologie bijwerken na de fit** | De fit kan geen gaten of randen toevoegen of weghalen: een gemist gat blijft gemist | Clusters "zekere mat" binnen het bovenvlak (in ≥ 2 bovenaanzichten) worden een gat. Korte randen weg als de energie nauwelijks stijgt (BIC). Ronde uitsparingen worden gaten | H | M |
| V5 | ~~Mat v2~~ (gedaan in v0.3) | Een donker onderdeel op een zwart vak is onzichtbaar: in de stresstest werd maar ~50 % van het silhouet gezien | Gedaan: stippenraster in de zwarte vakken, eigen marker-ID's per formaat, Letter. **Open:** donkere stippen in de witte marges rond de markers (die zijn maar 2,5 mm breed en de markerdetectie heeft ze nodig), middengrijze vakken, en controleren of de stippen op gewone printers goed uitkomen (V1) | H | M |
| V6 | ~~Preflight bij het uploaden~~ (gedaan in v0.3) | Een slechte fotoset blijkt nu pas na de verwerking | Gedaan: zie §3b. **Open:** een live camerabeeld met dezelfde controle (V25) | H | M |

### 4.2 Robuustheid op echte foto's

| # | Verbetering | Aanpak | Impact | Moeite |
|---|---|---|---|---|
| V7 | ~~Anisotrope printschaal (sx ≠ sy)~~ (gedaan in v0.3) | Gedaan: meetlijnen X en Y, de schaal zit in de matgeometrie. De printschaalcontrole zit in `camtocad valideer` (alle lengtes procentueel te groot of te klein). **Open:** de meetlijnen zijn in de foto's zelf niet te meten, want ze schalen mee met de print; alleen een onafhankelijk object met bekende maat (eindmaat, bankpas 85,60 × 53,98 mm) kan de schaal controleren | H | S–M |
| V8 | Segmentatiecascade | GrabCut met de bestaande driedeling object/mat/onbekend. Kleur (afstand tot de zwart-witte mat) als extra objectbewijs. [PyMatting](https://github.com/pymatting/pymatting) (MIT) voor een subpixel-alfarand. Schaduwdetectie op regioniveau voor egale vlakken ([overzicht](https://arxiv.org/abs/1304.1233)) | H | M |
| V9 | Optioneel een geleerd masker, alleen met Apache-2.0-code en -gewichten | [SAM 2.1](https://github.com/facebookresearch/sam2), [HQ-SAM 2](https://github.com/SysCV/sam-hq) of [EfficientViT-SAM](https://huggingface.co/mit-han-lab/efficientvit-sam) via ONNX Runtime op de CPU, geprompt met een kader en punten uit het matmasker. Altijd combineren met het matresidu en met meerdere foto's: SAM faalt op spiegelend metaal en lage contrasten. De CPU-snelheid is nog niet gemeten | M–H | M |
| V10 | Foto's zoals telefoons ze maken | EXIF lezen: oriëntatie, lens, brandpunt, digitale zoom. HEIC via pillow-heif. Groeperen per camera of lens, en een cameramodel per toestel bewaren (voor kleine scans). Waarschuwen bij σ(f)/f > 0,3 % | H | M |
| V11 | Mat niet vlak | Een residukaart per hoek over alle foto's toont krul. Eventueel een bundelaanpassing met een laag-orde matoppervlak | M | M |
| V12 | Meer dan één ding op de mat, of het object deels ernaast | Waarschuwen, en het object kiezen dat in de bovenaanzichten steun heeft. Een liniaal of munt kan groter zijn dan het onderdeel. "Ligt deels naast de mat" als expliciete melding | M | S |
| V13 | Maskerrand-bias (~0,13 px naar binnen) en onscherpte per foto | Meten op de bekende randen van de mat. Daarmee de 50 %-regel en de kleinste herkenbare afronding per scan instellen, en onscherpe foto's markeren | M | M |
| V14 | Kwaliteitspoort verfijnen | Residuclusters per foto: een gemist gat, een extra uitstulping. Snappen beoordelen op het energieverschil in plaats van het IoU-verschil, want 0,5 mm fout verandert de IoU maar ~0,005 | M | S |

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
| V21 | Nog ~2× sneller | Carving per z-vlak met één homografie, undistort-maps één keer berekenen, threadpool over foto's (OpenCV geeft de GIL vrij), grote JPEG's verkleind decoderen, gatringen vectoriseren | M | S |
| V22 | Structuur | Stapfuncties met gecachte tussenresultaten (`--vanaf`). Alle drempels in één `Tuning`-dataclass, ook in `report.json`. Een `ScanResult`, dode code weg (fijne hull, `surface_points`). Maskers als uitsnede: nu ~3,6 GB bij 300 foto's | M | M |
| V23 | Installatie | Lockfile (uv of pixi). CadQuery 2.8 met cadquery-ocp 7.9.3.x: 8.0.1 breekt CadQuery. VTK-vrije OCP. `camtocad doctor`: één cv2-wheel, versies, hoekverschuiving. Paden met niet-ASCII-tekens op Windows (`imdecode`/`tofile`). Installer via [PyApp](https://github.com/ofek/pyapp) of pixi-pack | M | M |
| V24 | Server | HTTPS op het LAN, nodig voor een live camera in de browser ([getUserMedia](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)). Quota en opruimen, verwijderen, PDF atomisch schrijven, `--lan` expliciet, per bestand uploaden met hervatten, magic bytes controleren. Bij AGPL een bronlink in de pagina (§13) | M | S–M |
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

1. **Fase 0 met echte foto's.** Het gereedschap is klaar in v0.3: V1-harnas, V5 (mat v2), V6 (fotocontrole), V7 (printschaal). Nu de meetset zelf maken en meten volgens [FASE-0.md](FASE-0.md): meten is weten.
2. **v0.4:** V2 (fit op randen), V3 (U95) en V4 (topologie). Samen geven ze nauwkeurigheid en een eerlijke onzekerheid, afgesteld op de Fase 0-metingen.
3. **v0.5:** de objectklasse (V15–V19) en de segmentatiecascade (V8, V10).
4. **v0.6 en verder:** vrije vormen (V20), live begeleiding (V24–V25) en een installer (V23).

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
