# Route A — lokaal en open source (v0.9)

Dit is de uitvoering van profiel A uit [OPEN-SOURCE-LOKAAL.md](OPEN-SOURCE-LOKAAL.md): foto's van een onderdeel op een geprinte kalibratiemat gaan naar je eigen pc, die er een CAD-model van maakt. Er is geen cloud nodig en er zijn geen licentiekosten; alle afhankelijkheden hebben een permissieve open-sourcelicentie.

| | |
|---|---|
| **Status** | v0.9 — werkt end-to-end op synthetische scans en is gehard tegen storingen uit echte foto's (schaduw, weinig of scheve bovenaanzichten, OpenCV 4.x/5.x). Nieuw in v0.9: de randen voor de subpixelfit komen uit de grijswaarden (en kleur) zelf, niet meer uit de maskerrand met een vaste fractie; daardoor kloppen vooral de gaten van donkere en gekleurde onderdelen beter. En een verzonken gat (verzinking 90°, zoals voor een verzonken schroef) wordt herkend, gemodelleerd en gemaatvoerd. In v0.8: kleur telt mee als bewijs voor het object (een gekleurd onderdeel is ook te zien waar het even donker is als de mat), foto's zoals telefoons ze maken (HEIC, en foto's van een andere lens of met digitale zoom worden aan de EXIF herkend), een exacte modelrenderer voor de pixelfit, en een eerlijke U95 voor een gat of sleuf waar langs de rand weinig mat te zien is. In v0.7: een afgeschuinde of afgeronde bovenrand (rondom) en een rechte trede worden gemodelleerd en gemaatvoerd, niet alleen gemeld; de hoogte is dan de echte totale hoogte. In v0.6: sleuven en rechthoekige uitsparingen als echte, maatgevoerde features, en een melding als het onderdeel niet 2,5D is, in plaats van een stil fout model. In v0.5: de maten worden op subpixelniveau op de silhouetranden gefit, elke maat krijgt een eigen U95 (jackknife over de foto's, afgesteld op 95% dekking), en een gat dat de startcontour miste wordt alsnog toegevoegd. In v0.4.1, na de eerste echte fotoset: een controle of het onderdeel tussendoor is verplaatst, en maskers die zich per foto aanpassen aan onscherpte, posefout en glans. In v0.4: zwarte en witte onderdelen op mat v2, afrondingen en overbodige hoekpunten in de fit. Sinds v0.3: mat v2, fotocontrole vooraf, printschaal in X en Y, en gereedschap om scans met schuifmaatmetingen te vergelijken ([Fase 0](FASE-0.md)) |
| **Objectklasse** | 2,5D-onderdelen die plat op de mat liggen: extrusie van een contour (rechte randen met scherpe of afgeronde hoeken, of een cirkel) met doorgaande gaten, sleuven en rechthoekige uitsparingen; de bovenrand rondom afgeschuind (45°) of afgerond, of één rechte trede dwars over het onderdeel |
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

HEIC-foto's (het standaardformaat van iPhones) leest de server-installatie ook. Gebruik je alleen de opdrachtregel, installeer dan `pip install -e ".[heic]"` (pi-heif), of stel de iPhone in op "Meest compatibel" (Instellingen > Camera > Formaten) voor JPG.

Controleer de installatie met de demo. Die rendert een synthetische scan van een beugel en verwerkt die, zonder camera:

```bash
camtocad demo --uit camtocad-demo
```

## Gebruiken

1. **Mat printen.** `camtocad mat --formaat A4` (of `A3`, `Letter`) schrijft `kalibratiemat_A4.pdf`.
   - Print op 100% (werkelijke grootte), op mat papier, en leg de mat vlak op een stijve ondergrond.
   - Meet beide meetlijnen van 100,0 mm na: **X** onder de mat, **Y** links, en geef ze op, bijvoorbeeld `--meetlijn 99.6 99.8`, of in de velden op de telefoonpagina. De printschaal wordt dan per richting gecorrigeerd. Geef ze ook op als ze precies 100,0 mm zijn: zonder meting rekent de U95 met een onbekende printschaal (0,3%, ±0,5 mm op 80 mm).
   - Dit is mat v2: in elk zwart vak een raster witte stippen, zodat ook een donker onderdeel op een zwart vak te zien is. Een eerder geprinte mat (v1) werkt nog en wordt automatisch herkend, net als het formaat.
2. **Foto's maken.** Leg het onderdeel plat in het midden van de mat, bij diffuus licht. Maak 30–60 foto's met de gewone camera-app van je telefoon (JPG of HEIC, in kleur):
   - **raak het onderdeel niet aan tot de laatste foto**: één scan is één vaste ligging. Wil je de mat draaien, draai dan mat en onderdeel samen. Een andere kant (omgedraaid, op zijn kant) is een aparte scan;
   - rondom, op twee à drie hoogtes (ongeveer 35°, 55° en 70° boven de mat), op zo'n 25–35 cm afstand met de hele mat in beeld;
   - **4–6 foto's recht van boven**: nodig voor de contour en om door gaten heen te kijken;
   - de mat steeds grotendeels in beeld, niet inzoomen, zelfde lens (geen groothoek/tele wisselen). Een iPhone schakelt dichtbij vanzelf naar de macrolens: blijf op 25–35 cm, of zet Macrobesturing aan en de macrostand uit. Foto's van een andere lens of met digitale zoom worden herkend (EXIF) en niet gebruikt.
3. **Controleren en verwerken**, op één van twee manieren:
   - **Via de browser van je telefoon:** start `camtocad server` op de pc en open op je telefoon (zelfde wifi) de link die in de terminal verschijnt, inclusief `?token=…`. Kies of maak daar de foto's.
     - Elke foto wordt direct gecontroleerd: is de mat gevonden, is de foto scherp en goed belicht.
     - Een dekkingskaart toont in het rood welke richtingen nog ontbreken, met aanwijzingen zoals "nog 2 foto's recht boven het onderdeel".
     - Slechte foto's verwijder je met ×. Ontbrekende foto's voeg je toe, ook later aan een mislukte scan.
     - Daarna tik je op **Verwerken**. Rapport, STEP, STL en script staan daarna klaar om te downloaden, bij een fout met de debugbeelden erbij.
   - **Via de opdrachtregel:** kopieer de foto's naar een map. `camtocad controleer <map>` beoordeelt de set in een paar seconden. `camtocad scan <map>` verwerkt hem; de uitvoer komt in `<map>_cad/`.

Opties voor `scan`:

- `--mat`: `A4`, `A3`, `Letter`, `A4-v1` of `A3-v1`. Standaard wordt de mat herkend aan de markers.
- `--meetlijn X [Y]`: gemeten lengte van de 100 mm-lijnen. Weglaten betekent: niet gemeten.
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
  - Een klein duwtje (een paar millimeter) valt hier niet op. Dat zie je aan "betrouwbaarheid: laag" in het rapport: het model past dan in een deel van de foto's slecht (in de test met 10 mm verschuiving: IoU 0,69 in de slechtste foto).
- **"gekozen mat A4, maar de foto's tonen mat A3"**: de mat op de foto's is gebruikt. Controleer of dat de mat is die je bedoelde.
- **"betrouwbaarheid: laag"** in het rapport. Het model is gemaakt, maar iets klopt niet; de reden staat erbij. Bijvoorbeeld een uitsparing die geen gat, sleuf of rechthoek is, of een model dat in sommige foto's slecht past. Controleer die maten.
- **"trede gemodelleerd"** of **"bovenrand gemodelleerd"** in de log: het onderdeel paste niet in één prisma, en een trede of een afschuining of afronding van de bovenrand rondom paste duidelijk beter (de energie daalde minstens 5%). Het model heeft die vorm dan ook: het rapport geeft de totale hoogte, de maat van de afschuining of afronding, of de plaats en hoogte van de trede, en het bovenaanzicht tekent ze gestippeld. Controleer die maten met een schuifmaat, vooral de soort (afschuining of afronding): die twee verschillen in de silhouetten maar weinig.
- **"geen 2,5D-vorm?"** in het rapport: het onderdeel past ook met een trede of een afschuining niet in het model. "Langs de rand van (60, 0) tot (60, 40) ligt de bovenkant lager" is een trede, kamer of plaatselijke afschuining op die plek die niet als één rechte trede te modelleren was (werkcoördinaten, zoals in het rapport). "In de lage foto's steekt de bovenrand rondom iets boven het model uit": een afschuining of afronding van de hele bovenrand die niet te modelleren was; de gemeten hoogte is dan die van de onderkant ervan. "Ligt de bovenrand rondom binnen het model": tapse wanden of een ronde vorm (een liggende cilinder). Meet de hoogtes na met een schuifmaat.
- **"gat Ø … is een uitsparing"** of **"korte sleuf … is een gat"** in de log: de startcontour zag een opening anders dan hij is (door parallax lijkt een uitsparing in de bovenaanzichten ronder, en een schuin gezien gat langwerpig). De fit heeft beide vormen geprobeerd en houdt de vorm die het best bij de silhouetten past.
- **"randfit niet gebruikt"** in de log: de subpixelfit wilde meer dan 0,5 mm van de pixelfit afwijken, meestal omdat de contour rommelig is. De maten komen dan uit de pixelfit, en U95 is de grove schatting uit resolutie en aantal foto's; het rapport noemt de methode onder de maattabel. Dit gaat meestal samen met "betrouwbaarheid: laag".
- **"gat toegevoegd"** in de log: een gat ontbrak in de startcontour, maar in de foto's recht van boven is daar door het bovenvlak heen mat te zien, en het model past met het gat duidelijk beter. Controleer het gat als het onderdeel daar geen gat hoort te hebben.
- **Foto's "niet gebruikt (afwijkend formaat)"**: andere lens, zoom of bijgesneden. Staand opgeslagen foto's worden automatisch teruggedraaid.
- **"foto('s) van een andere camera, lens of zoom niet gebruikt"**: volgens de EXIF-gegevens is een deel van de foto's met een andere lens of met digitale zoom gemaakt, vaak de macrolens van een iPhone die dichtbij vanzelf inschakelt, met hetzelfde beeldformaat. Die passen niet in één cameramodel. De melding noemt de lens van die foto's en die van de rest.
- **"brandpuntsafstand onzeker (σ … %)"**: de camera is uit te weinig verschillende hoeken gekalibreerd. Maak foto's van meer hoeken en hoogtes, met de mat steeds grotendeels in beeld.
- **"HEIC-foto's … alleen te lezen met de extra 'heic'"**: installeer `pip install -e ".[heic]"`, of zet de foto's om naar JPG.
- **"gat 2: te weinig bewijs rond de rand"**: langs de rand van dat gat is (bijna) nergens mat te zien, meestal een zwart onderdeel met het gat boven een zwart vak. Maat en plaats komen dan uit de pixelfit, met een ruime U95 (in de stresstests ±0,6 mm), en worden niet gesnapt. Meet het gat na, of scan opnieuw met het onderdeel anders op de mat. `diagnose.json` (`bewijs binnenvormen`) laat per gat en sleuf zien welk deel van de rand bewijs heeft.
- **"foto's onscherp (σ > 1,8 px)"**: bewogen of niet scherp gesteld. Maak ze opnieuw; `camtocad controleer` laat per foto de onscherpte zien.

## Hoe het werkt

| Stap | Module | Wat er gebeurt | Architectuur |
|---|---|---|---|
| Mat | `mat.py`, `pdf.py` | ChArUco-mat als vector-PDF op exacte schaal, met meetlijnen X en Y en een stippenraster in de zwarte vakken. Eigen marker-ID's per formaat: de mat wordt herkend | §4.5 |
| Fotocontrole | `preflight.py` | Per foto: mat, onscherpte (gemeten aan de matranden), belichting, kijkhoek. Per scan: waar ligt het object, dekking per richting en hoogte, aanwijzingen | — |
| Foto's | `imgio.py` | JPG, PNG, TIFF en HEIC (met pi-heif). Uit de EXIF: toestel, lens, brandpuntsafstand en digitale zoom; foto's van een andere lens of met zoom gaan niet in het cameramodel. De kleur wordt apart bewaard, op halve resolutie | — |
| Camera | `calib.py` | Mat herkennen, camera zelf kalibreren (Zhang), per foto een metrische pose: de mat is de tracker. De printschaal (X en Y) zit in de matgeometrie. De hoekverschuiving van de geïnstalleerde OpenCV-versie wordt gemeten en gecorrigeerd; staand opgeslagen foto's worden teruggedraaid. De onzekerheid van de brandpuntsafstand wordt gemeld als die boven 0,3% komt | §4.5, OPEN-SOURCE-LOKAAL §2.1 |
| Maskers | `masks.py` | Voorspel per foto hoe de mat eruitziet; afwijkingen zijn object. "Zekere mat" alleen waar het lokale patroon de mat herhaalt (correlatie), zodat donkere en witte onderdelen niet wegvallen en schaduwen mat blijven. Randpixels volgens de 50%-regel. Waar object en mat even donker of licht zijn (zwart op zwart), is een pixel geen bewijs: binnen het object wordt zo'n stuk opgevuld, de fit negeert het. Per foto aangepast aan onscherpte (de voorspelling even vaag), posefout (gemeten aan de matranden) en glans (zwart lichter dan verwacht). Kleur is een derde soort bewijs: de mat is zwart-wit, dus wat na de witbalans (gemeten op de mat) gekleurd is, is object; de rand ligt op het halve kleurcontrast. Rond het object ook een zachte objectfractie (alpha) uit grijs en kleur, voor subpixelranden in de randfit | §5.3 [R3c] |
| Ligging | `placement.py` | Ligt het onderdeel in alle foto's op dezelfde plek? Per voxel (3 mm) hoeveel foto's hem op het object zien en hoeveel op de mat; per foto of de andere foto's zijn objectpixels steunen en of hij de gezamenlijke hull niet op de mat ziet. Foto's met verschillende liggingen vallen zo in groepen uiteen | — |
| Lokaliseren | `hull.py` | Grove visual hull (2 mm): waar ligt het object, en een bovengrens voor de hoogte | §5.3 [R3c] |
| Startmodel | `initial.py`, `profile.py` | Hoogte zoeken waarop de bovenaanzichten samenvallen, terugprojecteren → contour → randen, afrondingen, gaten; met terugvalopties en een uitgelegde fout | §6.3–6.5 |
| Model fitten | `silhouette.py`, `pipeline.py` | Analysis-by-synthesis: model-silhouet renderen in élke foto (exact: een pixel hoort erbij als zijn midden binnen de projectie ligt), pixelverschil minimaliseren (hoogte, randen, afrondingen, gaten). Daarna per hoek grote stappen in de afronding proberen, en hoekpunten weghalen waar het model zonder even goed past (een knik of een korte schuine rand zonder bewijs) | §6.8, §7.3 |
| Gaten, sleuven, uitsparingen | `profile.py`, `holes.py`, `pipeline.py` | Een opening in de startcontour wordt een gat, een sleuf of een rechthoek met afgeronde hoeken, of blijft een polygoon. Na de pixelfit worden de alternatieven getoetst: gat of uitsparing, korte sleuf of gat, polygoon of sleuf; de vorm die het best bij de silhouetten past, blijft. Waar de foto's recht van boven door het bovenvlak heen mat zien, ontbreekt een gat: het wordt toegevoegd als het model er duidelijk beter door past | §7.3 |
| Randfit | `edgefit.py` | Kleinste kwadraten op de subpixelafstand tussen de geprojecteerde modelranden en de rand in de foto (uit alpha, anders de maskerrand), robuust tegen uitschieters, binnen ±0,5 mm van de pixelfit. Per foto en per punt van de contour de rand die het silhouet vormt: de onderrand als de wand naar de camera kijkt, anders de bovenrand, of bij een afschuining de schouder eronder. Bij een trede ook de verticale randen ervan. Een gat of sleuf zonder bewijs rondom (zwart op zwart) blijft staan. De bovenrand van een verzinking telt als rand binnen het object: gemeten in de grijswaarden | §6.8, §7.3 |
| Verzinkingen | `countersink.py`, `pipeline.py` | Een ring rond een gat in de foto's van boven (rondom, in de meeste foto's op dezelfde afstand) wordt een verzinking van 90°; daarna de randfit opnieuw, met die ring als rand binnen het object | — |
| Vorm | `prismcheck.py`, `pipeline.py` | Past een prisma wel? Een stuk bovenrand dat in alle foto's lager of hoger ligt dan het model, of een bovenrand die in de lage foto's boven het model uitsteekt. Zo ja, dan wordt een rechte trede (door de uiteinden van dat stuk) of een afschuining of afronding van de bovenrand rondom als model geprobeerd, en ook als een goedkope proef laat zien dat een afschuining zonder fit al beter past (v0.8: bij een zwart onderdeel, of zonder foto's onder 45°); die vorm blijft als het model er duidelijk beter door past. Anders volgt een melding in plaats van een stil compromis | §7.3 |
| Onzekerheid | `uncertainty.py` | Per maat: de spreiding als steeds een zesde van de foto's wegvalt (jackknife), plus een systematisch deel per soort maat en de printschaal. Afgesteld op 95% dekking op gerenderde scans. Een gat of sleuf met bewijs aan maar een deel van de rand krijgt een groter systematisch deel (tot 8×) | §6.1, §8 |
| Ontwerpintentie | `snapping.py`, `cadmodel.py` | Werkassenstelsel met datum, randen exact haaks, Bayesiaans snappen (hele mm, ISO 273, tapboormaten, standaardstralen, verzinkingen volgens DIN 74-1), steekcirkels | §6.6 |
| CAD | `cadmodel.py`, `cadhelpers.py` | OpenCascade-solid via CadQuery (een afschuining als getaperde extrusie, een afronding als fillet, een trede als weggesneden blok, een verzinking als kegel), STEP/STL-export, leesbaar script met benoemde maten | §6.9 |
| Kwaliteit | `pipeline.py`, `debug.py` | Kwaliteitspoort (past het model bij de foto's?) en diagnosebeelden in `debug/` | §4.6 |
| Rapport | `report.py` | Maten met U95 en snapreden, betrouwbaarheid, bovenaanzicht, samenvatting, waarschuwingen; de geometrie staat ook in `report.json` | §4.6 |
| Validatie | `validate.py` | Scans vergelijken met schuifmaatmetingen (`maten.json`): fout per maat, binnen U95 ja/nee, bias per soort, printschaalcontrole | §8 |

**Waarom silhouetten en geen MVS?** Voor deze objectklasse zijn silhouetten nauwkeuriger en robuuster: ze hebben geen textuur op het object nodig en zijn ongevoelig voor glans. Een visual hull alléén is niet genoeg: van bovenaf is het bovenvlak niet te zien, en de wanden worden te ruim. De maten komen daarom uit het fitten van het parametrische model op de randen in alle foto's — de "model-gebaseerde verfijning" uit het architectuurdocument. MVS (COLMAP/OpenMVS) volgt voor vrije vormen.

## Resultaten op synthetische scans

Gerenderde scans van 46 foto's (1600 × 1200 pixels, ~0,25 mm/pixel op het object), met lensvervorming, ruis en belichtingsverloop. De camera en de poses worden uit de foto's zelf bepaald. De beugel met sleuf en uitsparing en de drie beugels die geen prisma zijn (afschuining, afronding, trede) komen uit de stresstests (scenario met ruis, JPEG en vignettering).

| Onderdeel | Maat | Waarheid | Gefit | Na snappen |
|---|---|---|---|---|
| Beugel (op 17° gedraaid) | lengte × breedte × hoogte | 80 × 40 × 12 | 79,97 × 40,02 × 12,01 | 80 × 40 × 12 |
| | afrondingen (4x) | R3 | R3,04 | R3 |
| | gaten (2x) | Ø 6,6 op (10, 20) en (70, 20) | Ø 6,59 op (9,95, 19,98) en (70,01, 20,02) | Ø 6,59 (niet gesnapt: 6,5 en 6,6 beide plausibel); posities exact |
| L-vorm (niet-convex) | maten | 60 / 30 / 25 / 50, hoogte 8 | 60,00 / 29,99 / 24,95 / 49,99, hoogte 8,03 | exact |
| | afrondingen | 4 × R2, 2 scherpe hoeken | 4 × R2,17; de scherpe hoeken blijven scherp | R2,17 (niet gesnapt, binnen U95) |
| | gat | Ø 5,5 op (15, 12) | Ø 5,49 op (15,00, 11,98) | Ø 5,5 (ISO 273 M5) op (15, 12) |
| Beugel met sleuf en uitsparing | lengte × breedte × hoogte | 80 × 40 × 12 | 79,98 × 40,01 × 12,00 | 80 × 40 × 12 |
| | sleuf | 6,6 breed, hartafstand 16, op (48, 20) | 6,62, hartafstand 15,92, op (48,00, 20,01) | 6,62 (niet gesnapt: 6,5 en 6,6 beide plausibel), hartafstand 16, op (48, 20) |
| | uitsparing | 10 × 8, R1,5, op (70, 20) | 10,01 × 8,00, R1,57, op (69,99, 20,04) | 10 × 8, R1,57 (niet gesnapt), op (70, 20) |
| Beugel met afschuining rondom | hoogte, afschuining | 12, 1,5 × 45° | 12,05, 1,59 | 12, 1,5 |
| | lengte × breedte, gaten | 80 × 40, 2 × Ø 6,6 op (10, 20) en (70, 20) | 79,98 × 40,00, Ø 6,61 op (9,93, 19,96) en (70,00, 20,03) | 80 × 40, Ø 6,61 (niet gesnapt), posities exact |
| Beugel met afronding rondom | hoogte, afronding | 12, R2 | 12,08, R2,11 | 12, R2 |
| Beugel met trede | hoogte; trede (de rechter 20 mm) | 12; op x = 60, 6 hoog | 12,00; op 59,97, 6,02 hoog | 12; op 60, 6 hoog |
| Ronde flens | diameter × hoogte | Ø 50 × 10 | Ø 49,96 × 10,03 | Ø 50 × 10 |
| | gatenpatroon | 4 × Ø 4,5 op steekcirkel Ø 35 | 4 × Ø 4,43 op Ø 34,98 | 4 × Ø 4,43 (niet gesnapt, binnen U95) op Ø 35, parametrisch patroon |

Het rapport geeft bij elke maat een U95. Bij deze scans zijn geen meetlijnen opgegeven, dus telt 0,3% printschaal mee; bij lange maten overheerst die (±0,48 mm op 80 mm). Zonder die term is U95 ongeveer ±0,05 mm voor de hoogte, ±0,06 mm voor lengtes, ±0,08 mm voor gatposities, ±0,16 mm voor gaten en ±0,41 mm voor afrondingen. Met gemeten meetlijnen telt de printschaal voor 0,05%: een lengte van 80 mm krijgt dan ±0,10 mm.

Over alle stresstests die geen waarschuwing geven (ook een zwart, een wit en gekleurde onderdelen, zie [ROUTE-A-VERBETERPUNTEN.md §3e–§3i](ROUTE-A-VERBETERPUNTEN.md#3i-opgelost-in-v09-v2-open-randen-uit-de-grijswaarden-v16-verzinkingen)), in v0.9:

- prisma's: buitenmaten −0,05 tot +0,02 mm, hoogtes −0,03 tot +0,02 mm (v0.8: ±0,05 en −0,04 tot +0,03);
- gaten −0,05 tot +0,02 mm (v0.8: −0,07 tot 0,00). Een gat van een zwart onderdeel boven een groot zwart markervlak heeft maar 1% bewijs langs de rand: het wordt gemeld, krijgt een ruime U95 (±1,2 mm; de fout is −0,23 mm) en wordt niet gesnapt;
- afrondingen −0,11 tot +0,10 mm; scherpe hoeken blijven scherp;
- elke maat binnen zijn eigen U95 (119 van 119), en van de gesnapte maten is er geen fout gesnapt;
- sleuven en uitsparingen in zes scenario's: buitenmaten −0,02 tot +0,02 mm, sleufbreedte 0,00 tot +0,03 mm, hartafstand −0,07 tot +0,08 mm, uitsparing −0,01 tot +0,04 mm in lengte en breedte; alle 96 maten binnen U95, geen foute snap;
- afschuining, afronding en trede: hoogte −0,07 tot +0,06 mm, maat van de afschuining of afronding −0,06 tot +0,08 mm, trede op 0,03 mm en in hoogte op 0,01 mm; ook bij het zwarte onderdeel en zonder foto's onder 45°; alle 47 maten binnen U95. De soort (afschuining of afronding) was steeds goed. Een afronding van de bovenrand trekt de buitenmaten iets mee (tot −0,10 mm);
- gekleurde onderdelen: buitenmaten −0,04 tot 0,00 mm, gaten −0,07 tot −0,01 mm (v0.8: tot +0,06 en tot −0,18); 52 van 53 maten binnen U95 (de misser: een gatpositie van de donkerblauwe beugel onder warm licht, +0,085 bij U95 ±0,083), geen foute snap;
- verzonken gaten (v0.9): de verzinking Ø 12,4 × 90° (M6) komt uit op 12,38, gesnapt naar 12,4; gat en posities binnen 0,02 mm. Ook bij een zwart onderdeel herkend (Ø 12,395); daar hangt de U95 (±0,43) af van het bewijs rond de gaten, en een gat boven een zwart vak houdt zijn verzinking zoals herkend.

Dat is binnen het Precisie-doel van ±(0,2 mm + 0,1% · L). **Let op:** dit zijn synthetische scans met mat v2. Echte foto's hebben schaduwen, autofocus, compressie en een niet perfect vlakke mat. Hoe dicht v0.9 daarbij in de buurt komt, moet Fase 0 uitwijzen ([FASE-0.md](FASE-0.md)).

De eerste echte fotoset (een zwarte accu op mat v1) is niet gelukt: het onderdeel lag in minstens vier verschillende liggingen, en een zwart onderdeel op mat v1 is op de zwarte vakken onzichtbaar. Wat dat opleverde staat in [ROUTE-A-VERBETERPUNTEN.md §2c](ROUTE-A-VERBETERPUNTEN.md#2c-de-eerste-echte-fotoset-september-2026).

## Beperkingen van v0.9

- **Objectklasse:** 2,5D-onderdelen plat op de mat, met doorgaande gaten (ook verzonken, 90°), sleuven en rechthoekige uitsparingen, een afgeschuinde (45°) of afgeronde bovenrand rondom, of één rechte trede dwars over het onderdeel. Geen blinde gaten, kamerboringen, kamers, meer treden, een afschuining langs één rand, schroefdraad of vrije vormen; zo'n vorm wordt gemeld ("geen 2,5D-vorm?"), niet gemodelleerd. Zonder foto's onder 45° is een afschuining rondom slecht te zien; sinds v0.8 vindt een proef hem in de stresstest toch, maar maak altijd ook lage foto's. Afschuining of afronding: de silhouetten verschillen weinig, controleer de soort. Afschuiningen op verticale hoeken worden als afronding benaderd; buitencontouren met bogen groter dan een hoekafronding (bijv. een halfronde kop) worden met rechte randen benaderd. Een uitsparing die geen sleuf of rechthoek is (bijv. een L-vormige opening), blijft een polygoon en wordt gemarkeerd.
- **Bovenaanzichten zijn verplicht**: zonder foto's recht van boven stopt de verwerking met een duidelijke melding.
- **Eén ligging per scan.** Een verplaatst of omgedraaid onderdeel wordt herkend en gemeld; een klein duwtje van een paar millimeter niet (zie "Als het niet lukt").
- **Belichting:**
  - Schaduwen op de gestructureerde delen van de mat worden herkend.
  - Een harde slagschaduw over een egaal vak kan nog als object meetellen. De kwaliteitspoort markeert het resultaat dan als onbetrouwbaar (in de stresstests: altijd).
  - Gebruik diffuus licht, en mat papier voor de mat.
- **Zwarte en witte onderdelen:** op mat v2 lukken ze in de stresstests. Waar het onderdeel even donker (of licht) is als de mat, telt de foto niet mee; binnen het onderdeel wordt zo'n stuk opgevuld, en de fit gebruikt alleen echt bewijs. Het zwakste punt is een gat boven een groot zwart vlak van een marker: langs de rand is daar (bijna) nergens mat te zien. Sinds v0.8 herkent de scan zo'n gat: het rapport waarschuwt, maat en plaats komen uit de pixelfit met een ruime U95 (bij ~0,25 mm/pixel ±0,6 mm voor de plaats en ±1,2 mm voor de diameter), en ze worden niet gesnapt. De rest van het onderdeel wordt gewoon subpixel gefit. Op een mat v1 (zonder stippen) mislukt een zwart onderdeel nog; het resultaat wordt dan gemarkeerd. Print mat v2.
- **Kleur** telt mee als de foto's in kleur zijn en het onderdeel duidelijk gekleurd is (v0.8): een donkerblauw of donkerrood onderdeel is dan overal te zien, ook boven de zwarte vakken, en heeft geen gaten zonder bewijs. Grijs, zwart, wit en blank metaal krijgen er niets bij. Kleur telt als bewijs voor de mat alleen direct buiten de buitenrand, nog niet in gaten (glans op het onderdeel is ook kleurloos). Sinds v0.9 komen de randen voor de randfit uit de grijswaarden en de kleur zelf; zie de resultaten hierboven. Een echte camera is niet lineair (toonkromme, verscherping): hoeveel dat de randen verschuift, moet Fase 0 uitwijzen.
- **Onzekerheid (U95)** per maat: afgesteld op gerenderde scans, waar zonder waarschuwing op één na alle fouten erbinnen vallen (v0.9: 345 van 346; de misser is een gatpositie van een donkerblauw onderdeel onder warm licht, 0,002 mm buiten U95). Een gat of sleuf met weinig bewijs langs de rand krijgt een ruimere U95. Op echte foto's kan de systematiek groter zijn; meet het zelf met `camtocad valideer` ([FASE-0.md](FASE-0.md)). Zonder opgegeven meetlijnen telt 0,3% printschaal mee (±0,5 mm op 80 mm): meet de meetlijnen voor een kleinere U95, en geef ze ook op als ze precies 100,0 mm zijn.
- **Verzinkingen** (v0.9): alleen een kegel van 90° (verzonken schroeven), herkend aan de ring in de foto's van boven (≥ 60°). Een kamerboring (cilinderkopschroef), een afschuining van een gatrand kleiner dan ~0,5 mm of een andere tophoek nog niet. Een verzinking verandert het silhouet nauwelijks; zonder foto's van boven wordt ze dus niet gevonden.
- **Afrondingen** zijn het minst nauwkeurig (tot ±0,4 mm, U95 ~0,4 mm): ze bepalen maar een klein stukje van het silhouet. Gelijke afrondingen worden gegroepeerd en alleen gesnapt als dat zeker is.
- **Afrondingen kleiner dan ~4,5 pixels** (bij de demo ~1,1 mm) zijn niet te onderscheiden van scherpe hoeken en worden als scherp gemodelleerd; het rapport meldt dat.
- **Rekentijd:** ~60–100 s per scan van 46 foto's (1600 × 1200) op een gewone CPU met 4 kernen (de beugel uit de stresstests: 100 s), waarvan ~4 s voor de controle op verplaatsing, ~5–20 s voor de randfit en de onzekerheid per maat, en ~5 s per gat om te toetsen of het geen uitsparing is; foto's worden standaard teruggeschaald naar 2000 pixels. Kleur kost ~10–15% extra. Een afschuining, afronding of trede modelleren kost ~40 s extra. Als er veel overbodige hoekpunten uit de contour moeten (zwart onderdeel, schaduw) duurt het langer: 1,5 minuut voor de zware stresstest, en ~7 minuten voor het zwarte onderdeel met afschuining. De fotocontrole kost ~0,1–0,3 s per foto.
- **Nog geen native app:** de telefoon gebruikt de browser. De geleide AR-opname uit het architectuurdocument volgt met de Android-app.

## Licenties

Alle gebruikte bibliotheken zijn open source met een permissieve licentie: NumPy en SciPy (BSD), OpenCV (Apache-2.0), CadQuery (Apache-2.0) op OpenCascade (LGPL-2.1 met uitzondering), Pillow (MIT-CMU), en FastAPI/Uvicorn (MIT/BSD). Voor HEIC: pi-heif (BSD-3) met libheif en libde265 (LGPL-3.0, dynamisch gelinkt; geen x265, dat GPL is). Welke licentie Cam-to-CAD zelf krijgt, is nog een open beslissing; OPEN-SOURCE-LOKAAL.md §1 adviseert AGPL-3.0-or-later.

## Volgende stappen

Een uitgebreid, geprioriteerd overzicht staat in [ROUTE-A-VERBETERPUNTEN.md](ROUTE-A-VERBETERPUNTEN.md). Het is gebaseerd op stresstests, een codereview en onderzoek naar de stand van de techniek. Kort:

1. **Fase 0 met echte foto's:** 10–20 onderdelen met bekende maten scannen, afwijkingen en de dekking van U95 meten, de systematische termen en drempels bijstellen (debugbeelden in `resultaat/debug/`).
2. **Randen uit de grijswaarden** (subpixel, in plaats van de maskerrand) en uit beeldgradiënten binnen het object: gatranden, treden en afschuiningen die in het silhouet niet te zien zijn.
3. **Uitbreiding van de objectklasse:** meer treden en kamers (meerdere hoogteniveaus), afschuiningen langs één rand en van gatranden; gatenpatronen op rechthoekige delen.
4. **Vrije vormen:** COLMAP + OpenMVS met de matposes, en `camtocad cad` voor puntenwolken.
5. **Android-app** (GPL, zonder Google Play Services) met geleide opname: dekkingskoepel, automatisch afdrukken, stills op volle resolutie.
6. **FreeCAD-export** (`.FCStd` met PartDesign-feature tree) naast het CadQuery-script.
