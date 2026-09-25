# Cam-to-CAD — Technisch en functioneel architectuurdocument

| | |
|---|---|
| **Versie** | 0.1 — concept ter review |
| **Datum** | 25 september 2026 |
| **Platform** | Android-app + cloudverwerking (iOS in een latere fase) |
| **Status** | Voorstel. Aannames en doelwaarden worden gevalideerd in Fase 0 (§9). |

> **Kernboodschap.** Een smartphonescan omzetten in bruikbare CAD is geen conversieprobleem (*mesh → STEP*), maar een **meet- en interpretatieprobleem**. De camera levert metingen mét onzekerheid. De pipeline zoekt het eenvoudigste ontwerp — vlakken, cilinders, extrusies, revoluties, afrondingen, patronen — dat die metingen binnen die onzekerheid verklaart. Elke architectuurkeuze in dit document volgt uit dat principe.

**Leeswijzer.** De vier hoofdvragen komen aan bod in §4 (functionaliteit en UX, inclusief schaalkalibratie), §5 (technische stack en reconstructie-engine), §6 (de mesh-to-parametric pipeline) en §7 (edge cases). §1–3 bevatten samenvatting, scope en systeemoverzicht; §8–10 validatie, roadmap en bijlagen.

## Inhoud

1. [Managementsamenvatting](#1-managementsamenvatting)
2. [Scope, uitgangspunten en kwaliteitsdoelen](#2-scope-uitgangspunten-en-kwaliteitsdoelen)
3. [Systeemoverzicht](#3-systeemoverzicht)
4. [Functionaliteit en user experience](#4-functionaliteit-en-user-experience)
5. [Technische stack](#5-technische-stack)
6. [De mesh-to-parametric pipeline](#6-de-mesh-to-parametric-pipeline)
7. [Edge cases en technische knelpunten](#7-edge-cases-en-technische-knelpunten)
8. [Validatie, metrics en teststrategie](#8-validatie-metrics-en-teststrategie)
9. [Roadmap, team en open beslissingen](#9-roadmap-team-en-open-beslissingen)
10. [Bijlagen](#10-bijlagen)

Aanvullend: [Haalbaarheidsanalyse — volledig open source en zonder cloudkosten](OPEN-SOURCE-LOKAAL.md)

---

## 1. Managementsamenvatting

Met Cam-to-CAD scant een gebruiker met een Android-smartphone een fysiek onderdeel en krijgt een bewerkbaar CAD-model terug: een schone B-rep in STEP AP242 **én** een parametrische feature tree. De architectuur bestaat uit vier lagen:

1. **Geleide opname op het toestel.** ARCore zorgt voor metrische tracking, Camera2 levert stills op volle resolutie, en een real-time kwaliteitsengine leidt de gebruiker met AR-overlays naar een volledige, scherpe en goed belichte dataset.
2. **Metrische reconstructie in de cloud.** Marker-ondersteunde fotogrammetrie met schaalfusie vormt de nauwkeurige ruggengraat. Oppervlakte-Gaussian-Splatting vult aan voor volledigheid, feed-forward 3D-modellen voor robuustheid.
3. **Mesh-to-CAD.** Een hybride, neuro-symbolische pipeline: neurale netwerken stellen segmentatie en featurestructuur voor, robuuste geometrie en een CAD-kernel meten, bouwen en verifiëren.
4. **Review en integratie.** De gebruiker ziet maten met onzekerheid, corrigeert gericht, en exporteert naar STEP/IGES, code-CAD en CAD-platformen.

### 1.1 Belangrijkste architectuurbeslissingen

| # | Onderwerp | Beslissing | Kernargument |
|---|---|---|---|
| 1 | Output | Twee leveringen: een schone B-rep in **STEP AP242** en een **parametrische feature tree** (CadQuery/build123d-code, FreeCAD, API-koppelingen) | STEP en IGES bevatten geen constructiehistorie (§2.4) |
| 2 | Absolute maat | Gecodeerde **kalibratiemat** en referentiematen als gewogen constraints in de bundle adjustment | De schaal uit ARCore alleen wijkt typisch 1–3% af |
| 3 | Mobiel platform | **Native Android** (Kotlin + C++/NDK), ARCore + Camera2 via de *Shared Camera* API, met een opnamestrategie per toestel | Volledige controle over focus, stabilisatie, RAW en tijdstempels (§5.1) |
| 4 | Verwerkingslocatie | **Cloud-GPU**, of dezelfde containers **lokaal** (desktop of eigen server) zonder cloudkosten; op het toestel alleen kwaliteitsbewaking en preview | Rekenlast, modelgrootte, herverwerkbaarheid; keuzevrijheid in kosten (§5.5) |
| 5 | Reconstructie | **Fotogrammetrie als metrische ruggengraat**, aangevuld met oppervlakte-Gaussian-Splatting (2DGS/PGSR-principes) en feed-forward 3D-modellen | Nauwkeurigheid én volledigheid; NeRF is hier geen eerste keus (§5.3) |
| 6 | Mesh-to-CAD | **Hybride neuro-symbolisch**; de keuze tussen hypothesen verloopt via *analysis-by-synthesis* | Betrouwbaarheid én echte parametriek |
| 7 | CAD-kernel | **Open CASCADE Technology (OCCT)** via CadQuery/build123d; Parasolid als latere optie | De enige volwassen open-source B-rep-kernel |
| 8 | Orkestratie | **Temporal**-workflows op **Kubernetes** met GPU-autoscaling in een EU-regio | Lange, herstartbare jobs met human-in-the-loop |
| 9 | Betrouwbaarheid | Onzekerheid per maat altijd zichtbaar; **graceful degradation** over vier outputniveaus | Engineers moeten weten welke maten ze kunnen vertrouwen |
| 10 | Licenties | Alleen componenten met commercieel bruikbare code **én** gewichten; eigen modellen trainen waar nodig | Veel toonaangevende modellen zijn alleen niet-commercieel beschikbaar (§5.4) |

### 1.2 Haalbaarheid in één oogopslag (stand 2025/2026)

| Capability | Status |
|---|---|
| Metrisch nauwkeurige reconstructie van matte, getextureerde objecten | **Productierijp** |
| Robuuste camerabepaling bij textuurarme objecten (mat, geleerde matchers, feed-forward modellen) | **Productierijp** |
| Automatische herkenning en fitting van vlak, cilinder, kegel, bol en torus | **Haalbaar** met gedegen engineering |
| Automatische feature tree voor prismatische en gedraaide onderdelen | **Haalbaar**, met human-in-the-loop bij twijfel |
| Glanzend metaal zonder scanspray | **Deels haalbaar** (flitser-cues, silhouetten); spray blijft de betrouwbare route |
| Vrije-vormoppervlakken → NURBS (niet-parametrisch) | **Haalbaar** (auto-surfacing) |
| Organische vormen → parametrische features | **Onderzoeksfase** |
| Transparante objecten zonder coating | **Onderzoeksfase** |

---

## 2. Scope, uitgangspunten en kwaliteitsdoelen

### 2.1 Doelgroepen en use-cases

- **Onderhoud, reparatie en MKB-maakindustrie:** reverse engineering van versleten of niet meer leverbare onderdelen.
- **Productontwikkeling:** bestaande onderdelen als startpunt, of voor een passingscontrole van nieuwe ontwerpen.
- **Makers en 3D-printgebruikers:** vervangingsonderdelen, adapters en beugels.
- **Engineeringbureaus:** een snelle eerste opname vóór de inzet van een meetmachine (CMM) of structured-light-scanner.

### 2.2 Productscope

| Dimensie | v1 (MVP) | Later | Buiten scope |
|---|---|---|---|
| Objectgrootte | ca. 2–50 cm | tot ca. 2 m (sessies samenvoegen) | ruimtes, gebouwen |
| Vormklasse | prismatisch (gefreesd), gedraaid, eenvoudig giet- en spuitgietwerk | plaatwerk, vrije vorm (NURBS), meerdelige samenstellingen | organisch, vervormbaar |
| Oppervlak | mat, of mat gemaakt met scanspray; textuurarm met mat of markers | glanzend metaal zonder spray (flitser-cues) | transparant zonder coating |
| Output | STEP AP242, IGES, STL/3MF, CadQuery/build123d, FreeCAD | koppelingen met Onshape, Fusion en SolidWorks; PMI/GD&T | — |

### 2.3 Kwaliteitsdoelen

We specificeren nauwkeurigheid zoals bij meetmachines gebruikelijk is: als uitgebreide onzekerheid (95%) van de vorm **U95 = a + b · L**, met *L* de gemeten lengte. De waarden hieronder zijn **ontwerpdoelen**; Fase 0 valideert ze met een referentieset (§8).

| Modus | Schaalbron | Doel U95 (goed zichtbare features) | Toepassing |
|---|---|---|---|
| **Precisie** | geverifieerde of gekalibreerde kalibratiemat, optioneel met een schuifmaatreferentie | ±(0,2 mm + 0,1% · L) | passende onderdelen, vervangingsdelen |
| **Standaard** | referentieobject (bankpas, munt, A4) | ±(0,5 mm + 0,5% · L) | behuizingen, beugels, bevestigingen |
| **Snel** | alleen ARCore-bewegingstracking (VIO) | ±(1 mm + 3% · L) | visualisatie, concepten |

| Eigenschap | Doel |
|---|---|
| Opnametijd (klein onderdeel, beide zijden) | 3–8 minuten |
| Verwerkingstijd | Preview ≤ 3 min · Standaard p50 ≤ 20 min, p95 ≤ 45 min · Precisie ≤ 60 min |
| Automatiseringsgraad (in-scope objecten) | MVP: ≥ 60% bruikbaar zonder correctie en ≥ 90% met hooguit 5 minuten correctie · v2: ≥ 80% zonder correctie |
| Geldigheid | 100% van de geëxporteerde solids doorstaat B-rep-validatie en een STEP-round-trip; anders volgt automatisch een lager outputniveau (§6.9) |
| Computekosten | ≤ € 0,50 per scan in Standaard-modus |
| Privacy | EU-dataresidentie, versleuteling at rest en in transit, gebruik voor training alleen na opt-in |

### 2.4 Kernnuance: "parametrisch" is niet hetzelfde als STEP of IGES

STEP (ISO 10303: AP203, AP214, AP242) en IGES zijn **uitwisselingsformaten voor geometrie**. Ze transporteren een *boundary representation* (B-rep): faces, edges en vertices met de onderliggende analytische of NURBS-oppervlakken, en in AP242 ook productinformatie (PMI). De **constructiehistorie** — schetsen, geometrische constraints, extrusies, parameters — gaat in de praktijk níet mee. De STEP-delen voor parametrisatie en procedurele modellen (o.a. ISO 10303-55, -108 en -111) bestaan wel, maar vrijwel geen CAD-pakket ondersteunt ze.

Daarom zijn **"parametrisch" en "STEP" voor deze architectuur twee afzonderlijke leveringen**:

1. **STEP AP242 met een schone B-rep.** Analytische oppervlakken (vlak, cilinder, kegel, bol, torus) in plaats van duizenden driehoekjes. Zo'n model is direct bewerkbaar met *direct modeling* en *feature recognition* in gangbare CAD (bijv. SolidWorks FeatureWorks, Siemens NX Synchronous Technology, Creo Flexible Modeling, direct editing in Fusion).
2. **Een echte feature tree** (schets → extrusie/revolutie → gaten → afrondingen → patronen), geleverd als:
   - **code-CAD:** een leesbaar CadQuery- of build123d-script (Python op OCCT) met benoemde parameters bovenaan — volledig parametrisch, versiebeheerbaar en door iedereen uit te voeren (voorbeeld in §6.9);
   - **FreeCAD `.FCStd`:** een PartDesign-body met schetsen en constraints, in een open formaat;
   - **API-koppelingen** die de feature tree natief opbouwen: Onshape (REST API / FeatureScript), Fusion (add-in via de Fusion API), later SolidWorks (API).
3. **IGES** alleen voor legacysystemen. Het formaat is verouderd (laatste versie 5.3, uit 1996) en solids komen niet in elk pakket betrouwbaar over.

---

## 3. Systeemoverzicht

```text
+------------------------------------------------------------------------------------------------+
| ANDROID-APP  (Kotlin + C++/NDK)                                                                |
|                                                                                                |
| ARCore-sessie                Camera2 via Shared Camera          Sensoren                       |
| - VIO-poses (metrisch)       - stills op volle resolutie        - IMU, 200-400 Hz              |
| - keyframes (~1080p)         - handmatige focus/belichting      - gedeelde tijdbasis           |
| - raw depth + confidence     - OIS/EIS/vervormingscorr. uit                                    |
| - lichtschatting (HDR)                                                                         |
|       |                              |                                  |                      |
|       +------------------------------+----------------------------------+                      |
|                                      v                                                         |
| Live kwaliteitsengine: dekking + next-best-view, blur, belichting, glans, textuur,             |
|                        matdetectie, objectmasker, opnamesturing                                |
|       |                             |                                                          |
|       v                             v                                                          |
| AR-overlays + haptiek               Scan Bundle: keyframes, stills, poses, intrinsics,         |
| (begeleiding van de gebruiker)      diepte, IMU, markers, maskers (versleuteld)                |
+-------------------------------------|----------------------------------------------------------+
                                      | HTTPS, hervatbare upload (al tijdens het scannen)
+-------------------------------------v----------------------------------------------------------+
| CLOUD  (EU-regio)                                                                              |
|                                                                                                |
| API-gateway (OIDC) -> Scan-API -> object storage (bundles, artefacten) + Postgres (metadata)   |
|                                     |                                                          |
|                                     v  event: upload compleet                                  |
| Temporal-workflow per scan -> workerpools op Kubernetes (CPU + GPU, autoscaling)               |
|                                                                                                |
|   [1] Ingest/QA -> [2] SfM + schaalfusie -> [3] Dichte geometrie -> [4] Mesh-to-CAD            |
|   -> [5] B-rep-validatie -> [6] Export + meetrapport ---------------------------+              |
|                                     ^                                           |              |
|                                     | signals: correcties en bevestigingen      |              |
| Review-/editorservice (web + app) --+                                           |              |
+---------------------------------------------------------------------------------|--------------+
                                                                                  v
  STEP AP242 | IGES | STL/3MF | CadQuery/build123d | FreeCAD | Onshape/Fusion | meetrapport
```

**Hoofdstroom**

1. **Opname (toestel).** ARCore levert real-time metrische cameraposes en doorlopende keyframes (~1080p); Camera2 levert op de doelposities stills op volle resolutie. De kwaliteitsengine begeleidt de gebruiker en stuurt de opname automatisch.
2. **Scan Bundle.** Keyframes, stills, poses, intrinsics, diepte, IMU-data, markerdetecties en objectmaskers worden versleuteld en in delen geüpload — al tijdens het scannen.
3. **Reconstructie (cloud).** Marker-ondersteunde Structure-from-Motion met schaalfusie, daarna dichte geometrie met een onzekerheid per punt.
4. **Mesh-to-CAD (cloud).** Segmentatie → primitieven → ontwerpintentie → B-rep en feature tree → verificatie tegen scan en beelden.
5. **Review en export.** De gebruiker ziet het model, een afwijkingskaart en maten met onzekerheid, corrigeert waar nodig en exporteert.

> **Waarom niet alles op het toestel?** Een dichte reconstructie van honderden keyframes en tientallen stills van 12 MP, plus de CAD-modellen (segmentatienetwerk, taalmodel, CAD-kernel) vraagt minuten GPU-rekentijd en gigabytes geheugen. Op een telefoon leidt dat tot thermische throttling en grote verschillen tussen toestellen. Cloudresultaten kunnen bovendien opnieuw worden berekend zodra de modellen verbeteren. Het toestel doet wat alleen het toestel kan: real-time begeleiding, keyframe-selectie en een grove preview.

---

## 4. Functionaliteit en user experience

### 4.1 User journey stap voor stap

**Fase A — Voorbereiden**

1. **Onboarding en toestelcheck (eenmalig).** De app bepaalt het capability-profiel van het toestel: ondersteuning voor ARCore en de Depth API; welke extra Camera2-stream naast ARCore haalbaar is (Shared Camera); handmatige sensorbesturing (`MANUAL_SENSOR`) en of de focusafstand gekalibreerd is; RAW; uitschakelbare optische stabilisatie; lensvervormingscorrectie; en de tijdbasis van de cameratijdstempels. Op basis daarvan krijgt het toestel een tier en een opnamestrategie (§5.1). Optioneel filmt de gebruiker 30 seconden de kalibratiemat voor een toestelspecifieke lenskalibratie (intrinsics, vervorming, focusafhankelijke brandpuntsafstand, flitserprofiel).
2. **Nieuw project en intentie.** Twee vragen: *Wat wil je met het model?* (passend vervangingsonderdeel / 3D-print / visualisatie) en *Wat voor onderdeel is het?* (gefreesd, gedraaid, plaatwerk, giet- of spuitgietwerk, vrije vorm, weet ik niet). De antwoorden bepalen de modus, de benodigde nauwkeurigheid en de priors in de CAD-pipeline, bijvoorbeeld lossingshoeken bij spuitgietwerk.
3. **Setup-assistent en pre-scan.** Eerst een korte checklist: object op de mat, diffuus licht, uit de zon. Daarna een pre-scan van ~5 seconden: één zwaai rond het object, waarbij de app belichting, glans, transparantie en textuur beoordeelt. Het resultaat is concreet advies vóór de echte scan begint, bijvoorbeeld *"Glanzend metaal gedetecteerd: gebruik scanspray"* of *"Weinig textuur: plak 6 markerstickers"*.

**Fase B — Scannen**

4. **Object selecteren en scanvolume.** De gebruiker tikt op het object. On-device segmentatie levert een masker; samen met het matvlak geeft dat een eerste 3D-bounding box, die na een paar views via de visual hull (§4.4) wordt aangescherpt. De gebruiker kan de box in AR met handgrepen aanpassen.
5. **Geleide rondes.** Rond het object verschijnt een AR-koepel met drie ringen van doelposities (bijv. op 15°, 40° en 65° elevatie, 24–36 posities per ring). De gebruiker loopt rustig rond. De app legt **automatisch** keyframes vast zodra het toestel stil genoeg is, het beeld scherp is en de opname een nog niet gedekt gebied toevoegt; op de doelposities volgt bovendien een still op volle resolutie (§5.1). Elke geaccepteerde opname geeft een korte haptische tik, zodat de gebruiker niet voortdurend naar het scherm hoeft te kijken.
6. **Detailpass.** De app herkent kleine features en scherpe randen (gaten, sleuven, afschuiningen) en vraagt gericht om close-ups met een minimale resolutie: *"Kom dichterbij bij de vier gaten linksboven."*
7. **Omdraaien.** Voor de onderzijde volgt een tweede sessie: *"Draai het object om en leg het terug op de mat."* De app controleert of de zijwanden in beide sessies genoeg overlappen om ze later samen te voegen. Bij symmetrische of textuurarme objecten stelt de app markerstickers voor (§7.2).
8. **Volledigheidscheck.** Vóór het afronden toont de app de dekkingsgraad en de resterende blinde vlekken, als 3D-pinnen met een pijl naar de beste volgende positie. Afronden met open plekken mag, maar dat komt in het meetrapport.

**Fase C — Verwerken**

9. **Upload en directe preview.** Het uploaden loopt al tijdens het scannen op de achtergrond en is hervatbaar. Direct na de scan toont de app een grove on-device preview (de visual hull uit de maskers, aangevuld met ARCore-diepte).
10. **Cloudverwerking met tussenresultaten.** Een pushbericht per mijlpaal: na ~2–3 minuten een preview met schaalzekerheid (*"scan bruikbaar"*), na ~10–20 minuten het CAD-voorstel.

**Fase D — Reviewen en exporteren**

11. **Review.** Drie gekoppelde weergaven: het CAD-model met feature tree, een afwijkingskaart scan↔CAD, en een maatlijst met onzekerheden en snap-suggesties (*"Ø 7,96 ± 0,05 → 8,00 mm?"*). De gebruiker bevestigt of corrigeert (§4.6).
12. **Exporteren en delen.** STEP AP242, IGES, STL/3MF, CadQuery/build123d-script en FreeCAD; direct naar Onshape of Fusion; plus een meetrapport (PDF/HTML) met schaalbron, dekking en onzekerheden.

### 4.2 Real-time begeleiding tijdens het scannen

**AR-overlays**

- **Scankoepel met doelposities** die van grijs naar groen kleuren.
- **Dekkingsverf op het object zelf:** rood = niet gezien, oranje = te weinig of te schuin gezien, groen = voldoende.
- **Next-best-view-pijl** en een "spookcamera" die de volgende positie aanwijst.
- **Afstandsring** rond het richtpunt: te dichtbij, goed of te ver, afgeleid van de beoogde objectresolutie.
- **Snelheidsmeter** die tot rustige bewegingen aanzet.
- **Waarschuwingsiconen, verankerd op het object** op de plek waar glans, overbelichting of onscherpte is gedetecteerd.
- **Blinde-vlekpinnen** aan het eind van de scan.

**Feedbackmatrix.** Elk signaal wordt op het toestel gemeten en leidt tot een concrete, gelokaliseerde aanwijzing. De drempels zijn startwaarden; telemetrie (opnamekwaliteit versus eindresultaat) stelt ze later bij.

| Signaal | Meting op het toestel | Startdrempel | Feedback aan de gebruiker |
|---|---|---|---|
| Bewegingsonscherpte | voorspelde blur *b* (formule hieronder) uit gyroscoop en ARCore-snelheid; Laplace-variantie binnen het objectmasker | *b* > 0,5 px (Precisie) of > 1 px | geen opname; "Langzamer bewegen"; snelheidsmeter kleurt oranje |
| Scherpte en focus | dieptebereik van het objectdeel in beeld versus de scherptediepte bij de huidige focusafstand | objectdeel buiten de scherptediepte | focus bijsturen of focus-bracketing |
| Belichting | % geclipte pixels en gemiddelde luminantie binnen het masker; ISO en sluitertijd uit de capture-metadata | > 1% geclipt, ISO > 800 of sluitertijd > 1/125 s | belichting vergrendelen op het object; "Meer licht nodig" of flitsermodus |
| Harde lichtbron | ARCore Environmental HDR: sterkte van het hoofdlicht ten opzichte van het omgevingslicht | hoge verhouding | "Gebruik diffuus licht of ga uit direct zonlicht" |
| Glans | kijkhoekafhankelijke hooglichten: verzadigde, kleurarme vlekken die tussen frames over het oppervlak verschuiven | > 5% van het objectoppervlak | glansicoon op de plek zelf; advies scanspray |
| Textuur | aantal keypoints per cm² objectoppervlak | onder drempel | "Textuurarm: gebruik de mat of markers"; tracking leunt automatisch zwaarder op de mat |
| Tracking | ARCore `TrackingState` en `TrackingFailureReason` | niet `TRACKING` | melding per oorzaak: te weinig licht, te snelle beweging, te weinig features |
| Afstand en resolutie | objectresolutie van de stills, GSD ≈ *Z / f*<sub>px</sub>, met afstand *Z* uit de mat-pose of de visual hull | buiten 0,05–0,15 mm/px (Precisie) | "Dichterbij" of "Verder weg" via de afstandsring |
| Parallax en overlap | hoek tussen opeenvolgende keyframes | > 15° | "Te grote sprong, ga iets terug" |
| Objectbeweging | het objectmasker valt niet meer samen met de projectie van de visual hull, of features op het object verschuiven ten opzichte van de mat | verschuiving > 1 mm | "Object verschoven" → nieuwe sessie |
| Warmte | thermische status en headroom van het toestel | ≥ ernstig | on-device modellen terugschakelen, pauze voorstellen |

De voorspelde bewegingsonscherpte in pixels:

```math
b \approx \left(\lVert\boldsymbol{\omega}\rVert + \frac{\lVert\mathbf{v}_\perp\rVert}{Z}\right) t_\text{exp}\, f_\text{px}
```

Hierin is *ω* de hoeksnelheid (gyroscoop), *v*<sub>⊥</sub> de snelheid loodrecht op de kijkrichting (ARCore), *Z* de objectafstand, *t*<sub>exp</sub> de sluitertijd en *f*<sub>px</sub> de brandpuntsafstand in pixels. Een voorbeeld: bij *f*<sub>px</sub> ≈ 2.800, een sluitertijd van 1/250 s en 30 cm afstand geeft 1 cm/s zijwaartse beweging al ~0,4 px onscherpte. De app wacht daarom niet op een sluiterknop, maar maakt foto's op de rustigste momenten in de gyroscoopstroom (*motion-gating*).

### 4.3 Belichting, materiaal en optiek

- **Ideaal licht is helder en diffuus:** bewolkt daglicht, of twee LED-panelen met diffuser. Vermijd direct zonlicht en spots. Een praktisch detail: de eigen schaduw van de gebruiker beweegt mee rond het object en verstoort de fotometrische consistentie — nog een reden voor licht van meerdere kanten.
- **Meten:** de app schat de scèneluminantie uit de belichtingsmetadata (sluitertijd, ISO, diafragma → EV) en de lichtrichting en -hardheid uit ARCore Environmental HDR.
- **Belichtingsstrategie:** belichting en witbalans per ring vergrendelen (consistente kleuren voor de dichte reconstructie), meten op het objectmasker in plaats van de achtergrond, een korte sluitertijd (richtwaarde ≤ 1/250 s) en ISO ≤ 800. In Precisie-modus eventueel RAW (DNG) voor een subset detailopnamen.
- **Scherptediepte:** bij hoofdcamera's met een grote sensor en een groot diafragma (f/1,4–f/1,9) is de scherptediepte op 25–30 cm vaak maar enkele centimeters. Tijdens de tracking staat ARCore expliciet op autofocus (`FocusMode.AUTO`; de standaard is FIXED, wat voor dichtbij ongeschikt is). Voor stills stuurt de app de focusafstand handmatig naar het objectdeel in beeld, vergrendelt de focus per ring en legt de focusafstand per frame vast (nodig voor focusafhankelijke intrinsics, §4.5). Bij diepe objecten gebruikt de app focus-bracketing.
- **Materiaaladvies uit de pre-scan:** scanspray bij glans of transparantie, markers bij gebrek aan textuur, flitsermodus bij textuurloze of glanzende objecten (§7.1, §7.2).
- **Achtergrond:** de mat, niet glanzend. Geen reflecterend tafelblad.

### 4.4 Blinde vlekken en volledigheid

**Dekkingsmodel.** Binnen het scanvolume houdt de app een grove oppervlakterepresentatie bij: een voxelgrid (2–5 mm) dat met de objectmaskers uit elke view wordt uitgesneden (*space carving*, oftewel een visual hull), met surfels op het resulterende oppervlak. ARCore-diepte vult dit aan waar ze betrouwbaar is. Ze is het nauwkeurigst tussen 0,5 en 5 m en heeft een lage resolutie (typisch ~160 × 120), dus op 20–40 cm is ze alleen een grove aanvulling. De visual hull hangt niet af van diepte of textuur, en ziet alleen holtes niet — precies de structurele blinde vlekken hieronder.

- Per surfel telt de app **goede observaties**: invalshoek < ~65°, resolutie binnen het doelbereik, scherp en niet overbelicht.
- Een surfel geldt als **gedekt** bij minstens drie goede observaties, waarvan minstens één paar met ≥ 10–15° triangulatiehoek (parallax).
- **Next-best-view:** elke kandidaatpositie op de koepel krijgt een score: de som van de nog niet gedekte, zichtbare surfels maal de verwachte opnamekwaliteit, min een straf voor loopafstand. De GPU bepaalt zichtbaarheid door de surfelkaart vanuit elke kandidaat te renderen; 2–5 Hz is voldoende.

**Structurele blinde vlekken** krijgen een eigen behandeling:

- **Contactvlak met de mat** → omdraai-workflow (§4.1, stap 7).
- **Diepe gaten en sleuven** (diepte/breedte > ~1,5) → de melding "niet volledig meetbaar", plus een vraag in de review: *doorgaand of blind? Diepte met de schuifmaat?*
- **Onbereikbare holtes** → gelabeld als *afgeleid, niet gemeten*, in het model en in het meetrapport.

### 4.5 Absolute schaal: kalibratie en fusie

Een camera meet alleen hoeken; afstanden volgen pas uit een bekende referentie. ARCore gebruikt de IMU voor een metrische schaal, maar door sensorbias en drift wijkt die typisch 1–3% af. Op 100 mm is dat 1–3 mm — te veel voor passende onderdelen. Cam-to-CAD combineert daarom meerdere **onafhankelijke schaalbronnen** en rapporteert het resultaat mét onzekerheid.

| Bron | Principe | Typische onzekerheid (1σ) | Beschikbaarheid |
|---|---|---|---|
| ARCore-VIO | metrische odometrie uit beeld + IMU | 1–3% | altijd |
| ToF-diepte | directe afstandsmeting | ~1%, plus materiaalafhankelijke bias | vrijwel afwezig: ARCore ondersteunt ToF alleen op enkele toestellen uit 2019–2020, en niet in Shared Camera-modus. Ontwerp dus voor diepte-uit-beweging |
| Kalibratiemat (ChArUco/AprilTag) | markerhoeken als controlepunten met bekende coördinaten in de bundle adjustment | zelf geprint: 0,1–0,5%; na verificatie ~0,05–0,1%; gekalibreerde Pro-mat: ≤ 0,05% | Precisie- en Standaard-modus |
| Referentieobject | automatisch herkend, randen subpixel gefit: bankpas (ID-1: 85,60 × 53,98 mm), €1-munt (Ø 23,25 mm), €2-munt (Ø 25,75 mm) | 0,2–0,5% | Standaard-modus |
| Schuifmaatmeting | de gebruiker koppelt een gemeten maat aan twee features | 0,02–0,05 mm absoluut | optioneel; de sterkste bron voor dat onderdeel |

**Ontwerp van de kalibratiemat**

- **ChArUco-patroon** (schaakbord + ArUco-markers) of een AprilTag-raster rond een vrij middenvlak. Schaakbordhoeken zijn subpixel-nauwkeurig te detecteren; unieke marker-ID's maken gedeeltelijke afdekking door het object onschadelijk.
- **Willekeurige, contrastrijke textuur tussen de markers**, die ook de camerabepaling helpt bij textuurarme objecten (§7.2).
- **Twee loodrechte meetlijnen van 100,0 mm** om de printschaal te verifiëren: printers en papier wijken per richting af.
- **Twee varianten:** een printbare mat (A4/A3) en een gekalibreerde **Pro-mat** (stijf en vlak, bijv. aluminium composiet) met een QR-code die het individuele correctiebestand ophaalt.
- De mat moet vlak liggen (papier op een stijve plaat) en mat zijn (geen fotopapier).

**Fusie**

- In de bundle adjustment zijn de markerhoeken controlepunten met bekende metrische coördinaten (met de onzekerheid van de mat). Dat legt de schaal vast én een Z-as loodrecht op de mat.
- ARCore-poses zijn zachte priors (positiecovariantie); ze helpen de convergentie bij weinig overlap.
- Referentieobjecten (en ToF-diepte op de zeldzame toestellen die dat hebben) leveren elk een schaalwaarneming *s*<sub>k</sub> ± *σ*<sub>k</sub>.
- Een ingevoerde schuifmaatmeting wordt **na** de feature-fit toegepast: ze verwijst naar features, niet naar pixels.
- De bronnen worden gecombineerd als gewogen gemiddelde, met een consistentietoets:

```math
s = \frac{\sum_k s_k/\sigma_k^2}{\sum_k 1/\sigma_k^2}, \qquad
\sigma_s^2 = \Big(\sum_k 1/\sigma_k^2\Big)^{-1}, \qquad
\chi^2 = \sum_k \frac{(s_k - s)^2}{\sigma_k^2}
```

- **Consistentietoets:** is *χ*² te groot, dan markeert de app de afwijkende bron. Het klassieke geval is een mat die met "aanpassen aan pagina" is geprint (94–97%): die wijkt duidelijk af van de VIO-schaal → *"Je mat lijkt verkleind geprint. Meet de 100 mm-lijn na."* De ingevoerde correctiefactor wordt per mat-ID bewaard.
- De UI toont een **schaalzekerheidslabel**, bijvoorbeeld *"Schaal ±0,06% — bron: mat + schuifmaat"*.

**Optische effecten die zich als schaalfout voordoen**

- **Focus-breathing.** Bij (auto)focus verandert de effectieve brandpuntsafstand. De app vergrendelt de focus per ring en logt de focusafstand per frame; in de bundle adjustment delen frames met dezelfde focus één set intrinsics. Twee Camera2-valkuilen: de focusafstand is alleen in dioptrieën als de focuskalibratie van het toestel `APPROXIMATE` of `CALIBRATED` is, en de fabriekskalibratie (`LENS_INTRINSIC_CALIBRATION`, ook per frame in de capture-metadata) is uitgedrukt in pixels van de volledige sensor vóór correctie. Die moet dus per outputstream worden geschaald en bijgesneden voordat ze als prior bruikbaar is.
- **Beeldstabilisatie.** Optische (OIS) en elektronische (EIS) stabilisatie verschuiven of vervormen het beeld per frame. De app schakelt beide uit waar het toestel dat toestaat (Camera2: `LENS_OPTICAL_STABILIZATION_MODE_OFF`, `CONTROL_VIDEO_STABILIZATION_MODE_OFF`, en geen preview-stabilisatie, want die overschrijft de OIS-instelling); anders wordt het hoofdpunt per frame geschat.
- **Lensvervormingscorrectie.** Veel toestellen corrigeren vervorming al in de beeldpijplijn. Kies één lijn: correctie uit (`DISTORTION_CORRECTION_MODE_OFF`) en zelf modelleren, of correctie aan en alleen de restvervorming modelleren — nooit gemengd binnen één scan.
- **Rolling shutter.** Door alleen bij (bijna) stilstand te fotograferen wordt het effect verwaarloosbaar; zo niet, dan een rolling-shutter-bewuste bundle adjustment.
- **Eén lens per intrinsics-groep.** Hoofdcamera, tele en groothoek hebben elk eigen intrinsics en worden niet gemengd.

### 4.6 Review en correctie (human-in-the-loop)

**Principe: copilot, geen autopilot.** Het systeem stelt voor; de gebruiker beslist alleen waar het systeem onzeker is. Dat is actief leren: gemiddeld 0–5 gerichte vragen per scan, in plaats van de gebruiker alles te laten controleren.

- **Feature tree** met per feature een betrouwbaarheidsindicator en de onderliggende meting (herleidbaarheid).
- **Maatlijst:** gemeten waarde ± U95, gesnapte ontwerpwaarde en de reden (*"ISO 273 doorgangsgat M6, middel"*).
- **Afwijkingskaart:** een kleurkaart scan↔CAD; een klik op een gebied toont de residuen.
- **Correcties met één tik:** regio's samenvoegen of splitsen, primitieftype wijzigen, scherp ↔ afgerond, een snap accepteren of weigeren, een patroon bevestigen.
- **Schuifmaatinvoer:** kies twee features en voer de gemeten maat in → globale herschaling of lokale correctie.
- **Optioneel in natuurlijke taal** (*"maak alle gaten M5-tapgat, 10 mm diep"*): vertaald naar gevalideerde feature-bewerkingen, nooit rechtstreeks naar geometrie.
- **Snelle regeneratie:** een correctie herstart de workflow alleen vanaf de laagst geraakte stap (met gecachte tussenresultaten), zodat een nieuw model binnen seconden tot een minuut klaarstaat.
- **Leren van correcties:** na opt-in worden correcties gelogd als trainingssignaal (§6.12).

---

## 5. Technische stack

### 5.1 Mobiele app (Android)

**Aanbeveling:** native Kotlin + Jetpack Compose voor de UI, met een gedeelde C++-kern (NDK) voor computer vision en geometrie. ARCore voor tracking; Camera2 via de ARCore **Shared Camera** API voor beeldopname; Filament voor rendering; LiteRT (voorheen TensorFlow Lite) voor on-device ML.

> **ARKit of ARCore?** ARKit is uitsluitend iOS. Voor Android is **ARCore** het equivalent; een latere iOS-app gebruikt ARKit (§5.3).

| Laag | Keuze | Toelichting |
|---|---|---|
| UI | Kotlin, Jetpack Compose | De AR-weergave als `SurfaceView` ingebed |
| AR en tracking | ARCore: motion tracking (VIO), Depth API (*raw depth* + confidence), vlakdetectie, Light Estimation (Environmental HDR), Recording & Playback | Metrische poses; grove diepte voor de dekking; playback voor regressietests (§8) — let op: sessies met Shared Camera kunnen wel worden opgenomen, maar niet afgespeeld |
| Camera | Camera2 naast ARCore via Shared Camera; handmatige focus en belichting; OIS, EIS en vervormingscorrectie uit. Gegarandeerd is maar één extra stream, op een van de ARCore-CPU-resoluties (typisch ≤ 1920 × 1080); stills op volle resolutie verlopen daarom per toestel gelijktijdig of via *stop-and-shoot* (zie hieronder) | Volledige controle over de beelden die de nauwkeurigheid bepalen |
| Sensoren | Gyroscoop en versnellingsmeter (≥ 200 Hz); cameratijdstempels in dezelfde tijdbasis als de sensoren (`SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME`). Koppelen via `Frame.getAndroidCameraTimestamp()`: de tijdbasis van `Frame.getTimestamp()` is niet gedefinieerd | Motion-gating, blurvoorspelling, optionele visueel-inertiële bundle adjustment in de cloud |
| CV-kern (C++) | OpenCV 4.x (ChArUco-/AprilTag-detectie, scherptemetrieken), Eigen, visual hull/surfel-kaart op de GPU (Vulkan- of GLES-compute) | Real-time kwaliteit en dekking |
| On-device ML | LiteRT (CompiledModel-API met GPU- en NPU-accelerators); alternatief ExecuTorch | Objectsegmentatie (EdgeTAM-klasse, of MediaPipe Interactive Segmenter voor tik-om-te-selecteren), materiaalclassifier (glans, transparant, textuurarm), keypoint-dichtheid |
| Rendering | Filament, direct of via SceneView (actief onderhouden maar snel veranderend: versies vastpinnen) | AR-overlays, dekkingsverf, previewmesh |
| Opslag en upload | Room (sessiemetadata), WorkManager + hervatbare upload (tus, of S3-multipart/GCS-resumable), AES-GCM met sleutels in de Android Keystore | Offline-first, bestand tegen slechte verbindingen |
| Telemetrie | OpenTelemetry + crashrapportage, aangevuld met opnamekwaliteitsmetrieken | Verband leggen tussen opnamekwaliteit en eindresultaat |

```text
 ARCore-sessie (30 fps)                          Camera2 (Shared Camera)
 pose, intrinsics, keyframe (~1080p),            stills op volle resolutie +
 raw depth, lichtschatting                       metadata (focus, belichting, ISO)
            |                                                |
            +------------------------+-----------------------+
                                     v
       Frame-synchronisatie (gedeelde tijdbasis, pose-interpolatie)
                                     |
         +---------------------------+---------------------------+
         v                           v                           v
 Kwaliteitsmodule            Dekkingsmodel               Opnamesturing
 blur, belichting,           visual hull uit maskers     motion-gating, parallax,
 glans, textuur,             + ARCore-diepte;            dekkingswinst; op doel-
 tracking                    next-best-view (GPU)        positie: still (stop-and-shoot)
         |                           |                           |
         +---------------> AR-overlays + haptiek <---------------+
                                                                 |
                                                                 v
                                              Scan Bundle-writer (AES-GCM, in delen)
                                                                 |
                                                                 v
                                              WorkManager: hervatbare upload
```

**Opnamestrategie: hoge resolutie naast ARCore.** ARCore volgt de camera op zijn eigen CPU-beeld: standaard 640 × 480, op de meeste toestellen instelbaar tot ~1920 × 1080. Via Shared Camera is maar één extra stream gegarandeerd, en alleen op een van die resoluties. Op 30 cm afstand geeft 1080p een objectresolutie van ~0,23 mm/px: genoeg voor dekking en camerabepaling, niet voor Precisie. Daarom legt de app twee soorten beelden vast:

1. **Doorlopende keyframes** (het ARCore-CPU-beeld, ~1080p) voor dekking, de verbinding tussen views en de preview.
2. **Stills op volle resolutie** (12 MP of meer, ~0,07–0,11 mm/px op 20–30 cm) op de doelposities en in de detailpass:
   - *gelijktijdig* op toestellen waar een extra hoge-resolutiestream naast ARCore aantoonbaar stabiel werkt (per toestel getest bij de certificering);
   - anders via **stop-and-shoot**: zodra het toestel stilstaat op een doelpositie, pauzeert de app ARCore kort, maakt via Camera2 een still met vergrendelde focus en belichting (eventueel RAW) en hervat de tracking. Dat kost in de orde van een seconde per still (te meten in Fase 0) en is alleen nodig op de doelposities: 20–60 per scan.

De ARCore-pose is daarbij alleen een prior. De exacte poses volgen in de cloud uit de beelden en de markers (§5.3), dus een korte trackingonderbreking kost geen nauwkeurigheid.

**Waarom native en niet Unity, Flutter of React Native?** De opnamekwaliteit hangt af van laag-niveau Camera2-controle (focus, OIS/EIS, RAW, vervormingscorrectie, tijdstempels) in combinatie met ARCore Shared Camera. Cross-platformlagen abstraheren dat weg of ondersteunen het niet. Unity AR Foundation is sterk in AR-UI maar beperkt in cameracontrole. Flutter of React Native kan voor schermen buiten de opname, maar voegt brugcomplexiteit toe zonder de kern te vereenvoudigen. Voor iOS komt later een aparte native app (Swift, ARKit, AVFoundation) die de C++-kern en het Scan Bundle-formaat deelt.

**Toesteltiers.** Android is gefragmenteerd, dus de app classificeert elk toestel:

- **Tier 1 — Precisie:** ARCore + Depth API; handmatige sensorbesturing (`MANUAL_SENSOR`) met een gekalibreerde focusafstand; OIS uit te schakelen; stills op volle resolutie (gelijktijdig of via stop-and-shoot); REALTIME-tijdstempels. Gecertificeerd na een labtest met de referentieset.
- **Tier 2 — Standaard:** ARCore, maar zonder handmatige focus of zonder betrouwbare stills; alleen keyframes op ~1080p.
- **Tier 3 — Snel:** ARCore zonder Depth API of met beperkte cameratoegang.

Toestelprofielen (eigenaardigheden, intrinsics-priors, maximale resoluties) worden server-side beheerd en bijgewerkt zonder app-release.

**Omvang van de upload.** Typisch 150–250 keyframes op ~1080p (0,3–0,6 MB per stuk) plus 20–60 stills van 12 MP (1,5–3 MB), diepte en maskers: ~150–300 MB per scan. Het uploaden start al tijdens het scannen, in delen van 8–16 MB, bij voorkeur via wifi.

### 5.2 Backend en cloudverwerking

**Principes:** event-driven, *durable workflows*, stateless services, GPU's die naar nul terugschalen, reproduceerbaarheid (bundle + pipelineversie = deterministisch resultaat; tussenresultaten per stap content-addressed gecachet) en EU-dataresidentie.

```text
 App/web --HTTPS--> API-gateway (OIDC) --> Scan-API --> Postgres (metadata)
                                              |  signed URLs
                                              v
                        Object storage (bundles, artefacten, exports)
                                              |  event "upload compleet" (Pub/Sub of SQS)
                                              v
                        Temporal: ScanProcessingWorkflow (1 per scan)
                                              |
          +-----------+-----------+-----------+-----------+-----------+-----------+
          v           v           v           v           v           v           v
        Ingest/QA   SfM +       Dichte      Segmentatie CAD-        B-rep +     Export +
        (CPU)       schaal      geometrie   (GPU)       synthese    validatie   rapport
                    (GPU+CPU)   (GPU)                   (GPU, LLM)  (CPU, OCCT) (CPU)
          |           |           |           |           |           |           |
          +------ workerpools op Kubernetes, autoscaling op wachtrijlengte -------+
                                              |
               Redis (voortgang) --> WebSocket/SSE + push (FCM) --> app / web
                                              |
               Review-/editorservice <--> Temporal-signals (human-in-the-loop)
```

| Component | Aanbeveling | Alternatief | Waarom |
|---|---|---|---|
| Cloud | GCP of AWS in een EU-regio met GPU-capaciteit (bijv. `europe-west4`, Nederland) | Azure | Beschikbaarheid van L4/L40S, managed Kubernetes |
| Compute | Kubernetes (GKE/EKS) met GPU-nodepools; schalen op wachtrijlengte (KEDA) met Karpenter of GKE-autoscaling; spot-instances voor herstartbare stappen | Serverless GPU (bijv. Modal, RunPod) voor het MVP | Schalen naar nul; kosten |
| Workflow | **Temporal**: durable, retries per stap, lange wachttijden voor human-in-the-loop via *signals* | Argo Workflows, AWS Step Functions | Een scan is een workflow van 10–60 minuten met optionele gebruikersinteractie |
| GPU's | NVIDIA L4 (24 GB) als standaard; L40S (48 GB) voor zware neurale reconstructie; H100 alleen voor training | A10G | Prijs/prestatie voor inferentie en splatting |
| Model serving | NVIDIA Triton (segmentatie, normalen), vLLM (CAD-taalmodel) | TorchServe, Ray Serve | Batching, GPU-deling |
| Opslag | Object storage met lifecycle-regels (ruwe bundles na 90 dagen naar cold storage), Postgres, Redis | — | Kosten; herverwerking met nieuwere modellen |
| Messaging | Pub/Sub of SQS/SNS | NATS JetStream, Kafka | Events tussen API en workflows |
| Realtime status | WebSocket/SSE + pushberichten (FCM) | — | UX tijdens 10–30 minuten verwerking |
| Observability | OpenTelemetry, Prometheus/Grafana, Sentry | Datadog | Tijd per stap, GPU-gebruik, faalredenen, kwaliteitsmetrieken |
| ML-platform | MLflow of W&B; datalake (Parquet op object storage); synthetische datafabriek (Blender/BlenderProc) | Vertex AI, SageMaker | Training en evaluatie (§6.12) |
| Security | OIDC (bijv. Keycloak, Auth0, Firebase Auth), KMS/CMEK, workers zonder internet-egress, sandbox (gVisor/Firecracker) voor gegenereerde CAD-code | — | IP-bescherming, AVG |

**Verwerkingstiers**

| Tier | Stappen | Doorlooptijd (doel) | Indicatieve compute | Gebruik |
|---|---|---|---|---|
| Preview | Feed-forward poses en geometrie op een subset keyframes + snelle splat-/TSDF-mesh | 1–3 min | < 2 GPU-min | Directe feedback: is de scan bruikbaar? |
| Standaard | Marker-SfM + MVS + oppervlakte-GS + mesh-to-CAD | 10–25 min | 10–20 GPU-min (L4) + 10–20 vCPU-min | Standaard |
| Precisie | + model-gebaseerde beeldverfijning, meer hypothesen, optioneel reflectiebewuste reconstructie | 30–60 min | 30–60 GPU-min (L4/L40S) | Passende onderdelen |

**Kosten (orde van grootte, prijzen september 2026).** Een L4-GPU kost on-demand circa $0,74 per uur (GCP `g2-standard-4` in `europe-west4`) tot $0,80 per uur (AWS `g6.xlarge`); spot-capaciteit is doorgaans fors goedkoper. Met 10–20 GPU-minuten plus 10–20 vCPU-minuten kost een Standaard-scan ~$0,15–0,30 aan compute, ruim binnen het doel van € 0,50. Opslag en dataverkeer (150–300 MB per bundle, met lifecycle-regels) kosten enkele centen per scan. De grootste kostenposten zijn niet de inferentie, maar mislukte scans die opnieuw moeten en de training van eigen modellen. Goede opnamebegeleiding is dus ook kostenbeheersing.

**Security en privacy**

- Scans zijn intellectueel eigendom van de klant: versleuteling at rest (KMS/CMEK) en in transit (TLS 1.3), tenant-isolatie, signed URLs, workers zonder internettoegang, audit-logging.
- AVG: EU-residentie, verwerkersovereenkomst, bewaartermijnen per tenant instelbaar (standaard: ruwe bundles 90 dagen, exports tot verwijdering), een API voor verwijdering, gebruik voor training alleen na opt-in.
- **Gegenereerde CAD-code is onbetrouwbare invoer:** uitvoeren in een sandbox zonder netwerk, met tijd- en geheugenlimieten (§6.7).
- Enterprise: on-premises of in de VPC van de klant via Helm-charts met dezelfde containers. Let op: GPL-componenten worden dan *gedistribueerd* (§5.4).

### 5.3 3D-reconstructie-engine

De reconstructie moet geen mooi plaatje opleveren, maar **metrisch correcte geometrie met een onzekerheid per punt**. Vergeleken voor deze use-case:

| Criterium | Fotogrammetrie (SfM + MVS) | NeRF / neurale SDF (NeuS, Neuralangelo) | 3D Gaussian Splatting (3DGS) | Oppervlakte-GS (2DGS, GOF, PGSR) | Feed-forward (VGGT, MapAnything, MASt3R) |
|---|---|---|---|---|---|
| Metrische nauwkeurigheid | Zeer goed (bundle adjustment met controlepunten) | Goed (SDF-varianten), afhankelijk van SfM-poses | Zwak (geometrie is geen optimalisatiedoel) | Goed | Matig (regressie zonder bundle adjustment) |
| Scherpe randen | Matig (MVS-vensters, smoothing) | Matig (SDF's ronden af) | Zwak | Goed (PGSR: vlakke delen) | Zwak |
| Textuurarme vlakken | Zwak (gaten) | Goed met geometrische priors | Matig | Goed met normaal-priors | Goed (geleerde priors) |
| Glans en reflectie | Zwak | Matig tot goed (reflectiebewuste varianten) | Zwak ("spookgeometrie" achter spiegelende vlakken) | Matig | Matig |
| Rekentijd (object, ~150 beelden, 1 GPU) | Minuten | Uren (≥ 12 u in de benchmarks hieronder) | ~10–15 min | ~10–60 min | Seconden |
| Onzekerheid en uitlegbaarheid | Zeer goed (residuen, covariantie) | Zwak | Zwak | Matig | Zwak |
| Volwassenheid | Zeer hoog | Hoog | Hoog | Middel (rijpt snel) | Middel (evolueert snel; let op licenties) |
| **Rol in Cam-to-CAD** | **Metrische ruggengraat** | Optioneel (Precisie-tier, reflecterende objecten) | Preview en visualisatie | **Volledigheid en oppervlak** | **Initialisatie, preview, fallback** |

**Indicatief — DTU-benchmark, gemiddelde Chamfer-afstand in mm (lager is beter):** 3DGS 1,96 · COLMAP 1,36 · SuGaR 1,33 · NeuS 0,84 · 2DGS 0,80 · GOF 0,74 · RaDe-GS 0,68 · Neuralangelo 0,61 · PGSR 0,52. De cijfers komen uit de respectievelijke papers (COLMAP en NeuS uit de NeuS-paper; PGSR uit de gepubliceerde TVCG-versie). De trainingstijd van de Gaussian-methoden ligt daar tussen ~10 minuten en een uur; die van de neurale SDF-methoden op ≥ 12 uur (Neuralangelo > 100 uur). Twee kanttekeningen: DTU-objecten en -camera's verschillen van onze situatie, en Chamfer is een gemiddelde — het zegt weinig over randscherpte of systematische schaalfouten. Die vangen we af met de mat (§4.5) en de CAD-verfijning (§6.8).

**Aanbevolen pipeline: hybride, met fotogrammetrie als ruggengraat**

```text
 keyframes + stills + maskers + ARCore-poses (priors) + markerhoeken
        |
        v
 [R1] features + matching: ALIKED/DISK + LightGlue; RoMa voor moeilijke paren
        |
        v
 [R2] SfM (COLMAP 4.x, global mapper) -> bundle adjustment (Ceres):
      markers als controlepunten, ARCore-poses als priors, intrinsics per focusgroep
      (init/fallback bij weinig textuur: MapAnything-Apache of VGGT-1B-Commercial)
        |
        +-----------------------------+-----------------------------+
        v                             v                             v
 [R3a] PatchMatch-MVS          [R3b] oppervlakte-GS          [R3c] visual hull
 dieptekaarten +               2DGS/PGSR-principes op        uit SAM 2-maskers
 confidence                    gsplat; maskers, normaal-
                               priors, MVS-diepte
        |                             |                             |
        +-----------------------------+-----------------------------+
                                      v
 [R4] confidence-gewogen fusie -> puntenwolk (+ normalen, sigma per punt), mesh,
      per-view diepte-/normaal-/randkaarten, camera's met covariantie -> mesh-to-CAD
```

- **[R1] Features en matching.** ALIKED of DISK met LightGlue (vanaf COLMAP 4.0 geïntegreerd); RoMa als dichte matcher voor moeilijke paren. *Niet* SuperPoint: die gewichten zijn niet commercieel bruikbaar (§5.4).
- **[R2] SfM en bundle adjustment.** COLMAP 4.x — de *global mapper* (voorheen GLOMAP) voor snelheid, de incrementele mapper als fallback — gevolgd door een eigen bundle adjustment in Ceres met de markerhoeken als controlepunten, ARCore-poses als priors en intrinsics per focusgroep (§4.5). Bij textuurarme scènes zorgt een feed-forward model (MapAnything, Apache-2.0-variant, of VGGT-1B-Commercial) voor initialisatie of fallback.
- **[R3a] MVS.** PatchMatch-MVS van COLMAP levert dieptekaarten met confidence: het metrisch betrouwbaarste deel van de geometrie.
- **[R3b] Oppervlakte-Gaussian-Splatting.** De principes van 2DGS/PGSR, geïmplementeerd op **gsplat** (Apache-2.0; de officiële 2DGS- en PGSR-code is niet commercieel bruikbaar). Gesuperviseerd met maskers, monoculaire normaal-priors (MoGe-2 of StableNormal) en MVS-diepte. Dit vult de gaten die MVS op textuurarme vlakken laat.
- **[R3c] Visual hull.** Uit de SAM 2-maskers volgt een harde buitengrens; silhouetten zijn ongevoelig voor textuur en glans.
- **[R4] Fusie.** Confidence-gewogen samenvoegen: MVS waar het betrouwbaar is, het GS-oppervlak waar MVS gaten laat, binnen de visual hull. De uitvoer bestaat uit een puntenwolk met normalen en *σ* per punt, een watertight mesh, per-view diepte-, normaal- en randkaarten, en de camera's met covariantie.

**Waarom geen NeRF als basis?** Het doel is geometrie, niet beeldsynthese. De dichtheidsvelden van NeRF vragen extra machinerie om oppervlakken te leveren. Neurale SDF's zijn nauwkeurig, maar trainen uren en ronden de scherpe CAD-randen af. 2DGS/PGSR halen vergelijkbare of betere geometrie in een fractie van de tijd. NeRF-achtige methodes blijven daarom een optionele Precisie-component, bijvoorbeeld reflectiebewuste reconstructie in de stijl van NeRO (§7.1).

**Build of buy?** Commerciële engines zoals Agisoft Metashape (met Python-API, markers en schaalstaven) of Epic RealityScan (voorheen RealityCapture) kunnen het MVP versnellen. Controleer vooraf of de licentie server-side, geautomatiseerd SaaS-gebruik toestaat; die voorwaarden zijn niet altijd openbaar. Het advies: eigen pipeline op open componenten, omdat de schaalfusie, de onzekerheid per punt en de koppeling met de CAD-stap de kern van het product zijn.

**iOS en LiDAR (later).** De LiDAR van iPhone Pro-modellen levert absolute diepte, maar met een te lage resolutie en nauwkeurigheid (orde millimeters tot centimeters) voor CAD-randen. Hij is waardevol als prior voor schaal en voor textuurarme vlakken, niet als primaire meetbron. Apple Object Capture (`ObjectCaptureSession`, vanaf iOS 17, op toestellen met LiDAR en A14 of nieuwer) is een goede UX-referentie en een mogelijk snel pad voor de preview.

### 5.4 Licentie-aandachtspunten

Veel toonaangevende modellen zijn alleen niet-commercieel beschikbaar, en **gewichten hebben vaak een andere licentie dan de code**. Voorwaarden van trainingsdatasets werken door in de gewichten. Onderstaande stand is geverifieerd in september 2026 en moet juridisch worden getoetst.

| Component | Code / gewichten | Commercieel in SaaS? | Advies |
|---|---|---|---|
| COLMAP 4.x (incl. global mapper, voorheen GLOMAP) | BSD-3 | Ja, mits zorgvuldig gebouwd | De standaardbuild bevat LSD (AGPL), SiftGPU (niet-commercieel) en CGAL (GPL): bouwen met `-DLSD_ENABLED=OFF`, zonder GPU-SIFT, eventueel zonder CGAL |
| OpenMVS | AGPL-3.0; de standaardbuild bevat een max-flow-component die alleen voor onderzoek is gelicentieerd | Riskant | Vermijden; COLMAP-PatchMatch gebruiken |
| ALIKED / DISK / LightGlue / RoMa | BSD-3 / Apache-2.0 / Apache-2.0 / MIT | Ja | Standaardkeuze voor features en matching |
| SuperPoint (Magic Leap) | Alleen niet-commercieel onderzoek | Nee | Niet gebruiken, ook niet samen met LightGlue |
| DUSt3R / MASt3R | CC BY-NC-SA 4.0 (code en gewichten) | Nee | Alleen onderzoek |
| VGGT | Code: VGGT License; VGGT-1B: CC BY-NC; **VGGT-1B-Commercial**: VGGT License (op aanvraag) | Alleen de Commercial-checkpoint | Het gebruiksbeleid sluit o.a. militaire toepassingen en de bediening van zware machines uit — toetsen aan de doelklanten |
| MapAnything | Code Apache-2.0; gewichten in een CC BY-NC- én een **Apache-2.0-variant** | Ja (Apache-variant) | Voorkeurskandidaat voor de feed-forward-laag |
| π³ (Pi3) | Code BSD-3; gewichten CC BY-NC | Nee (gewichten) | Alleen onderzoek |
| 3DGS / 2DGS / PGSR (officiële code) | Niet-commercieel (Inria/MPII, resp. ZJU) | Nee | Herimplementeren op **gsplat** (Apache-2.0, bevat 2DGS-rasterisatie); de `*_inria_wrapper`-functies van gsplat vermijden |
| SAM 2 / EdgeTAM | Apache-2.0 | Ja | Maskers (cloud) en on-device segmentatie |
| Depth Anything V2 | Small: Apache-2.0; Base/Large/Giant: CC BY-NC | Alleen Small | — |
| MoGe / MoGe-2, StableNormal, Marigold | MIT; Apache-2.0; code Apache-2.0 met v1-1-gewichten onder OpenRAIL++-M | Ja (bij Marigold gebruiksbeperkingen doorgeven in de voorwaarden) | Normaal- en diepte-priors |
| DSINE | Niet-commercieel; sluit betaalde diensten expliciet uit | Nee | Vervangen door MoGe-2 of StableNormal |
| nvdiffrast | NVIDIA Source Code License (niet-commercieel) | Nee | PyTorch3D (BSD-3) of een eigen rasterizer |
| Point Transformer V3 / Sonata | Code MIT / Apache-2.0; gewichten getraind op niet-commerciële datasets, resp. CC BY-NC | Code ja, gewichten nee | Architectuur gebruiken, zelf (voor)trainen |
| CAD-Recode / cadrille | CC BY-NC (code, gewichten én dataset) / code Apache-2.0, gewichten CC BY-NC | Nee (gewichten en data) | Eigen model trainen op eigen procedurele data (§6.12) |
| CGAL (Shape Detection, Shape Regularization, Polygonal Surface Reconstruction) | GPL-3.0+ | Alleen server-side, of een commerciële licentie (GeometryFactory) | Nooit in de APK; bij on-premises-levering een commerciële licentie |
| PyMeshLab | GPL-3.0 | Alleen server-side | Idem |
| Open3D / PyTorch3D | MIT / BSD-3 | Ja | — |
| OCCT | LGPL-2.1 met Open CASCADE-exceptie | Ja | CAD-kernel |
| CadQuery / build123d / FreeCAD | Apache-2.0 / Apache-2.0 / LGPL-2.1+ | Ja | Code-CAD en FCStd-export |

> **Consequentie:** voor de lerende componenten van de CAD-stap (segmentatie en programmasynthese) bestaat nog geen commercieel bruikbaar kant-en-klaarmodel. Een eigen data-engine en eigen training (§6.12) zijn dus geen luxe, maar een randvoorwaarde. Laat naast licenties ook een *freedom-to-operate*-analyse uitvoeren: algoritmen uit papers kunnen octrooibeschermd zijn, los van de licentie van de code.

### 5.5 Deploymentprofielen: cloud, lokaal of eigen server

De verwerkingspipeline bestaat uit containers die niet aan één cloud gebonden zijn. Dezelfde software kan daardoor op drie manieren draaien:

| Profiel | Waar | Kosten per scan | Wanneer |
|---|---|---|---|
| **Gehoste dienst (cloud)** | Kubernetes + Temporal in een EU-regio (§5.2) | ~$0,15–0,30 compute | Gebruikers zonder eigen rekenkracht; schaal |
| **Lokaal (desktop)** | pc van de gebruiker met Docker Compose; de telefoon stuurt de Scan Bundle via wifi | alleen stroom (orde € 0,05) | Makers, MKB, MVP; data blijft lokaal |
| **Eigen server** | server of NAS met GPU binnen een bedrijf | hardware + stroom | Teams, IP-gevoelige klanten (on-premises) |

Lokaal en op een eigen server vervalt de cloudorkestratie: een lichte taakwachtrij vervangt Temporal en Kubernetes. Of het geheel ook **volledig open source** kan — zonder ARCore, zonder niet-commerciële modellen en zonder cloudkosten — staat in de haalbaarheidsanalyse [OPEN-SOURCE-LOKAAL.md](OPEN-SOURCE-LOKAAL.md). Kort antwoord: ja, met drie concessies.

---

## 6. De mesh-to-parametric pipeline

Dit is de kern van het product en de moeilijkste stap: van een ruizige, onvolledige puntenwolk of mesh naar een geldige solid met een zinvolle, bewerkbare opbouw.

### 6.1 Ontwerpprincipes

1. **Meten ≠ modelleren.** Elke CAD-parameter heeft een herleidbare meetbasis: welke punten, welke beelden, welke onzekerheid.
2. **Neuro-symbolisch.** Netwerken *stellen voor* (segmentatie, oppervlaktetypes, randen, programmastructuur); geometrie *beslist* (robuuste fit, constraint-solving, kernelvalidatie).
3. **Getallen uit de meting, structuur uit het model.** Een taalmodel mag de opbouw voorstellen, maar nooit zelf maten verzinnen (§6.7).
4. **Analysis-by-synthesis.** Bouw meerdere hypothesen, vergelijk ze met de scan én de originele beelden, en kies de beste op fit, eenvoud en geldigheid (*Minimum Description Length*).
5. **Onzekerheidsbewust.** *σ* per punt → *σ* per parameter → beslissingen over snappen, relaties en vragen aan de gebruiker.
6. **Graceful degradation.** Liever een correcte B-rep zonder historie dan een feature tree die niet klopt (§6.9).

### 6.2 Overzicht

```text
 Invoer: puntenwolk (+ normalen, sigma), mesh, beelden + camera's, maskers, randkaarten
   |
 [A] Voorbewerking ......... canoniek assenstelsel (Manhattan-frame), ruis/outliers, resampling
   |
 [B] Segmentatie ........... 3D-netwerk: type + instantie + rand per punt
   |                         + 2D-cues (SAM 2-delen, randen, normalen), multi-view geprojecteerd
   |                         + klassiek: region growing op normalen en kromming
   |
 [C] Primitieven ........... robuuste fit: vlak, cilinder, kegel, bol, torus, B-spline
   |                         + modelselectie (GRIC/BIC), kinematische analyse, merge/split
   |
 [D] Ontwerpintentie ....... relaties: parallel, loodrecht, coaxiaal, symmetrie, patronen
   |                         + globale herfit met constraints; snappen naar standaardmaten
   |
 [E] Structuur ............. twee routes, parallel:
   |                         E1 bottom-up: aangrenzing -> snijkrommen -> B-rep -> features
   |                         E2 top-down:  puntenwolk + primitieven -> CAD-taalmodel -> code
   |
 [F] Analysis-by-synthesis . bouwen in OCCT, vergelijken met scan en beeldranden,
   |                         parameterverfijning, selectie op fit + eenvoud + geldigheid
   |
 [G] Validatie ............. BRepCheck, gesloten solid, zelfdoorsnijding, STEP-round-trip,
   |                         afwijkingsrapport; bij falen -> lager outputniveau
   |
 [H] Export ................ STEP AP242, IGES, STL/3MF, CadQuery/build123d, FreeCAD, API's
```

### 6.3 Stap A — Voorbewerking

- **Canoniek assenstelsel.** Z staat loodrecht op de mat (Precisie) of op het dominante vlak. Daarna volgt een **Manhattan-frame**: de normalen worden op de eenheidsbol geclusterd (mean-shift) en een orthogonaal drietal dominante richtingen gezocht. Gefreesde onderdelen hebben vrijwel altijd zo'n frame; schetsvlakken en extrusierichtingen volgen er direct uit.
- **Opschonen.** Achtergrond en mat verwijderen via maskers en het matvlak; statistische outlierverwijdering; **randbehoudende** ruisonderdrukking (bilaterale normaalfiltering, RIMLS). Géén Laplace-smoothing: die rondt randen af.
- **Normalen** met adaptieve buurten (jet fitting/PCA), consistent georiënteerd via de zichtlijnen van de camera's.
- **Resampling** naar 100–300k punten (Poisson-disk) voor de netwerken; de volledige dichtheid blijft beschikbaar voor het fitten.
- **Onzekerheid per punt** (*σ*<sub>i</sub>) uit de MVS-consistentie gaat mee naar alle volgende stappen.

### 6.4 Stap B — Multi-cue segmentatie

Doel: elk punt krijgt een oppervlaktetype, een instantie (bij welk vlak of welke cilinder het hoort) en een rand- en hoekkans.

- **3D-netwerk.** Een backbone uit de Point Transformer V3-familie met koppen zoals in HPNet en SED-Net:
  1. type per punt: vlak, cilinder, kegel, bol, torus, vrije vorm, of blend (afronding);
  2. instantie-embedding, geclusterd met mean-shift of HDBSCAN;
  3. randkans;
  4. hoekkans.

  Het netwerk wordt getraind op synthetische scans met realistische artefacten (§6.12).
- **2D-cues, naar 3D geprojecteerd.** Op de originele beelden — vooral de stills — draaien SAM 2 (deelsegmenten), een geleerde subpixel-randdetector en een monoculaire normaalschatter. Via de bekende camera's en een zichtbaarheidstest (z-buffer tegen de mesh) stemmen die cues per 3D-punt over alle views. De resolutie van de stills (~0,07–0,1 mm/px) is doorgaans fijner dan de reconstructieruis; juist bij randen zijn deze cues daarom doorslaggevend.
- **Klassiek.** Region growing op normalen en kromming (bijv. CGAL Shape Detection) als baseline, fallback en plausibiliteitscontrole.
- **Fusie.** Een graph-cut/CRF op de meshgraaf combineert de unaire termen (3D-netwerk + 2D-stemmen) met een gladheidsterm die bij een hoge randkans wordt verzwakt.

### 6.5 Stap C — Primitieven fitten en kiezen

- **Initialisatie in gesloten vorm uit de normalen.** Een cilinderas is bijvoorbeeld de eigenvector bij de kleinste eigenwaarde van Σ **n**<sub>i</sub>**n**<sub>i</sub><sup>T</sup>. Daarna volgt een robuuste niet-lineaire kleinste-kwadratenfit (Levenberg–Marquardt met Huber- of Tukey-verlies in Ceres), gewogen met 1/*σ*<sub>i</sub>². Voor zeer ruizige segmenten: Efficient RANSAC (Schnabel et al.).
- **Randbandexclusie.** Punten binnen ~2–3*σ* van een voorspelde rand krijgen weinig gewicht. Daar zit de systematische afronding van de reconstructie (§7.3).
- **Modelselectie met GRIC/BIC.** Het eenvoudigste model wint, tenzij een complexer model significant beter past. Voorbeeld: een "kegel" met een halve tophoek van 0,2° ± 0,3° wordt een cilinder — tenzij de fabricageklasse spuitgietwerk is, want dan is een lossingshoek van 0,5–3° juist ontwerpintentie.
- **Samenvoegen en splitsen** van segmenten op basis van statistisch gelijkwaardige fits, respectievelijk bimodale residuen.
- **Kinematische oppervlakteanalyse (lijngeometrie).** Een klassiek en zeer bruikbaar resultaat uit reverse engineering (Pottmann & Randrup, 1998). Een oppervlak is invariant onder een uniforme beweging met snelheidsveld **v**(**x**) = **c̄** + **c** × **x** precies als dat veld overal raakt aan het oppervlak: **c̄** · **n**<sub>i</sub> + **c** · (**p**<sub>i</sub> × **n**<sub>i</sub>) = 0 voor alle punten **p**<sub>i</sub> met normaal **n**<sub>i</sub>. Met de Plücker-coördinaten van de normaallijnen wordt dat een eigenwaardeprobleem:

```math
\mathbf{u}_i = \big(\mathbf{n}_i,\; \mathbf{p}_i \times \mathbf{n}_i\big) \in \mathbb{R}^6, \qquad
M = \sum_i \mathbf{u}_i \mathbf{u}_i^{\top}, \qquad
\min_{\lVert \mathbf{x} \rVert = 1} \mathbf{x}^{\top} M \,\mathbf{x}, \quad \mathbf{x} = (\bar{\mathbf{c}}, \mathbf{c})
```

  De eigenvector bij de kleinste eigenwaarde classificeert het oppervlak direct:

  - **c** ≈ 0 → translatie langs **c̄**: een **extrusie**;
  - spoed *p* = (**c** · **c̄**) / ‖**c**‖² ≈ 0 → rotatie om een as met richting **c** door het punt (**c** × **c̄**) / ‖**c**‖²: een **omwentelingsoppervlak**;
  - anders een schroefbeweging met spoed *p*: **schroefdraad** of een spiraalgroef.

  Het aantal (bijna-)nul-eigenwaarden onderscheidt vlak en bol (3), cilinder (2) en kegel, torus, algemene omwenteling of extrusie (1). Eén analyse levert zo de ruggengraat van de feature tree: *schets + extrusie, revolutie of sweep*.

### 6.6 Stap D — Ontwerpintentie: relaties, regularisatie en snappen

- **Relaties ontdekken** tussen primitieven: parallel, loodrecht, standaardhoeken (30°, 45°, 60°), coaxiaal, coplanair, gelijke straal, symmetrievlakken, en lineaire of cirkelvormige patronen. Voorbeeld: een steekcirkel met zes gaten op 60° (cirkelfit door de gatcentra + gelijke hoekverdeling).
- **Statistisch accepteren.** Een relatie wordt alleen aangenomen als ze binnen de meetonzekerheid past (bijv. hoekverschil < 3*σ*) én vooraf plausibel is.
- **Globale herfit (het GlobFit-principe).** Alle primitieven worden samen opnieuw gefit, met de geaccepteerde relaties als harde constraints (via herparametrisatie, dus met minder vrijheidsgraden). Het resultaat is een onderling consistent model in plaats van losse, net-niet-parallelle vlakken.
- **Snappen naar standaardwaarden (Bayesiaans).** Voor een gemeten maat *x* ± *σ* worden kandidaatwaarden *c*<sub>k</sub> met prior *π*<sub>k</sub> afgewogen tegen de hypothese "vrije maat" (*π*<sub>0</sub>, uniforme dichtheid *u*(*x*) over het plausibele bereik):

```math
P(c_k \mid x) = \frac{\pi_k\, \mathcal{N}(x;\, c_k,\, \sigma^2)}{\pi_0\, u(x) + \sum_j \pi_j\, \mathcal{N}(x;\, c_j,\, \sigma^2)}
```

  Er wordt alleen gesnapt als de posterior > 0,9 én |*x* − *c*<sub>k</sub>| ≤ 2*σ*. Anders blijft de gemeten waarde staan, gemarkeerd als *niet gesnapt*. De kandidaten komen uit een **kennisbank**:

  - hele en halve millimeters, en voorkeursgetallen (Renard-reeksen, ISO 3);
  - doorgangsgaten (ISO 273) en tapboormaten voor metrisch schroefdraad (M3 → 2,5; M4 → 3,3; M5 → 4,2; M6 → 5,0; M8 → 6,8 mm);
  - passingen (ISO 286) en lagermaten (ISO 15, bijv. 608: 8 × 22 × 7 mm);
  - O-ringgroeven (ISO 3601), plaatdiktes en standaardstralen voor afrondingen.
- **Eenhedendetectie.** Een likelihood-ratio over álle maten beslist of het ontwerp metrisch of in inches is: maten clusteren rond hele millimeters óf rond 1/16"–1/64".
- **Fabricageklasse** — gefreesd, gedraaid, plaatwerk, giet- of spuitgietwerk, 3D-geprint — geschat uit beeld en geometrie of opgegeven door de gebruiker. Ze bepaalt de priors: lossingshoeken, constante wanddikte en buigradii (plaatwerk), rotatiesymmetrie (draaiwerk).

### 6.7 Stap E — Structuur: twee complementaire routes

**E1 — Bottom-up: van oppervlakken naar B-rep naar features**

1. **Aangrenzingsgraaf** van de segmenten opstellen.
2. **Snijkrommen** per aangrenzend paar berekenen uit de gefitte, analytische oppervlakken (OCCT `GeomAPI_IntSS`; vlak–vlak geeft een lijn, vlak–cilinder een ellips of lijnen) en bijsnijden tot het deel bij de waargenomen grens.
3. **Hoekpunten** bepalen waar drie of meer oppervlakken samenkomen: het punt met de kleinste afstand tot alle betrokken oppervlakken.
4. **Solid opbouwen.** Per oppervlak buiten- en binnencontouren (wires) in het parameterdomein → faces (`BRepBuilderAPI_MakeFace`) → aan elkaar naaien (`BRepBuilderAPI_Sewing`) → solid (`BRepBuilderAPI_MakeSolid`) → reparatie (`ShapeFix_Shape`).
5. **Puur vlakke objecten.** Hier is een hypothese-en-selectieaanpak bijzonder robuust (PolyFit: kandidaatfaces uit alle vlakdoorsnijdingen, selectie via integer programming op waterdichtheid en fit).
6. **Afrondingen en afschuiningen eerst weglaten.** De blendband wordt uitgesloten, zodat een scherpe basis-B-rep ontstaat. Daarna worden `BRepFilletAPI_MakeFillet` en `BRepFilletAPI_MakeChamfer` toegepast met de geschatte straal of breedte. Zo modelleert een ontwerper ook, en het levert een schonere feature tree op.
7. **Feature-herkenning op de B-rep.** Regels plus een graafnetwerk op de face-adjacency-graaf (BRepNet/UV-Net-klasse) herkennen extrusie en uitsparing, gaten (enkelvoudig, verzonken, cilindrisch verzonken, tapgat), kamer, sleuf, nok, rib, afronding, afschuining en patroon. Zo ontstaat de feature tree.

Voor extrusies en omwentelingen is er een directe variant: snijd de puntenwolk loodrecht op de extrusierichting (of radiaal rond de omwentelingsas uit §6.5), fit per doorsnede 2D-lijnen en -bogen, en los een schets met constraints op (coïncidentie, raaklijn, loodrecht, gelijke maat, symmetrie) met een geometrische constraint-solver: PlaneGCS (FreeCAD, LGPL) voor het MVP, eventueel Siemens D-Cubed later. Recent werk (MiCADangelo, NeurIPS 2025) bevestigt dat doorsneden naar schetsen mét constraints een sterke route zijn.

**E2 — Top-down: CAD-programmasynthese**

1. **Model:** een puntenwolk-encoder plus een klein taalmodel (1–3B parameters, open gewichten met een permissieve licentie, bijv. Qwen2.5-1.5B onder Apache-2.0), gefinetuned om CadQuery-code te genereren. Dat is het principe van CAD-Recode (ICCV 2025) en cadrille (ICLR 2026), die lieten zien dat dit voor schets-extrusie-onderdelen werkt. Hun gewichten en trainingsdata zijn niet-commercieel (§5.4), dus we trainen op eigen procedurele data.
2. **Conditionering op de metingen.** Het model krijgt naast de puntenwolk de primitieventabel uit stap C en D, bijvoorbeeld *vlak z = 12,00 ± 0,03; cilinder Ø 8,00, as ∥ Z op (20,0; 15,0)*.
3. **Metrische binding.** Numerieke waarden in de code moeten verwijzen naar gemeten of gesnapte waarden (`meas["cyl_3"].diameter`); een getal zonder meetbasis maakt de kandidaat ongeldig. Zo kan het model structuur voorstellen, maar geen maten hallucineren.
4. **Sampling.** 8–32 kandidaatprogramma's, elk uitgevoerd in een sandbox; ongeldige kandidaten vallen af. Gegenereerde code is onbetrouwbare invoer: gVisor of Firecracker, geen netwerk, tijd- en geheugenlimieten.
5. **Parameterverfijning** per kandidaat tegen scan en beeldranden, met een gradiëntvrije optimalisator (CMA-ES of Nelder–Mead: enkele tientallen parameters, een regeneratie in OCCT kost ~0,05–0,5 s) of via een differentieerbare benadering.

E1 kan met willekeurige topologie overweg, maar levert pas na de herkenningsstap een feature tree. E2 levert direct een natuurlijke ontwerphistorie, maar is beperkt tot de operaties uit de trainingsdata. Samen dekken ze het in-scope spectrum.

### 6.8 Stap F en G — Analysis-by-synthesis, verfijning en validatie

**Score per hypothese** (lager is beter), onder de harde eis dat *H* een geldige solid is:

```math
S(H) = w_1\,E_\text{scan}(H) + w_2\,E_\text{rand}(H) + w_3\,C(H) - w_4 \log P_\text{prior}(H)
```

Hierin is *E*<sub>scan</sub> de tweezijdige, *σ*-genormaliseerde afstand scan↔model; *E*<sub>rand</sub> de reprojectiefout van de modelranden ten opzichte van de beeldranden; *C* de complexiteit (aantal features en parameters, MDL); en *P*<sub>prior</sub> de plausibiliteit (fabricageklasse, standaardmaten).

- **Model-gebaseerde fotogrammetrische verfijning.** De winnende hypothese wordt verfijnd door haar geprojecteerde randen en silhouetten in alle beelden (met het meeste gewicht op de stills) op de subpixel-beeldranden te leggen, via differentieerbaar renderen (PyTorch3D of een eigen rasterizer). Dit is bundle adjustment op CAD-niveau: het model zelf wordt het meetinstrument, waardoor de afrondingsbias van MVS verdwijnt (§7.3).
- **Alternatieven tonen.** Liggen de twee beste scores dicht bij elkaar, dan krijgt de gebruiker een gerichte keuze: *"scherpe rand of R0,5?"*.
- **Validatie (G):**
  - *kernel:* `BRepCheck_Analyzer` meldt geen fouten; de schil is gesloten, het volume positief, er zijn geen zelfdoorsnijdingen (`BOPAlgo_ArgumentAnalyzer`) en de toleranties zijn consistent;
  - *export:* STEP-round-trip (schrijven → inlezen → volume, oppervlak en aantal faces vergelijken), optioneel een conformiteitscheck met de NIST STEP File Analyzer and Viewer;
  - *metrologie:* afwijkingskaart, RMS/P95 en het percentage binnen tolerantie; faces zonder scandekking worden gelabeld als *afgeleid*.
- **Terugvallen.** Faalt een niveau, dan valt de pipeline automatisch terug op het volgende outputniveau, mét opgave van de reden.

### 6.9 Stap H — Export, integratie en outputniveaus

| Niveau | Output | Wanneer |
|---|---|---|
| **3 — Parametrisch** | Feature tree (CadQuery/build123d, FreeCAD, API-koppeling) + STEP AP242 | E1 met featureherkenning of E2 slaagt en valideert |
| **2 — Schone B-rep** | STEP AP242 met analytische oppervlakken, zonder historie | Topologie is in orde, maar de feature tree is niet betrouwbaar |
| **1 — Hybride B-rep** | Analytische oppervlakken + NURBS-patches (auto-surfacing, §6.10) | Vrije-vormdelen |
| **0 — Mesh + referentiegeometrie** | STL/3MF/OBJ + gefitte vlakken, assen en cilinders als referentiegeometrie (STEP) | Opbouw van een B-rep faalt; de gebruiker modelleert zelf, met exacte referenties |

- **STEP AP242** via `STEPCAFControl_Writer` (XDE), zodat namen, kleuren per featuretype en later semantische PMI meegaan. Eenheid mm, of inch bij een gedetecteerd imperiaal ontwerp.
- **FreeCAD `.FCStd`:** een PartDesign-body met schetsen en constraints.
- **Onshape:** de feature tree via de REST API (schetsen en features als JSON) of FeatureScript. **Fusion:** een add-in die de feature tree via de Fusion API opbouwt.
- **Code-CAD** — zo ziet de parametrische output eruit (verkort voorbeeld, CadQuery):

```python
# Gegenereerd door Cam-to-CAD — scan 7f3c…, pipeline v1.4.2
import cadquery as cq

# --- Parameters: ontwerpwaarde (gemeten waarde ± U95, reden) ---
lengte      = 80.0  # 79.94 ± 0.08 mm — gesnapt: hele mm
breedte     = 40.0  # 40.03 ± 0.07 mm — gesnapt: hele mm
dikte       = 12.0  # 11.98 ± 0.05 mm — gesnapt: hele mm
gat_d       = 6.6   #  6.63 ± 0.04 mm — ISO 273 doorgangsgat M6 (middel)
gat_steek   = 60.0  # 59.97 ± 0.06 mm — gesnapt: hele mm
afronding_r = 3.0   #  2.9  ± 0.3  mm — standaardstraal

beugel = (
    cq.Workplane("XY")
    .box(lengte, breedte, dikte)
    .edges("|Z").fillet(afronding_r)
    .faces(">Z").workplane()
    .pushPoints([(-gat_steek / 2, 0), (gat_steek / 2, 0)])
    .hole(gat_d)
)

cq.exporters.export(beugel, "beugel.step")
```

### 6.10 Vrije-vormoppervlakken (niveau 1)

Organische delen, zoals een ergonomische handgreep, krijgen **auto-surfacing**:

1. krommingsgerichte quad-remeshing (Instant Meshes/QuadriFlow-klasse);
2. een patch-indeling langs featurelijnen;
3. per patch een B-spline-oppervlak (kleinste kwadraten + gladheidsterm, G1-continuïteit over de patchgrenzen);
4. in OCCT (`GeomAPI_PointsToBSplineSurface`, `GeomPlate`) aan elkaar naaien tot een solid.

Analytische regio's blijven analytisch (hybride B-rep). Dit levert geen parametrisch model op, maar wel een nette, bewerkbare solid.

### 6.11 Methodes voor primitieve-herkenning vergeleken

| Methode | Soort | Sterk in | Zwak in | Rol |
|---|---|---|---|---|
| Efficient RANSAC (Schnabel et al., 2007; CGAL) | klassiek | robuust, geen training, vijf primitieftypes | parametergevoelig, over-segmentatie bij ruis | baseline en fallback (B/C) |
| Region growing op normalen en kromming | klassiek | snel, scherpe grenzen bij weinig ruis | ruis, geleidelijke overgangen (afrondingen) | initialisatie en controle (B) |
| Kinematische analyse (Pottmann & Randrup, 1998) | klassiek (lijngeometrie) | extrusie, omwenteling en helix in één analyse | vraagt goede normalen | structuur (C/E) |
| GlobFit (Li et al., 2011) | klassiek (optimalisatie) | ontwerpintentie, consistentie | vraagt een goede initiële segmentatie | relaties (D) |
| PolyFit (Nan & Wonka, 2017) | klassiek (optimalisatie) | waterdichte polyhedra uit vlakken | alleen vlakken | prismatische delen (E1) |
| ParSeNet / HPNet / SED-Net | geleerd (type + embedding per punt) | robuust bij ruis, typeherkenning, randen | domeinkloof synthetisch → echt | segmentatie (B) |
| Point2CAD / ComplexGen / Split-and-Fit | geleerd + geometrie | volledige B-rep-topologie | gevoelig voor ontbrekende data | topologie (E1), als bouwsteen of referentie |
| BRepNet / UV-Net | geleerd op de B-rep-graaf | featureherkenning op een B-rep | vraagt een geldige B-rep | featureherkenning (E1) |
| CAD-Recode / cadrille-principe | geleerd (programmasynthese) | ontwerphistorie, intentie | beperkte operatieset; hallucinatie van getallen | E2, met metrische binding |

**Onderzoeksradar 2025/2026** — om te volgen, niet om blind op te bouwen:

| Werk | Venue | Relevantie |
|---|---|---|
| CAD-Recode | ICCV 2025 | Puntenwolk → CadQuery-code met een LLM (Qwen2-1.5B), getraind op 1M procedurele sequenties |
| cadrille | ICLR 2026 | Punten, beelden of tekst → CadQuery; supervised training + online reinforcement learning |
| MiCADangelo | NeurIPS 2025 | Doorsneden van de scan → schetsen mét constraints → extrusies |
| BrepGaussian | CVPR 2026 | Multi-view beelden → Gaussian Splatting met geleerde features → B-rep |
| CADDreamer | CVPR 2025 | Eén beeld → primitief-bewuste multi-view diffusie → waterdichte B-rep |
| Img2CAD | SIGGRAPH Asia 2025 | Een VLM voorspelt de opbouw, een tweede netwerk de parameters |

### 6.12 Trainingsdata en ML-strategie

**Data-engine "render-and-reconstruct"** — de belangrijkste investering, omdat er geen commercieel bruikbare kant-en-klaarmodellen zijn (§5.4):

1. **CAD-bronnen:** procedureel gegenereerde onderdelen (schets-extrusie, revolutie, gaten, afrondingen, patronen, plaatwerk) als basis; publieke CAD-collecties (ABC, Fusion 360 Gallery, DeepCAD) alleen na controle van de gebruiksvoorwaarden.
2. **Renderen** per model: fotorealistische multi-view beeldreeksen met een smartphone-cameramodel (ruis, JPEG, lensvervorming, vignettering, bewegingsonscherpte, auto-exposure), willekeurige materialen (mat, glanzend, textuurloos), de kalibratiemat en variërende belichting (Blender/BlenderProc).
3. **Door onze eigen reconstructiepipeline halen.** Zo ontstaan realistische artefacten — afgeronde randen, gaten, ruis, kleine uitlijnfouten — gekoppeld aan exacte labels uit de B-rep: type, instantie, rand en parameters per punt, plus de featurereeks.
4. **Goedkopere schaalvergroting:** een geleerd degradatiemodel, getraind op paren (schone CAD-sample, echte reconstructie), genereert realistische ruis direct op puntenwolken. Het dure renderen blijft dan beperkt tot een representatieve subset.

**Echte data:**

- Een referentieset van 200–500 fysieke onderdelen met bekende CAD (geprint of gefreesd vanuit CAD, plus standaarddelen), nagemeten met schuifmaat, CMM of structured-light.
- Na opt-in: gebruikersdata mét correcties. Actief leren geeft prioriteit aan gevallen met een lage betrouwbaarheid.

**Modellen:**

| Model | Architectuur | Omvang (indicatief) | Training |
|---|---|---|---|
| 3D-segmentatie | PTv3-familie + HPNet/SED-Net-koppen | ~50M parameters | Synthetisch, daarna finetunen op echte data |
| 2D-cues | SAM 2, een randdetector, MoGe-2/StableNormal | bestaand | Randdetector finetunen op onderdeelranden |
| CAD-taalmodel (E2) | puntenwolk-encoder + LLM van 1–3B (permissieve licentie) | 1–3B | Supervised finetuning op procedurele code, daarna RL met geometrische beloning (IoU/Chamfer + geldigheid) |
| Featureherkenning (E1) | BRepNet/UV-Net-klasse | ~10M | Gelabelde B-reps uit de procedurele generator |

---

## 7. Edge cases en technische knelpunten

Hieronder de drie grootste technische knelpunten, elk met een gelaagde mitigatiestrategie: detectie → opname → algoritme → CAD-priors → UX-fallback.

### 7.1 Knelpunt 1 — Glanzende, spiegelende, donkere en transparante oppervlakken

**Probleem.** Fotogrammetrie en MVS veronderstellen dat een punt er vanuit elke richting hetzelfde uitziet (Lambertiaans). Spiegelende hooglichten bewegen mee met de camera. Dat geeft foute matches, gaten en "spookgeometrie": reflecties die als geometrie achter het oppervlak worden gereconstrueerd. Transparante objecten tonen de achtergrond, vervormd door breking; zeer donkere oppervlakken leveren weinig signaal. Juist machinaal bewerkt metaal (aluminium, staal, chroom) hoort bij de kerndoelgroep.

**Mitigatie**

1. **Vroeg detecteren.** De pre-scan combineert een on-device materiaalclassifier met een multi-frame hooglichttest: sterk kijkhoekafhankelijke intensiteit op hetzelfde 3D-punt. Het resultaat is een glanskaart per regio en direct advies, vóórdat de gebruiker vijf minuten scant.
2. **Tijdelijk matteren.** Sublimerende scanspray verdwijnt na enkele uren zonder residu; de laagdikte ligt in de orde van micrometers, verwaarloosbaar ten opzichte van onze nauwkeurigheidsklasse. Dit is de standaard in industriële 3D-scanning en de betrouwbaarste route.
3. **Kruispolarisatie** (optionele accessoire): een polarisatiefilter op de lens en polarisatiefolie voor een lichtbron, 90° gedraaid, blokkeert de spiegelende reflectie. Dit werkt goed op glanzende kunststof, lak en keramiek. **Op blank metaal** ontbreekt de diffuse component grotendeels, zodat het oppervlak donker wordt; daar blijft spray de betrouwbare route.
4. **Hooglichten als signaal in plaats van ruis.** De flitser staat vrijwel naast de lens. Een spiegelend hooglicht verschijnt daarom precies waar de oppervlaknormaal naar de camera wijst (**n** ∥ kijkstraal): elk gedetecteerd hooglichtpixel is een normaalmeting op een bekende 3D-straal. Bij gedraaide of gefreesde cilinders en vlakken liggen die hooglichten op lijnen langs de beschrijvende lijnen, en die leggen as en straal sterk vast. In de flitsermodus maakt de app bij elke still (stop-and-shoot, §5.1) een paar opnamen, met en zonder flitser; zo levert elke still deze metingen. Ze worden als extra residuen meegenomen in de primitieve-fit (§6.5).
5. **Reflectiebewuste reconstructie.** Per view worden hooglichtmaskers uitgesloten van MVS. In de Precisie-tier komt daar optioneel een reflectiebewuste neurale oppervlaktereconstructie bij (NeRO-klasse), die kijkhoekafhankelijke kleur scheidt van geometrie.
6. **Silhouetten zijn glansongevoelig.** SAM 2-maskers leveren visual-hull-constraints en silhouetuitlijning van de CAD-kandidaten (§6.8). Bij machinale delen bevatten silhouetten een groot deel van de maatinformatie.
7. **CAD-priors.** Glanzende delen zijn vrijwel altijd machinaal bewerkt, dus met een sterke primitieven-regularisatie: weinig betrouwbare punten volstaan voor een correct vlak of een correcte cilinder.
8. **Donkere oppervlakken:** belichting meten op het objectmasker, flitser, RAW met multi-frame ruisonderdrukking.
9. **Transparant:** in v1 buiten de automatische scope. De app detecteert het en vraagt om spray. Brekingsbewuste reconstructie (NeTO-klasse) is een onderzoekstraject.

**Haalbaarheid.** Maatregelen 1–3 en 5–8 zijn productierijp in 2026. "Hooglichten als signaal" en reflectiebewuste neurale reconstructie vragen engineering (Fase 2). Transparant zonder coating blijft onderzoek.

### 7.2 Knelpunt 2 — Textuurloze, uniforme en symmetrische objecten

**Probleem.** SfM en MVS hebben unieke lokale textuur nodig. Wit spuitgietwerk, gelakte of gepoedercoate delen en egaal gietwerk hebben die niet. Het gevolg: tracking valt weg (ARCore meldt te weinig features), poses worden fout en MVS laat gaten in vlakke delen. Symmetrische objecten — een gladde as, een vierkant blok — veroorzaken bovendien pose-ambiguïteit, vooral na het omdraaien.

**Mitigatie**

1. **Ontkoppel de camerabepaling van het object.** De mat met ChArUco/AprilTag-markers en willekeurige textuur levert vrijwel alle pose-informatie. Het object zelf hoeft voor de SfM geen textuur te hebben.
2. **Andere cues dan textuur:**
   - **Silhouetten:** SAM 2-maskers → visual hull. Textuurloze objecten hebben tegen de mat meestal een scherp contrasterend silhouet.
   - **Flitserfotometrie:** een uniforme albedo is precies het ideale geval voor fotometrische stereo. *De zwakte van MVS is de kracht van fotometrie.* De app neemt bij elke still een paar opnamen met en zonder flitser (§5.1); het verschilbeeld isoleert het flitsdeel, met een bekende lichtpositie (bij de camera) en een bekende 1/*r*²-afval (via de diepte). Inverse rendering in de stijl van IRON en WildLight reconstrueert daaruit normalen en vorm, juist waar MVS blind is. Het flitserprofiel wordt per toestelmodel gekalibreerd op de witte vlakken van de mat.
   - **Monoculaire normaal- en diepte-priors** (MoGe-2, StableNormal) als regularisatie in de oppervlakte-GS (het MonoSDF-principe): goedkoop en effectief voor vlakke en gebogen textuurloze delen.
   - **Geleerde dichte matchers** (RoMa) en feed-forward modellen voor robuuste poses als markers gedeeltelijk zijn afgedekt.
3. **Tijdelijke textuur als fallback bij Precisie:** verwijderbare gecodeerde markerstickers (6–10 stuks, 3–6 mm) of afwasbare spikkelspray. De app herkent markers en vult de geometrie onder de stickers later in via de primitieve-fits.
4. **Primitief eerst.** Textuurloze vlakken zijn vrijwel altijd eenvoudig: een vlak of een cilinder. Een vlak heeft alleen zijn begrenzende randen en een paar betrouwbare punten nodig; de CAD-stap trekt het primitief over de MVS-gaten heen. Een gat in de mesh is geen gat in het model.
5. **Symmetrie-ambiguïteit.** Elke sessie wordt ten opzichte van de mat geregistreerd. Bij het samenvoegen van de omgedraaide sessie gebruikt de pipeline geometrische registratie met meerdere hypothesen (FPFH/RANSAC of geleerde registratie, gevolgd door ICP). Passen meerdere registraties even goed (symmetrie), dan beslissen niet-symmetrische details, zoals een enkel gat. Ontbreken die, dan vraagt de app om één markersticker.

**Haalbaarheid.** Maatregelen 1, 2 (silhouetten, priors, matchers), 3, 4 en 5 zijn productierijp of vragen alleen engineering. Inverse rendering met de flitser is engineering met een onderzoekscomponent (Fase 2), met de monoculaire priors als fallback.

### 7.3 Knelpunt 3 — Randnauwkeurigheid, kleine features en een topologisch correcte B-rep

**Probleem.** Elke reconstructiemethode werkt als een laagdoorlaatfilter: MVS-vensters, Poisson-smoothing en de footprint van Gaussians ronden scherpe randen af met een straal van enkele pixels objectresolutie. Kleine features (gaten < ~3 mm, afschuiningen, dunne wanden) vervagen of verdwijnen. Toch zitten juist daar de functionele maten: passingen en aansluitende randen. Daarbij is een B-rep onverbiddelijk: één ontbrekende of foute rand en er is geen gesloten solid.

**Mitigatie**

1. **Randen worden berekend, niet gemeten.** Een scherpe rand is de doorsnijding van twee robuust gefitte oppervlakken, gebaseerd op duizenden punten ver van de rand. Punten in de randband worden uitgesloten van de fit (§6.5), zodat de systematische afrondingsbias verdwijnt.
2. **Randen uit de beelden, niet uit de mesh.** Subpixel-randdetectie in de hoge-resolutiebeelden (~0,07–0,1 mm/px), gevolgd door multi-view triangulatie van 3D-krommen (LiMAP/EMAP/NEF-klasse). Die 3D-randkrommen worden extra constraints in stap C en D: de snijlijn van twee vlakken moet samenvallen met de getrianguleerde beeldrand.
3. **PSF-bewuste classificatie: scherp of afgerond?** De pipeline schat per scan hoe sterk de reconstructie een scherpe rand uitsmeert (de *edge spread*), uit MVS-venstergrootte × objectresolutie, en valideert dat op bekende scherpe randen — bijvoorbeeld een meetblokje uit de Pro-kit. Een afronding met een straal tot ~1,5× die spreiding telt als scherpe rand. Daarboven is het een echte afronding: de straal wordt geschat via het krommingsprofiel dwars op de rand (een rollende bol met constante straal) en gesnapt naar standaardwaarden. Afschuiningen zijn vlakke stroken met een hoek (vaak 45°) en een breedte.
4. **Model-gebaseerde fotogrammetrische verfijning — de sleutelinnovatie.** Na de keuze van de CAD-hypothese worden alle parameters verfijnd door de geprojecteerde randen en silhouetten van het model in alle views — keyframes én stills — op de subpixel-beeldranden te leggen (§6.8). Dit is het CAD-equivalent van bundle adjustment: honderden views × duizenden randpixels geven randposities met een precisie onder één pixel objectresolutie.
5. **Topologie via constructie.** Waar mogelijk worden solids opgebouwd via features en booleaanse bewerkingen (extrusie, revolutie, uitsparing) in plaats van faces aan elkaar te naaien. Zo zijn ze geldig per constructie. Face-naaien blijft over voor de rest; meerdere topologiehypothesen worden via analysis-by-synthesis getoetst.
6. **Kleine features.** Een detailpass met minimale objectresolutie (§4.1, stap 6). Features kleiner dan ~3× de objectresolutie worden gemarkeerd als *onzeker*, en de app vraagt om een schuifmaatwaarde of een bevestiging: *"Gat Ø 2,5 → M3-tapgat?"*
7. **Gerichte vragen in plaats van gokken.** Liggen twee hypothesen dicht bij elkaar (scherp of R0,5), dan stelt de app één vraag met één tik.

**Haalbaarheid.** Maatregelen 1, 3, 5, 6 en 7 zijn productierijp met gedegen engineering. Maatregelen 2 en 4 vragen meer engineering (Fase 2), maar bouwen op beproefde technieken: subpixel-randen, bundle adjustment en differentieerbaar renderen.

### 7.4 Overige risico's

| Risico | Impact | Mitigatie |
|---|---|---|
| Schaal- en intrinsics-fouten (focus-breathing, OIS, printschaal) | systematische maatfouten | intrinsics per focusgroep, OIS/EIS uit, matverificatie, *χ*²-consistentietoets (§4.5) |
| Verborgen geometrie (onderkant, diepe gaten, interne holtes, schroefdraad) | onvolledig model | omdraai-workflow; labels *afgeleid*; inferentie of een gat doorgaand is, plus een vraag; schroefdraad als cosmetische feature (tapboormaat → M-maat) |
| Dunne wanden en kleine delen | niet te reconstrueren | detailpass, macro- of telelens, minimale objectresolutie, schuifmaatinvoer |
| Object beweegt tijdens de scan | registratiefouten | objectpose bewaken ten opzichte van de mat; sessie splitsen |
| Android-fragmentatie | wisselende kwaliteit; een hoge-resolutiestream naast ARCore is niet gegarandeerd | capability-matrix, toesteltiers, stop-and-shoot-stills, testlab |
| Hallucinatie in de CAD-synthese | foute maten of opbouw | metrische binding, verificatiepoort, alternatieven tonen |
| Rekenkosten en doorlooptijd | marge, UX | tiers, spot-GPU's, caching per stap, incrementele herberekening |
| IP en privacy | vertrouwen, AVG | EU-residentie, versleuteling, opt-in voor training, on-premises-optie |
| Licenties en octrooien | juridisch | §5.4; freedom-to-operate-analyse |
| Reverse engineering van beschermde ontwerpen | aansprakelijkheid | gebruiksvoorwaarden; bewustwording bij de gebruiker |

---

## 8. Validatie, metrics en teststrategie

**Referentieset ("golden set").** 50 fysieke onderdelen in Fase 0 en 300 voor v1, verdeeld over klassen (prismatisch, gedraaid, plaatwerk, spuitgietwerk), materialen (mat, glanzend, textuurloos) en formaten (2–50 cm). Grondwaarheid: het CAD-model (voor onderdelen die uit CAD zijn geprint of gefreesd) plus een referentiemeting (CMM of structured-light-scanner met ≤ 0,02–0,05 mm, schuifmaat voor de kernmaten).

**Metrics**

| Categorie | Metric |
|---|---|
| Maatnauwkeurigheid | fout per feature \|*x* − *x*<sub>ref</sub>\|, per modus en objectklasse |
| **Onzekerheidskalibratie** | het aandeel werkelijke fouten binnen de gerapporteerde U95 moet ≈ 95% zijn — een eerlijke onzekerheid is net zo belangrijk als een kleine |
| Oppervlak | tweezijdige Chamfer- en Hausdorff-afstand, P95-afwijking |
| Structuur | nauwkeurigheid van het primitieftype, segmentatie-mIoU, rand-F1, topologische correctheid (aantal faces en edges, genus), edit-afstand van de feature tree tot de referentie, snap-nauwkeurigheid |
| Product | % bruikbaar zonder correctie, correctietijd, time-to-CAD, verdeling van faalredenen, doorlooptijd p50/p95, kosten per scan |

**Testautomatisering**

- **Opnamelogica, deterministisch:** de kwaliteitsengine en de opnamesturing draaien in de gedeelde C++-kern en worden getest door opgenomen Scan Bundles (beelden, IMU, poses) offline opnieuw af te spelen.
- **AR-laag op echte toestellen:** ARCore Recording & Playback in een device farm. Twee beperkingen: sessies met Shared Camera kunnen niet worden afgespeeld, en playback gebeurt in real time en is niet deterministisch. Deze tests draaien dus zonder Shared Camera en met toleranties.
- **Pipeline-regressie:** elke nacht de volledige referentieset door elke nieuwe pipeline- en modelversie halen, met dashboards per metric; nieuwe versies eerst als canary op een deel van het verkeer.
- **Reproduceerbaarheid:** elk resultaat is gekoppeld aan (bundle-hash, pipelineversie, modelversies, configuratie), zodat herverwerking mogelijk is zodra modellen verbeteren.

---

## 9. Roadmap, team en open beslissingen

| Fase | Duur (indicatief) | Resultaat | Go/no-go-criterium |
|---|---|---|---|
| **0 — Haalbaarheid** | 8–10 weken | Opnameprototype (ARCore + Shared Camera + mat + auto-capture); offline pipeline (ALIKED/LightGlue + COLMAP + Ceres-BA met markers; MVS + oppervlakte-GS op gsplat; RANSAC-fitting + OCCT voor vlakken en cilinders); referentieset van 50 onderdelen | Precisie haalt ±(0,3 mm + 0,15% · L) op ≥ 80% van de goed zichtbare features van matte onderdelen; geldige STEP voor ≥ 70% van de prismatische onderdelen |
| **1 — MVP** | 4–6 maanden | Volledige geleide UX (dekking, next-best-view, omdraaien); cloudpipeline (Standaard-tier); route E1 + ontwerpintentie; STEP AP242 + CadQuery + FreeCAD; review/editor; meetrapport | ≥ 60% bruikbaar zonder correctie; p50-doorlooptijd ≤ 20 min |
| **2 — Intelligentie** | 6–9 maanden | Geleerde segmentatie (synthetische data-engine); route E2 met metrische binding; afrondingen, afschuiningen en patronen; model-gebaseerde beeldverfijning; flitsermodus; koppelingen met Onshape en Fusion; Precisie-tier | ≥ 80% bruikbaar zonder correctie; U95-dekking tussen 90% en 97% |
| **3 — Verbreding** | 9–18 maanden | Auto-surfacing voor vrije vorm, plaatwerk, samenstellingen, iOS (ARKit + LiDAR), on-premises voor enterprise, PMI/GD&T in AP242 | per segment te bepalen |

**Kernteam (Fase 1–2)**

- 2–3 engineers computer vision / 3D-reconstructie
- 1–2 engineers geometrie en CAD-kernel (OCCT, constraint-solving)
- 2 ML-engineers (segmentatie, programmasynthese, data-engine)
- 2 Android-engineers (camera en AR, C++/NDK)
- 1–2 backend- en platformengineers (Kubernetes, Temporal, security)
- 1 UX-designer (AR-interactie)
- 0,5–1 FTE metrologie en QA (referentieset, meetprotocollen)

**Open beslissingen voor stakeholders**

1. **Eerste markt:** B2B (MKB/MRO) of prosumer? Dit bepaalt de behoefte aan on-premises, het prijsmodel en de accessoires.
2. **Accessoires:** worden de Pro-mat, markerstickers en scanspray onderdeel van het aanbod?
3. **CAD-koppelingen:** in welke volgorde Onshape, Fusion en SolidWorks? Bij voorkeur te beslissen op basis van klantinterviews.
4. **Data:** het beleid voor het gebruik van klantdata voor training (opt-in, anonimisering, B2B-contracten).
5. **iOS-timing:** LiDAR en Apple Object Capture verlagen de drempel aanzienlijk.
6. **Build of buy** voor de reconstructiekern in het MVP (§5.3).
7. **Deploymentmodel:** lokaal-eerst en volledig open source (geen cloudkosten), of een gehoste dienst? Zie §5.5 en [OPEN-SOURCE-LOKAAL.md](OPEN-SOURCE-LOKAAL.md).

---

## 10. Bijlagen

### 10.1 Scan Bundle-formaat

```text
scan_<uuid>/
  manifest.json              schemaversie, toestelprofiel, modus, sessies, hashes
  annotations.json           gebruikersinvoer: referentiematen, intentie, materiaalhints
  sessions/<n>/
    frames.jsonl             per beeld: tijd, pose, intrinsics, focus, belichting, kwaliteit
    keyframes/000123.jpg     doorlopende keyframes (ARCore-CPU-beeld, ~1080p)
    stills/000031.jpg        stills op volle resolutie (of .heic; .dng in Precisie-modus)
    depth/000123.png         16-bit raw depth (mm) + confidence/000123.png
    masks/000123.png         objectmasker (on-device; in de cloud verfijnd)
    imu.bin                  gyroscoop en versnellingsmeter, 200-400 Hz, gedeelde tijdbasis
    markers.json             gedetecteerde ChArUco/AprilTag-hoeken (subpixel) per frame
    arcore_recording.mp4     optioneel: ARCore-opname voor debugging en regressie
```

Voorbeeld van één regel in `frames.jsonl` (hier over meerdere regels weergegeven):

```json
{
  "frame": 31,
  "t_ns": 81234567890123,
  "timestamp_source": "REALTIME",
  "session": 1,
  "camera": "main",
  "stream": "still",
  "capture_mode": "stop_and_shoot",
  "pose_world_from_cam": [0.9981, -0.0123, 0.0602, 0.1432,
                          0.0151, 0.9990, -0.0421, 0.0217,
                          -0.0596, 0.0429, 0.9973, 0.2981,
                          0.0, 0.0, 0.0, 1.0],
  "intrinsics": {"fx": 2812.4, "fy": 2812.4, "cx": 2016.3, "cy": 1511.8,
                 "width": 4032, "height": 3024, "intrinsics_group": "main_f3.35"},
  "focus_distance_diopter": 3.35,
  "exposure_time_ns": 4000000,
  "iso": 200,
  "ois": "off",
  "predicted_blur_px": 0.21,
  "pose_source": "arcore_last_before_pause",
  "coverage_gain": 0.031,
  "image": "stills/000031.jpg",
  "mask": "masks/still_000031.png"
}
```

### 10.2 Woordenlijst

| Term | Betekenis |
|---|---|
| **B-rep** (boundary representation) | Solid beschreven door zijn begrenzing: faces op onderliggende oppervlakken, begrensd door edges en vertices, met topologische samenhang |
| **NURBS / B-spline** | Vrije-vormkrommen en -oppervlakken; de CAD-standaard voor niet-analytische vormen |
| **Feature tree** | Geordende reeks modelleeroperaties (schets, extrusie, gat, afronding…) met parameters en constraints — "de parametrie" |
| **SfM** (Structure-from-Motion) | Bepaalt cameraposities en een ijle 3D-puntenwolk uit overeenkomende beeldpunten |
| **MVS** (Multi-View Stereo) | Berekent dichte diepte of punten bij bekende camera's |
| **Bundle adjustment (BA)** | Gezamenlijke kleinste-kwadratenoptimalisatie van camera's, intrinsics en 3D-punten |
| **VIO** (visual-inertial odometry) | Real-time camerabepaling uit beeld + IMU; de basis van ARCore-tracking |
| **GSD / objectresolutie** | Afmeting op het object die één pixel bestrijkt (mm/pixel) |
| **Edge spread / PSF** | De mate waarin het meetsysteem een scherpe rand uitsmeert |
| **TSDF** | *Truncated signed distance field*: volumetrische fusie van dieptebeelden |
| **Next-best-view (NBV)** | De opnamepositie met de grootste verwachte informatiewinst |
| **PMI** | *Product and manufacturing information*: maten, toleranties en GD&T in het CAD-model |
| **AP242** | STEP-applicatieprotocol ISO 10303-242 voor 3D-modellen, inclusief PMI |
| **GRIC / BIC / MDL** | Criteria voor modelselectie die de fit afwegen tegen de modelcomplexiteit |
| **Manhattan-frame** | Drie onderling loodrechte dominante richtingen in een object of scène |

### 10.3 Referenties (selectie)

**Reconstructie en camerabepaling**

- Schönberger & Frahm, *Structure-from-Motion Revisited* (COLMAP), CVPR 2016 · Pan et al., *Global Structure-from-Motion Revisited* (GLOMAP), ECCV 2024
- Lindenberger et al., *LightGlue*, ICCV 2023 · Zhao et al., *ALIKED*, IEEE TIM 2023 · Edstedt et al., *RoMa*, CVPR 2024
- Wang et al., *VGGT: Visual Geometry Grounded Transformer*, CVPR 2025 · Keetha et al., *MapAnything*, 3DV 2026 · Duisterhof et al., *MASt3R-SfM*, 3DV 2025
- Kerbl et al., *3D Gaussian Splatting*, SIGGRAPH 2023 · Huang et al., *2D Gaussian Splatting*, SIGGRAPH 2024 · Yu et al., *Gaussian Opacity Fields*, SIGGRAPH Asia 2024 · Chen et al., *PGSR*, IEEE TVCG 2024
- Wang et al., *NeuS*, NeurIPS 2021 · Li et al., *Neuralangelo*, CVPR 2023 · Yu et al., *MonoSDF*, NeurIPS 2022
- Ravi et al., *SAM 2*, 2024 · Wang et al., *MoGe-2*, NeurIPS 2025 · Ye et al., *StableNormal*, SIGGRAPH Asia 2024

**Reverse engineering en CAD-reconstructie**

- Várady, Martin & Cox, *Reverse engineering of geometric models — an introduction*, Computer-Aided Design 29(4), 1997
- Pottmann & Randrup, *Rotational and helical surface approximation for reverse engineering*, Computing 60, 1998
- Schnabel, Wahl & Klein, *Efficient RANSAC for Point-Cloud Shape Detection*, Computer Graphics Forum, 2007
- Li et al., *GlobFit*, SIGGRAPH 2011 · Nan & Wonka, *PolyFit*, ICCV 2017
- Sharma et al., *ParSeNet*, ECCV 2020 · Yan et al., *HPNet*, ICCV 2021 · Li et al., *SED-Net*, SIGGRAPH 2023
- Guo et al., *ComplexGen*, SIGGRAPH 2022 · Liu et al., *Point2CAD*, CVPR 2024 · Liu et al., *Split-and-Fit*, SIGGRAPH 2024
- Wu et al., *DeepCAD*, ICCV 2021 · Khan et al., *CAD-SIGNet*, CVPR 2024 · Rukhovich et al., *CAD-Recode*, ICCV 2025 · Kolodiazhnyi et al., *cadrille*, ICLR 2026 · Karadeniz et al., *MiCADangelo*, NeurIPS 2025
- Lambourne et al., *BRepNet*, CVPR 2021 · Jayaraman et al., *UV-Net*, CVPR 2021
- Wu et al., *Point Transformer V3*, CVPR 2024

**Randen, materialen en belichting**

- Liu et al., *3D Line Mapping Revisited* (LiMAP), CVPR 2023 · Ye et al., *NEF*, CVPR 2023 · Li et al., *3D Neural Edge Reconstruction* (EMAP), CVPR 2024
- Zhang et al., *IRON*, CVPR 2022 · Cheng et al., *WildLight*, CVPR 2023 · Liu et al., *NeRO*, SIGGRAPH 2023 · Li et al., *NeTO*, ICCV 2023

**Datasets, standaarden en tools**

- Koch et al., *ABC dataset*, CVPR 2019 · Willis et al., *Fusion 360 Gallery*, SIGGRAPH 2021
- ISO 10303-242 (STEP AP242) · ISO 273 · ISO 286 · ISO 3 · ISO 15 · ISO 3601
- OCCT · CadQuery · build123d · FreeCAD · COLMAP · Ceres Solver · gsplat · Open3D · CGAL · NIST STEP File Analyzer and Viewer
