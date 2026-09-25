# Route A — lokaal en open source: eerste werkende versie (v0.1)

Dit is de uitvoering van profiel A uit [OPEN-SOURCE-LOKAAL.md](OPEN-SOURCE-LOKAAL.md): foto's van een onderdeel op een geprinte kalibratiemat gaan naar je eigen pc, die er een CAD-model van maakt. Er is geen cloud nodig en er zijn geen licentiekosten; alle afhankelijkheden hebben een permissieve open-sourcelicentie.

| | |
|---|---|
| **Status** | v0.1 — werkt end-to-end op synthetische scans; validatie met echte foto's volgt (Fase 0) |
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

1. **Mat printen.** `camtocad mat --formaat A4` schrijft `kalibratiemat_A4.pdf`. Print op 100% (werkelijke grootte) op mat papier en meet beide lijnen van 100,0 mm na. Leg de mat vlak op een stijve ondergrond.
2. **Foto's maken.** Leg het onderdeel plat in het midden van de mat, bij diffuus licht. Maak 30–60 foto's met de gewone camera-app van je telefoon (JPG):
   - rondom, op twee à drie hoogtes (ongeveer 35°, 55° en 70° boven de mat);
   - **4–6 foto's recht van boven**: nodig voor de contour en om door gaten heen te kijken;
   - de mat steeds grotendeels in beeld, niet inzoomen, zelfde lens (geen groothoek/tele wisselen).
3. **Verwerken**, op één van twee manieren:
   - **Via de browser van je telefoon:** start `camtocad server` op de pc en open op je telefoon (zelfde wifi) de link die in de terminal verschijnt, inclusief `?token=…`. Kies of maak daar de foto's; na de verwerking staan rapport, STEP, STL en script klaar om te downloaden.
   - **Via de opdrachtregel:** kopieer de foto's naar een map en draai `camtocad scan <map>`. De uitvoer komt in `<map>_cad/`.

Opties voor `scan`: `--mat A3`, `--max-zijde 2000` (werkresolutie), `--snapdrempel 0.8`, `--inch` (snappen naar inchmaten).

## Hoe het werkt

| Stap | Module | Wat er gebeurt | Architectuur |
|---|---|---|---|
| Mat | `mat.py`, `pdf.py` | ChArUco-mat als vector-PDF op exacte schaal, met meetlijnen en stippen voor extra textuur | §4.5 |
| Camera | `calib.py` | Mat herkennen, camera zelf kalibreren (Zhang), per foto een metrische pose: de mat is de tracker | §4.5, OPEN-SOURCE-LOKAAL §2.1 |
| Maskers | `masks.py` | Voorspel per foto hoe de mat eruitziet; afwijkingen zijn object. Randpixels volgens de 50%-regel (onvertekend) | §5.3 [R3c] |
| Lokaliseren | `hull.py` | Grove visual hull (2 mm): waar ligt het object, hoe hoog is het ongeveer | §5.3 [R3c] |
| Startmodel | `silhouette.py`, `profile.py` | Bovenaanzichten terugprojecteren op het bovenvlak → contour → randen, afrondingen, gaten | §6.3–6.5 |
| Model fitten | `silhouette.py` | Analysis-by-synthesis: model-silhouet renderen in élke foto, pixelverschil minimaliseren (hoogte, randen, afrondingen, gaten) | §6.8, §7.3 |
| Ontwerpintentie | `snapping.py`, `cadmodel.py` | Werkassenstelsel met datum, randen exact haaks, Bayesiaans snappen (hele mm, ISO 273, tapboormaten, standaardstralen), steekcirkels | §6.6 |
| CAD | `cadmodel.py`, `cadhelpers.py` | OpenCascade-solid via CadQuery, STEP/STL-export, leesbaar script met benoemde maten | §6.9 |
| Rapport | `report.py` | Maten met U95 en snapreden, bovenaanzicht, samenvatting, waarschuwingen | §4.6 |

**Waarom silhouetten en geen MVS?** Voor deze objectklasse zijn silhouetten nauwkeuriger en robuuster: ze hebben geen textuur op het object nodig en zijn ongevoelig voor glans. Een visual hull alléén is niet genoeg: van bovenaf is het bovenvlak niet te zien, en de wanden worden te ruim. De maten komen daarom uit het fitten van het parametrische model op de randen in alle foto's — de "model-gebaseerde verfijning" uit het architectuurdocument. MVS (COLMAP/OpenMVS) volgt voor vrije vormen.

## Resultaten op synthetische scans

Gerenderde scans van 46 foto's (1600 × 1200 pixels, ~0,25 mm/pixel op het object), met lensvervorming, ruis en belichtingsverloop. De camera en de poses worden uit de foto's zelf bepaald.

| Onderdeel | Maat | Waarheid | Gefit | Na snappen |
|---|---|---|---|---|
| Beugel (op 17° gedraaid) | lengte × breedte × hoogte | 80 × 40 × 12 | 79,93 × 39,93 × 12,05 | 80 × 40 × 12 |
| | afrondingen (4x) | R3 | R3,04 | R3 |
| | gaten (2x) | Ø 6,6 op (10, 20) en (70, 20) | Ø 6,57 op (9,98, 19,97) en (69,97, 19,97) | Ø 6,565 (niet gesnapt: 6,5 en 6,6 beide plausibel); posities exact |
| L-vorm (niet-convex) | maten | 60 / 30 / 25 / 50, hoogte 8 | 59,93 / 29,95 / 24,93 / 49,96, hoogte 8,05 | exact |
| | afrondingen | 4 × R2, 2 scherpe hoeken | 4 × R2,17, 2 × scherp | R2,17 (niet gesnapt, binnen U95) |
| | gat | Ø 5,5 op (15, 12) | Ø 5,44 op (14,96, 11,97) | Ø 5,5 (ISO 273 M5) op (15, 12) |
| Ronde flens | diameter × hoogte | Ø 50 × 10 | Ø 50,09 × 10,03 | Ø 50 × 10 |
| | gatenpatroon | 4 × Ø 4,5 op steekcirkel Ø 35 | 4 × Ø 4,46 op Ø 34,96 | 4 × Ø 4,5 (M4) op Ø 35, parametrisch patroon |

De gefitte maten liggen binnen ±0,1 mm van de waarheid (afrondingen binnen +0,2 mm). Dat is ruim binnen het Precisie-doel van ±(0,2 mm + 0,1% · L). **Let op:** dit zijn synthetische scans. Echte foto's hebben schaduwen, autofocus, compressie en een niet perfect vlakke mat; hoe dicht v0.1 daarbij in de buurt komt, moet Fase 0 uitwijzen.

## Beperkingen van v0.1

- **Objectklasse:** alleen 2,5D-onderdelen plat op de mat, met doorgaande gaten. Geen blinde gaten, kamers, treden in de hoogte, afschuiningen op de bovenrand, schroefdraad of vrije vormen. Afschuiningen op verticale hoeken worden als afronding benaderd; buitencontouren met bogen groter dan een hoekafronding (bijv. een sleufvorm) worden met rechte randen benaderd.
- **Bovenaanzichten zijn verplicht**: zonder foto's recht van boven stopt de verwerking met een duidelijke melding.
- **Belichting:** harde schaduwen van het object op de mat kunnen de maskers verstoren. Gebruik diffuus licht, en mat papier voor de mat.
- **Donkere objecten** op de zwarte vakken geven minder randinformatie; de fit leunt dan op de overige foto's.
- **Onzekerheid (U95)** is een indicatie op basis van resolutie en aantal foto's, nog niet gekalibreerd op echte metingen (zie ARCHITECTURE.md §8).
- **Afrondingen kleiner dan ~3 pixels** (bij de demo ~0,8 mm) zijn niet te onderscheiden van scherpe hoeken en worden als scherp gemodelleerd; het rapport meldt dat.
- **Rekentijd:** ~40 s per scan van 46 foto's op een gewone CPU; foto's worden standaard teruggeschaald naar 2000 pixels.
- **Nog geen native app:** de telefoon gebruikt de browser. De geleide AR-opname uit het architectuurdocument volgt met de Android-app.

## Licenties

Alle gebruikte bibliotheken zijn open source met een permissieve licentie: NumPy en SciPy (BSD), OpenCV (Apache-2.0), CadQuery (Apache-2.0) op OpenCascade (LGPL-2.1 met uitzondering), en FastAPI/Uvicorn (MIT/BSD). Welke licentie Cam-to-CAD zelf krijgt, is nog een open beslissing; OPEN-SOURCE-LOKAAL.md §1 adviseert AGPL-3.0-or-later.

## Volgende stappen

1. **Fase 0 met echte foto's:** 10–20 onderdelen met bekende maten scannen, afwijkingen en U95-kalibratie meten, drempels bijstellen (debugbeelden in `resultaat/debug/`).
2. **Gatranden en de hoogte uit beeldranden** (subpixel) als extra term in de fit — nauwkeuriger bij weinig bovenaanzichten.
3. **Uitbreiding van de objectklasse:** treden en kamers (meerdere hoogteniveaus), afschuiningen, sleufvormen, gatenpatronen op rechthoekige delen.
4. **Vrije vormen:** COLMAP + OpenMVS met de matposes, en `camtocad cad` voor puntenwolken.
5. **Android-app** (GPL, zonder Google Play Services) met geleide opname: dekkingskoepel, automatisch afdrukken, stills op volle resolutie.
6. **FreeCAD-export** (`.FCStd` met PartDesign-feature tree) naast het CadQuery-script.
