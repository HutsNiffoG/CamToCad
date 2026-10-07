# Cam-to-CAD

Scan een fysiek onderdeel met de camera van je telefoon en krijg een bewerkbaar CAD-model terug: een schone B-rep in STEP én een parametrisch model (CadQuery-script met benoemde maten).

**Status:** route A (lokaal en volledig open source, geen cloud) werkt (v0.13) voor platte 2,5D-onderdelen met doorgaande gaten (ook verzonken, met een kleine faas of met een kamerboring) en blinde gaten, sleuven en rechthoekige uitsparingen, ook zwarte, witte en gekleurde, en met een afgeschuinde of afgeronde bovenrand rondom of een rechte trede. Getest op synthetische scans, ook met storingen uit echte foto's. Foto's mogen JPG of HEIC zijn, zo van de telefoon; kleur telt mee als bewijs voor het object, en foto's van een andere lens of met zoom worden herkend. De maten worden op subpixelniveau op de randen gefit, met de rand uit de grijswaarden en de kleur zelf, in lineair licht: de scan leert uit de mat hoe de telefoon de foto bewerkt (toonkromme, ook per stuk beeld zoals bij HDR, verscherping en onscherpte). Elke maat krijgt een eigen U95, afgesteld op 95% dekking. Past een onderdeel niet in het model, of is een gat in de foto's niet goed te zien, dan meldt de scan dat in plaats van stil een fout model of een te krappe U95 te leveren. De eerste echte fotoset leerde vooral dat het onderdeel tijdens de scan moet blijven liggen; dat wordt gecontroleerd, en sinds v0.12 wordt ook een klein duwtje herkend en gecorrigeerd. Een harde slagschaduw, een ander voorwerp op de mat of een vorm die het model mist, worden gemeld. Sinds v0.13 wijst een livecamera in de browser de weg tijdens het fotograferen en maakt hij zelf de foto's (via https op je eigen netwerk). Het gereedschap om met echte foto's en schuifmaatmetingen te valideren is klaar, ook in de browser ([Fase 0](docs/FASE-0.md)); de meetset zelf volgt.

## Snel starten

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[server]"
camtocad demo                  # synthetische testscan: van foto's naar STEP, zonder camera
camtocad mat --formaat A4      # printbare kalibratiemat (PDF); ook A3 en Letter
camtocad server                # telefoon via wifi (scan de QR-code): livecamera met aanwijzingen, of foto's uploaden
camtocad controleer <map>      # of via de opdrachtregel: fotoset controleren (mat, scherpte, dekking) ...
camtocad scan <map>            # ... en verwerken (--meetlijn 99.6 99.8 corrigeert de printschaal in X en Y)
camtocad valideer <map>        # scans vergelijken met schuifmaatmetingen (docs/FASE-0.md)
```

Gaat er iets mis, kijk dan in `<uitvoer>/debug/` (zie [Als het niet lukt](docs/ROUTE-A.md#als-het-niet-lukt)).

## Documentatie

- [Route A — gebruik, werking, resultaten en beperkingen](docs/ROUTE-A.md)
- [Route A — verbeterpunten](docs/ROUTE-A-VERBETERPUNTEN.md): stresstests, codereview en stand van de techniek (2026), geprioriteerd, met wat er in v0.2–v0.13 al is opgelost.
- [Stresstests](docs/STRESSTEST.md): `camtocad stresstest`, gerenderde scans met storingen uit echte foto's, en twee versies per maat vergelijken met de waarheid.
- [Fase 0 — meten met echte foto's](docs/FASE-0.md): meetset maken, fotograferen (ook met de livecamera), metingen invullen in de browser of in `maten.json`, `camtocad valideer`, en delen zonder foto's.
- [Architectuurdocument](docs/ARCHITECTURE.md) — user journey en schaalkalibratie, technische stack, 3D-reconstructie, de mesh-to-parametric pipeline, edge cases, validatie en roadmap.
- [Haalbaarheidsanalyse: volledig open source en zonder cloudkosten](docs/OPEN-SOURCE-LOKAAL.md)
