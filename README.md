# Cam-to-CAD

Scan een fysiek onderdeel met de camera van je telefoon en krijg een bewerkbaar CAD-model terug: een schone B-rep in STEP én een parametrisch model (CadQuery-script met benoemde maten).

**Status:** route A (lokaal en volledig open source, geen cloud) werkt (v0.10) voor platte 2,5D-onderdelen met doorgaande gaten (ook verzonken of met een kamerboring), sleuven en rechthoekige uitsparingen, ook zwarte, witte en gekleurde, en met een afgeschuinde of afgeronde bovenrand rondom of een rechte trede. Getest op synthetische scans, ook met storingen uit echte foto's. Foto's mogen JPG of HEIC zijn, zo van de telefoon; kleur telt mee als bewijs voor het object, en foto's van een andere lens of met zoom worden herkend. De maten worden op subpixelniveau op de randen gefit, met de rand uit de grijswaarden en de kleur zelf, in lineair licht (de toonkromme van de camera wordt per foto uit de mat geschat), en elke maat krijgt een eigen U95, afgesteld op 95% dekking. Past een onderdeel niet in het model, of is een gat in de foto's niet goed te zien, dan meldt de scan dat in plaats van stil een fout model of een te krappe U95 te leveren. De eerste echte fotoset leerde vooral dat het onderdeel tijdens de scan moet blijven liggen; dat wordt gecontroleerd. Het gereedschap om met echte foto's en schuifmaatmetingen te valideren is klaar ([Fase 0](docs/FASE-0.md)); de meetset zelf volgt.

## Snel starten

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[server]"
camtocad demo                  # synthetische testscan: van foto's naar STEP, zonder camera
camtocad mat --formaat A4      # printbare kalibratiemat (PDF); ook A3 en Letter
camtocad server                # foto's uploaden vanaf je telefoon via wifi (scan de QR-code), met directe fotocontrole
camtocad controleer <map>      # of via de opdrachtregel: fotoset controleren (mat, scherpte, dekking) ...
camtocad scan <map>            # ... en verwerken (--meetlijn 99.6 99.8 corrigeert de printschaal in X en Y)
camtocad valideer <map>        # scans vergelijken met schuifmaatmetingen (docs/FASE-0.md)
```

Gaat er iets mis, kijk dan in `<uitvoer>/debug/` (zie [Als het niet lukt](docs/ROUTE-A.md#als-het-niet-lukt)).

## Documentatie

- [Route A — gebruik, werking, resultaten en beperkingen](docs/ROUTE-A.md)
- [Route A — verbeterpunten](docs/ROUTE-A-VERBETERPUNTEN.md): stresstests, codereview en stand van de techniek (2026), geprioriteerd, met wat er in v0.2–v0.10 al is opgelost.
- [Fase 0 — meten met echte foto's](docs/FASE-0.md): meetset maken, fotograferen, `maten.json`, `camtocad valideer`.
- [Architectuurdocument](docs/ARCHITECTURE.md) — user journey en schaalkalibratie, technische stack, 3D-reconstructie, de mesh-to-parametric pipeline, edge cases, validatie en roadmap.
- [Haalbaarheidsanalyse: volledig open source en zonder cloudkosten](docs/OPEN-SOURCE-LOKAAL.md)
