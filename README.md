# Cam-to-CAD

Scan een fysiek onderdeel met de camera van je telefoon en krijg een bewerkbaar CAD-model terug: een schone B-rep in STEP én een parametrisch model (CadQuery-script met benoemde maten).

**Status:** route A (lokaal en volledig open source, geen cloud) werkt (v0.2) voor platte 2,5D-onderdelen met doorgaande gaten. Getest op synthetische scans, ook met storingen uit echte foto's; validatie met echte foto's volgt.

## Snel starten

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[server]"
camtocad demo                  # synthetische testscan: van foto's naar STEP, zonder camera
camtocad mat --formaat A4      # printbare kalibratiemat (PDF)
camtocad server                # foto's uploaden vanaf je telefoon via wifi
camtocad scan <map-met-fotos>  # of verwerken via de opdrachtregel (--meetlijn 99.6 corrigeert de printschaal)
```

Gaat er iets mis, kijk dan in `<uitvoer>/debug/` (zie [Als het niet lukt](docs/ROUTE-A.md#als-het-niet-lukt)).

## Documentatie

- [Route A — gebruik, werking, resultaten en beperkingen](docs/ROUTE-A.md)
- [Route A — verbeterpunten](docs/ROUTE-A-VERBETERPUNTEN.md): stresstests, codereview en stand van de techniek (2026), geprioriteerd.
- [Architectuurdocument](docs/ARCHITECTURE.md) — user journey en schaalkalibratie, technische stack, 3D-reconstructie, de mesh-to-parametric pipeline, edge cases, validatie en roadmap.
- [Haalbaarheidsanalyse: volledig open source en zonder cloudkosten](docs/OPEN-SOURCE-LOKAAL.md)
