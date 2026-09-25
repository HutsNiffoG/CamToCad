# Cam-to-CAD

Scan een fysiek onderdeel met de camera van je telefoon en krijg een bewerkbaar CAD-model terug: een schone B-rep in STEP én een parametrisch model (CadQuery-script met benoemde maten).

**Status:** route A (lokaal en volledig open source, geen cloud) werkt in een eerste versie (v0.1) voor platte 2,5D-onderdelen met doorgaande gaten. Getest op synthetische scans; validatie met echte foto's volgt.

## Snel starten

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[server]"
camtocad demo                  # synthetische testscan: van foto's naar STEP, zonder camera
camtocad mat --formaat A4      # printbare kalibratiemat (PDF)
camtocad server                # foto's uploaden vanaf je telefoon via wifi
camtocad scan <map-met-fotos>  # of verwerken via de opdrachtregel
```

## Documentatie

- [Route A — gebruik, werking, resultaten en beperkingen](docs/ROUTE-A.md)
- [Architectuurdocument](docs/ARCHITECTURE.md) — user journey en schaalkalibratie, technische stack, 3D-reconstructie, de mesh-to-parametric pipeline, edge cases, validatie en roadmap.
- [Haalbaarheidsanalyse: volledig open source en zonder cloudkosten](docs/OPEN-SOURCE-LOKAAL.md)
