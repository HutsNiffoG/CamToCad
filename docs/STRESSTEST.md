# Stresstests

De stresstests laten zien waar de verwerking breekt. Elke test is een gerenderde scan van een bekend onderdeel op de mat, met storingen zoals in echte foto's. De maten uit de scan worden vergeleken met de waarheid. Ze zeggen niet hoe nauwkeurig echte foto's zijn: dat moet [Fase 0](FASE-0.md) uitwijzen. Wel laten ze zien of een verandering iets beter of slechter maakt. De uitkomsten per versie staan in [ROUTE-A-VERBETERPUNTEN.md](ROUTE-A-VERBETERPUNTEN.md) (§3e–§3l).

## Draaien

```bash
camtocad stresstest --lijst                           # de scenario's
camtocad stresstest basis donker --uit run1           # twee scenario's
camtocad stresstest --uit run1 --cache scans --parallel 3   # alle scenario's, drie tegelijk
camtocad stresstest blauw grijs_kleur --grijs --uit run1_grijs   # kleurscans als grijsbeelden (controle)
camtocad stresstest --vergelijk run0 run1             # twee runs naast elkaar, met de waarheid
```

- Per scenario 46 foto's van 1600 × 1200 pixels: drie ringen schuine foto's (35°, 55° en 72°) en zes van boven, uit de hand (afstand, richting en draaiing wisselen). Er zit ruis in, JPEG, vignettering en verscherping. Per scenario komt daar één ding bij, bijvoorbeeld:
  - een slagschaduw of een handschaduw;
  - een toonkromme zoals van een telefoon (sRGB), ook met lokale toonbewerking (HDR);
  - een zwart, wit of gekleurd onderdeel;
  - weinig lage foto's.
- `naam:zaad` draait een scenario met een ander zaad: andere ruis en andere poses. De uitvoer heet dan `naam_zaad`.
- `--cache MAP` bewaart de gerenderde foto's. Een volgende run gebruikt ze opnieuw, en dan vergelijk je twee versies op precies dezelfde foto's. Renderen kost 20–60 s per scenario.
- Eén scenario kost 1,5–5 minuten op een gewone CPU met 4 kernen, en ~4–5 GB geheugen. Kies `--parallel` naar het geheugen: met 16 GB gaat 3 tegelijk nog net. Alle 49 scenario's tegelijk in drie groepen kosten ~35 minuten.
- De cijfers in ROUTE-A-VERBETERPUNTEN.md zijn gedraaid met `OMP_NUM_THREADS=1`, met de foto's uit de cache. Kalibratie en poses rekenen met één thread, dus een scan geeft bij herhaling dezelfde maten (op een paar duizendsten na).

## Uitvoer

Per scenario komt er een map met:

- `resultaat/`: wat `camtocad scan` ook maakt (`model.step`, `report.html`, `report.json`, `debug/`);
- `foto_*.jpg`: een paar van de gerenderde foto's;
- `stress.json`: status, rekentijd, de maten, de waarheid, het scenario en het log van de verwerking.

## Vergelijken

`camtocad stresstest --vergelijk OUD NIEUW` geeft per scenario en per maat:

- de gemeten waarde en de fout ten opzichte van de waarheid;
- de U95 zonder de printschaal. Een gerenderde mat heeft een exacte schaal, dus die term hoort hier niet bij. Tussen haakjes staat de U95 uit het rapport;
- de fout van de oude run;
- of de maat gesnapt is, en of de snap klopt;
- `<-- buiten U95` als de fout groter is dan de U95.

Onderaan volgen de dekking, de fouten en de snaps voor alle scans samen, en apart voor de scans zonder waarschuwing "onbetrouwbaar". Daarna dezelfde cijfers per soort maat: hoogte, lengte, gat, verzinking, kamerboring, blind gat, sleuf, positie, afronding, bovenrand en trede.

## In CI

Een kleine stresstest draait in CI als eigen taak, naast de gewone tests: `pytest -m stress`. Dat is het scenario "realistisch" met 20 foto's in plaats van 46. Alle maten moeten binnen hun U95 vallen, en buitenmaten, hoogte en gaten binnen 0,1 mm. Lokaal duurt dat ~2,5 minuut. De volledige set draait niet in CI: die duurt te lang en vraagt te veel geheugen.
