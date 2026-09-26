# Fase 0 — meten met echte foto's

Alle drempels en de opgegeven onzekerheid (U95) van route A zijn tot nu toe afgesteld op gerenderde scans. Fase 0 meet hoe nauwkeurig de scans met échte foto's zijn, en of U95 eerlijk is: bij een eerlijke onzekerheid valt ongeveer 95% van de fouten binnen U95. Dit document beschrijft hoe je de meetset maakt en hoe je hem draait (verbeterpunt V1 in [ROUTE-A-VERBETERPUNTEN.md](ROUTE-A-VERBETERPUNTEN.md)).

Kort: per onderdeel een map met foto's en een `maten.json` met je schuifmaatmetingen, dan `camtocad valideer <map>`.

## 1. Wat je nodig hebt

- **De mat v2.** Maak hem met `camtocad mat --formaat A4` (of `A3`, `Letter`).
  - Print op 100% (werkelijke grootte), op mat papier, en leg hem vlak op een stijve plaat.
  - Meet beide meetlijnen van 100,0 mm na met een schuifmaat: **X** onder de mat en **Y** links. Noteer ze; ze komen in `maten.json`.
  - Een mat v1 (zonder stippenraster) wordt nog herkend, maar een zwart onderdeel lukt daarop niet. Op mat v2 wel.
- **Een digitale schuifmaat** (0,01 mm). Radiusmallen voor afrondingen zijn handig maar niet nodig.
- **10–20 onderdelen**, zo gevarieerd mogelijk:
  - materiaal: aluminium en staal (mat en blank), zwart, wit en gekleurd kunststof;
  - formaat: van ~20 mm (klein plaatje) tot 100–150 mm;
  - vorm: rechthoekig met afgeronde hoeken, een L- of U-vorm, een ring of flens;
  - dikte 3–20 mm, met doorgaande gaten van verschillende diameters.
  Alleen 2,5D-onderdelen: plat, overal even dik, met doorgaande gaten.
- **Een referentie met bekende maten**, bijvoorbeeld een eindmaat of een nauwkeurig gefreesd blokje, en een ring of ringkaliber. Daarmee zie je het verschil tussen een meetfout van de scan en een meetfout van de schuifmaat.

## 2. Meten met de schuifmaat

- Meet elke maat drie keer, op verschillende plaatsen, en noteer het gemiddelde. Laat het onderdeel eerst op kamertemperatuur komen (niet direct na het vasthouden).
- **Lengte en breedte:** de grootste en kleinste buitenmaat langs de hoofdrichtingen van het onderdeel.
- **Hoogte:** de dikte.
- **Gaten:** de diameter met de binnenbekken, op twee plaatsen 90° verdraaid.
- **Hartafstand** tussen twee gaten: meet de binnenmaat tussen de gaten en tel de halve diameters erbij op (of de buitenmaat over beide gaten min de halve diameters).
- **Diameter** van een rond onderdeel: de buitendiameter.
- **Afrondingen:** alleen als je een radiusmal hebt; anders weglaten.

## 3. Fotograferen

- Leg het onderdeel plat in het midden van de mat, bij diffuus licht (bewolkt daglicht, of een lamp tegen het plafond).
- Maak 30–60 foto's met dezelfde telefoon, zonder zoom:
  - 4–6 recht van boven, met de hele mat in beeld;
  - rondom op ongeveer 35° en 60° boven de mat, om de ~45°.
- **Controleer de set direct**, vóór je het onderdeel weghaalt:
  - Via de telefoonpagina (`camtocad server`) gebeurt dat vanzelf. Elke foto krijgt een oordeel (goed, matig of onbruikbaar), en een dekkingskaart toont in het rood welke richtingen nog ontbreken.
  - Via de opdrachtregel draai je `camtocad controleer <map-met-fotos>`. Dat kost ongeveer 0,1 s per foto.
  - Maak de foto's bij die de controle vraagt, bijvoorbeeld "nog 2 foto's recht boven het onderdeel".

## 4. Mappen en `maten.json`

```
validatie/
  beugel-alu/
    maten.json
    fotos/IMG_0001.jpg ...     (de foto's mogen ook direct in de map staan)
  plaatje-zwart/
    maten.json
    fotos/...
```

Een `maten.json`:

```json
{
  "naam": "beugel aluminium",
  "meetlijn": [100.05, 99.95],
  "maten": {
    "lengte": 80.02,
    "breedte": 40.01,
    "hoogte": 12.03,
    "gaten": [6.62, 6.60],
    "hartafstanden": [60.01],
    "afrondingen": [3.0, 3.0, 3.0, 3.0]
  }
}
```

| Veld | Betekenis |
|---|---|
| `meetlijn` | De gemeten meetlijnen X en Y van de geprinte mat, in mm. Eén getal geldt voor beide. Laat je hem weg, dan wordt 100,0 aangenomen |
| `mat` | Optioneel: `A4`, `A3`, `Letter`, `A4-v1` of `A3-v1`. Standaard wordt de mat herkend aan de markers |
| `lengte`, `breedte`, `hoogte`, `diameter` | Eén waarde |
| `gaten`, `hartafstanden`, `afrondingen` | Een lijst. Elke waarde wordt gekoppeld aan de dichtstbijzijnde maat van het model |

## 5. Draaien

```bash
camtocad valideer validatie/
```

Per onderdeel wordt de scan verwerkt naar `resultaat/`. Een eerder resultaat wordt hergebruikt, tenzij er nieuwe foto's zijn of je `--opnieuw` meegeeft (bijvoorbeeld na een update van camtocad). Daarna volgen per maat:

- de referentie en de gefitte (ongesnapte) waarde;
- de fout: model min referentie;
- de U95 die de scan zelf opgeeft, en of de fout daarbinnen valt;
- de waarde na het snappen.

De samenvatting staat in de terminal en in `validatie/validatie.html` en `validatie.json`.

Voor een regressietest na een wijziging: `camtocad valideer validatie/ --opnieuw --eis-dekking 0.9` geeft een foutcode als minder dan 90% van de fouten binnen U95 valt.

Ook de demo levert zo'n map op: `camtocad demo --uit demo` en daarna `camtocad valideer demo`.

## 6. Lezen

| Wat je ziet | Betekenis |
|---|---|
| ~95% binnen U95 | De onzekerheid is eerlijk |
| Duidelijk minder dan 90% | U95 is te optimistisch; de fouten zijn groter dan de scan beweert |
| Alles ruim binnen U95 | U95 is te ruim; de scan is beter dan hij zegt |
| Bias per soort, bijvoorbeeld "gat: systematisch −0,06 mm" | Een systematische fout, te corrigeren in een volgende versie |
| "Lengtes wijken gemiddeld +0,3% af" | Vrijwel altijd de printschaal: meet de meetlijnen na en zet ze in `maten.json` |
| "gat ontbreekt in het model" of extra gaten | Een topologiefout. Kijk in `resultaat/debug/` (maskers, bovenaanzicht) |
| Scan met `fout` | De foutmelding staat erbij; zie [ROUTE-A.md — Als het niet lukt](ROUTE-A.md#als-het-niet-lukt) |

## 7. Resultaten delen

Wil je helpen de drempels en U95 af te stellen, deel dan:

- `validatie.json`;
- per onderdeel `resultaat/report.json` en `resultaat/debug/diagnose.json`.

Die bevatten geen foto's. De foto's zelf zijn alleen nodig als een scan mislukt en de debugbeelden niet genoeg zeggen.
