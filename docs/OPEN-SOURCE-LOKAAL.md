# Cam-to-CAD — Haalbaarheidsanalyse: volledig open source en zonder cloudkosten

| | |
|---|---|
| **Versie** | 0.1 — concept ter review |
| **Datum** | 25 september 2026 |
| **Hoort bij** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Vraag** | Kan Cam-to-CAD volledig met open-source software worden gebouwd, zonder te betalen voor cloud? |

## Conclusie

**Ja, dat kan.** Voor elke laag van de architectuur bestaat een volwassen open-source alternatief, en de verwerking kan op de eigen pc of een eigen server draaien. Er zijn dan geen cloudkosten. Wat overblijft is hardware die vaak al aanwezig is, en stroom (orde € 0,05 per scan).

Dat vraagt drie concessies:

1. **Geen ARCore in de 100%-open variant.** ARCore is gratis, maar niet open source: de runtime zit in de propriëtaire *Google Play Services for AR*. De vervanger is de kalibratiemat zelf. OpenCV berekent per frame de camerapose uit de ChArUco-markers — metrisch exact en zonder drift. De mat wordt daarmee verplicht, maar voor de Precisie-modus was ze dat al. ARCore kan als optionele plug-in blijven voor wie zonder mat wil scannen.
2. **De rekenkracht komt van de gebruiker.** Met een NVIDIA-GPU (≥ 8–12 GB videogeheugen) is een scan in ~10–30 minuten verwerkt. Alleen een CPU werkt ook, maar duurt ~30–90 minuten en mist de GPU-stappen. Deze tijden zijn indicatief en worden in Fase 0 gemeten.
3. **Geen kant-en-klare open, lerende CAD-modellen.** De gewichten van CAD-Recode, cadrille en vergelijkbare modellen zijn niet-commercieel en dus niet open source in OSI-zin. De **klassieke route** is wél volledig open en vraagt geen training: RANSAC, kinematische analyse, relaties, PolyFit, schetsen uit doorsneden en OCCT (ARCHITECTURE.md §6). Die volstaat voor v1 op prismatische en gedraaide onderdelen. Lerende componenten train je later zelf, en publiceer je open.

**Aanbeveling:** bouw Cam-to-CAD *local-first*: één set open-source containers die op een desktop, op een eigen server en — optioneel, later — in de cloud draait. Begin lokaal. Dat betekent nul cloudkosten en data die bij de gebruiker blijft, terwijl de architectuur uit ARCHITECTURE.md grotendeels intact blijft.

---

## 1. Uitgangspunten

- **"Volledig open source"** betekent: alleen licenties die door de Open Source Initiative zijn goedgekeurd (MIT, BSD, Apache-2.0, MPL-2.0, LGPL, GPL, AGPL). Daarbuiten vallen:
  - licenties voor "alleen niet-commercieel" of "alleen onderzoek" (bijv. CC BY-NC) — die zijn niet open source in OSI-zin;
  - propriëtaire runtimes en diensten: ARCore (Google Play Services for AR), Firebase/FCM, NPU-bibliotheken van chipfabrikanten, en de API's van Onshape en Fusion.
- **"Geen cloudkosten"** betekent: de verwerking draait op hardware van de gebruiker; er zijn geen betaalde clouddiensten nodig.
- **Buiten onze invloed:** de firmware, drivers en camera-HAL van een telefoon zijn propriëtair. Dat geldt voor elke Android-app.

**De licentie van Cam-to-CAD zelf.** Valt het project zelf onder GPL-3.0 of AGPL-3.0, dan zijn copyleft-componenten zoals CGAL, OpenMVS, PyMeshLab en OpenSplat gewoon bruikbaar, zonder commerciële licenties. De licentiepuzzel wordt dan eenvoudiger dan in de commerciële cloudvariant (ARCHITECTURE.md §5.4). Voorstel:

- **AGPL-3.0-or-later** voor de verwerkingspipeline: sluit aan op OpenMVS en OpenSplat, en houdt ook gehoste afgeleide versies open;
- **GPL-3.0-or-later** voor de Android-app: publiceerbaar via F-Droid.

Laat dit juridisch bevestigen.

## 2. Deploymentprofielen zonder cloudkosten

| Profiel | Waar draait de verwerking | Kosten | Kwaliteit | Geschikt voor |
|---|---|---|---|---|
| **A — Lokaal op pc** (aanbevolen) | desktop of laptop van de gebruiker; de telefoon stuurt de Scan Bundle via wifi | stroom, orde € 0,05 per scan | volledig (Standaard + Precisie) | makers, MKB, MVP |
| **B — Eigen server** | server of NAS met GPU, binnen een bedrijf of thuis | hardware + stroom | volledig; meerdere gebruikers | teams, IP-gevoelige bedrijven |
| **C — Alleen telefoon ("Lite")** | op het toestel zelf | geen | beperkt: eenvoudige prismatische delen, lagere nauwkeurigheid | onderweg, snelle preview |

Profiel A in één oogopslag:

```text
 ANDROID-APP  (GPL-3.0; zonder Google Play Services, publiceerbaar via F-Droid)
   Camera2 + OpenCV: ChArUco-pose per frame (metrisch, zonder VIO)
   EdgeTAM via LiteRT of ncnn: objectmasker
   dekking: visual hull + next-best-view;  stills op volle resolutie (Camera2)
   AR-overlays: Filament of OpenGL ES
        |
        |  Scan Bundle via lokaal wifi (HTTP-upload na QR-koppeling) of Syncthing
        v
 DESKTOP-COMPANION  (AGPL-3.0; Docker Compose op Linux, Windows + WSL2 of macOS)
   lokale API (FastAPI) + taakwachtrij (SQLite) + viewer in de browser (three.js)
   [1] ingest -> [2] bundle adjustment met matposes (COLMAP + Ceres)
   -> [3] MVS (COLMAP of OpenMVS) + oppervlakte-GS (gsplat of OpenSplat)
   -> [4] mesh-to-CAD (Open3D, CGAL, OCCT, CadQuery, PlaneGCS)
   -> [5] validatie -> [6] export: STEP, IGES, STL/3MF, CadQuery-script, FreeCAD
        |
        v
 FreeCAD (LGPL) als open-source CAD-omgeving voor nabewerking
```

### 2.1 De sleutel: de mat als tracker

Het object ligt toch al op de kalibratiemat. Uit de gedetecteerde ChArUco-hoeken en de bekende intrinsics (uit de eenmalige kalibratie met dezelfde mat) berekent OpenCV per frame de metrische camerapose (PnP). Dat heeft grote voordelen:

- **Exacte schaal per frame**, zonder VIO-drift en zonder de 1–3% schaalfout van ARCore.
- **Geen afhankelijkheid van Google Play Services**, en geen ARCore-certificering: vrijwel elk Android-toestel met Camera2 werkt.
- **De Shared Camera-beperking vervalt.** Zonder ARCore beheert de app Camera2 zelf. Een analysestream (~1080p) naast een stream op volle resolutie valt binnen de streamcombinaties die Android garandeert, dus stills op volle resolutie zonder stop-and-shoot.
- De cloudpipeline gebruikte de ARCore-poses toch alleen als prior; matposes zijn betere priors.

De beperkingen, met hun oplossing:

- **De mat moet (deels) in beeld zijn.** Bij close-ups kunnen de markers wegvallen. Zulke frames krijgen hun pose in de reconstructiestap, via de overlap met naburige frames. De live begeleiding valt terug op de laatst bekende pose plus de gyroscoop.
- **Geen diepte-uit-beweging.** De dekking loopt via de visual hull uit de objectmaskers — in de bijgewerkte architectuur al de primaire methode (ARCHITECTURE.md §4.4).
- **Geen lichtschatting van ARCore.** In plaats daarvan beeldstatistiek: belichtingsmetadata, histogram en schaduwdetectie.

**Alternatief zonder mat:** open-source VIO zoals Basalt (BSD-3) of OpenVINS (GPL-3.0). Dat werkt, maar vraagt per toestelmodel een camera-IMU-kalibratie (bijv. met Kalibr, BSD-3) en nauwkeurige tijdsynchronisatie. Veel engineering voor weinig winst, omdat de mat voor Precisie toch nodig is.

## 3. Componentkeuze: volledig open source

| Laag | Commerciële cloudvariant (ARCHITECTURE.md) | Volledig open-source variant | Licentie |
|---|---|---|---|
| Tracking op de telefoon | ARCore (VIO) | Mat-tracking met OpenCV (ChArUco + PnP); optioneel Basalt of OpenVINS | Apache-2.0; BSD-3 / GPL-3.0 |
| Camera | Camera2 via ARCore Shared Camera | Camera2 direct, inclusief een stream op volle resolutie | Android-framework (AOSP) |
| On-device ML | LiteRT met NPU-accelerators | LiteRT op CPU/GPU, of ncnn (Vulkan) of ONNX Runtime; NPU-bibliotheken van chipfabrikanten zijn niet open | Apache-2.0; BSD-3; MIT |
| Segmentatie op het toestel | EdgeTAM of MediaPipe | EdgeTAM | Apache-2.0 |
| Rendering | Filament, via SceneView | Filament of OpenGL ES (de AR-variant van SceneView leunt op ARCore) | Apache-2.0 |
| Pushberichten | FCM | niet nodig: de app volgt de voortgang via het lokale netwerk | — |
| Overdracht | HTTPS naar de cloud | lokale HTTP-upload (FastAPI) na QR-koppeling, of Syncthing | MIT; MPL-2.0 |
| Orkestratie | Temporal + Kubernetes | lichte lokale taakwachtrij met caching per stap | — |
| Features en matching | ALIKED/DISK + LightGlue, RoMa | idem | BSD-3; Apache-2.0; MIT |
| SfM en bundle adjustment | COLMAP 4.x + Ceres | idem; met matposes is het vooral triangulatie + bundle adjustment, dus ook op CPU snel | BSD-3 |
| MVS | COLMAP PatchMatch (CUDA) | COLMAP PatchMatch op NVIDIA; OpenMVS op CPU (build zonder de max-flow-component die alleen voor onderzoek is gelicentieerd) | BSD-3; AGPL-3.0 |
| Oppervlakte-GS | 2DGS/PGSR-principes op gsplat | gsplat (NVIDIA) of OpenSplat (NVIDIA, AMD, Apple Metal; op CPU ~100× trager) | Apache-2.0; AGPL-3.0 |
| Feed-forward-laag | MapAnything (Apache-variant) of VGGT-1B-Commercial | alleen MapAnything, Apache-2.0-variant (de VGGT-licentie is niet OSI-open) | Apache-2.0 |
| Normaal- en diepte-priors | MoGe-2, StableNormal, Marigold | MoGe-2, StableNormal; Marigold alleen v1-0 (de v1-1-gewichten vallen onder OpenRAIL, niet OSI) | MIT; Apache-2.0 |
| Puntenwolk en mesh | Open3D; CGAL en PyMeshLab alleen server-side | idem; GPL is geen probleem in een GPL/AGPL-project | MIT; GPL-3.0 |
| B-rep, export, parametriek | OCCT, CadQuery/build123d, FreeCAD | idem | LGPL-2.1 + exceptie; Apache-2.0; LGPL-2.1+ |
| Constraint-solver | PlaneGCS; later eventueel D-Cubed | PlaneGCS (D-Cubed is propriëtair) | LGPL |
| Differentieerbaar renderen | PyTorch3D | idem | BSD-3 |
| CAD-taalmodel (route E2) | eigen model | eigen model op een open basismodel: Qwen2.5-1.5B, of OLMo 2 (ook de trainingsdata zijn open) | Apache-2.0 |
| 3D-segmentatienetwerk | PTv3-architectuur, zelf getraind | idem; gewichten open publiceren | MIT |
| Synthetische trainingsdata | Blender + BlenderProc | idem | GPL |
| Viewer en nabewerking | web-editor; koppelingen met Onshape en Fusion | viewer in de browser (three.js) + FreeCAD; Onshape en Fusion vallen af (propriëtair) | MIT; LGPL |

## 4. Wat lever je in, en wat win je?

| Aspect | Commerciële cloudvariant | Open en lokaal |
|---|---|---|
| Kosten per scan | ~$0,15–0,30 compute | orde € 0,05 stroom |
| Privacy | EU-cloud | data verlaat het eigen netwerk niet |
| Scannen zonder mat | ja (ARCore) | alleen met de optionele ARCore-plug-in of open VIO |
| Stills op volle resolutie | per toestel: gelijktijdig of via stop-and-shoot | standaard (Camera2 zonder Shared Camera) |
| Nauwkeurigheid | bundle adjustment met markers | gelijk: dezelfde bundle adjustment |
| Doorlooptijd | 10–25 min (L4) | 10–30 min met een NVIDIA-GPU; 30–90 min op CPU |
| Automatisering in v1 | klassieke route, later lerende modellen | idem; lerende modellen zelf trainen |
| Gebruiksgemak | alleen de app | app + desktopsoftware installeren |
| Toestelondersteuning | ARCore-gecertificeerde toestellen | vrijwel elk toestel met Camera2 |
| CAD-koppelingen | STEP + Onshape/Fusion | STEP, CadQuery en FreeCAD |
| Beheer en support | centraal | diverse pc's, drivers en CUDA-versies → containers en een lijst met geteste hardware |

## 5. Hardware en doorlooptijd

Indicatief; te valideren in Fase 0.

| Configuratie | Voorbeeld | Standaard-scan | Beperkingen |
|---|---|---|---|
| NVIDIA-GPU, ≥ 8–12 GB | desktop met een RTX 3060 12 GB of beter | ~10–30 min | geen: de volledige pipeline |
| Apple Silicon, ≥ 16 GB | Mac mini of MacBook met M-chip | ~20–60 min | geen CUDA: MVS via OpenMVS (CPU), oppervlakte-GS via OpenSplat (Metal), netwerken via PyTorch (MPS) |
| AMD-GPU (Linux, ROCm) | Radeon-desktop | ~15–45 min | OpenSplat en OpenMVS werken; gsplat en COLMAP-PatchMatch vragen CUDA |
| Alleen CPU (8+ cores, 16–32 GB RAM) | gewone desktop of laptop | ~30–90 min | geen oppervlakte-GS: MVS + visual hull; netwerken traag |
| Alleen telefoon (profiel C) | recent topmodel | ~5–20 min, grof | lagere resolutie, warmte, alleen eenvoudige delen |

**Stroom:** een pc van ~300 W die een half uur rekent, verbruikt ~0,15 kWh — bij € 0,30 per kWh zo'n € 0,05 per scan.

## 6. Profiel C: alleen de telefoon

Een beperkte variant is haalbaar, maar niet als v1:

- **Poses** zijn bekend uit de mat, dus geen volledige SfM; alleen een kleine bundle adjustment op het toestel (Ceres compileert voor Android).
- **Dichte geometrie:** een lichte plane-sweep-stereo op de GPU (Vulkan compute) tussen naburige stills, of voor prismatische delen vooral silhouetten en beeldranden.
- **CAD:** de klassieke route in C++ (primitieven fitten, relaties, schetsen uit doorsneden). OCCT draait native op Android, dus STEP-export op het toestel is mogelijk.
- **Optioneel:** een klein, gekwantiseerd CAD-taalmodel via llama.cpp (MIT) op topmodellen.

De grenzen liggen bij geheugen, warmte (throttling), batterij en de lagere resolutie van de dichte stereo; het profiel is daardoor alleen geschikt voor eenvoudige prismatische en gedraaide delen. **Advies:** laat eerst de desktoppipeline rijpen. Omdat de kern gedeelde C++-code is, kunnen onderdelen later stap voor stap naar de telefoon — te beginnen met de preview en eenvoudige delen.

## 7. Welke kosten blijven er over?

- **Ontwikkeltijd:** gelijk aan de cloudvariant (ARCHITECTURE.md §9).
- **Hardware**, als er nog geen geschikte pc is.
- **Training van lerende componenten:** eenmalige GPU-tijd. Dat kan op een eigen GPU — het finetunen van een model van 1,5B met LoRA/QLoRA past op een kaart met 12–24 GB — of het wordt uitgesteld door in v1 alleen de klassieke route te gebruiken.
- **Verbruiksartikelen:** de printbare mat (gratis), eventueel scanspray.
- **Distributie:** F-Droid en GitHub-releases zijn gratis.

## 8. Aanbevolen aanpak

1. **Fase 0 zonder eigen code (1–2 weken).** Fotografeer 10–20 onderdelen op een geprinte ChArUco-mat en doorloop de keten met bestaande open-source tools: COLMAP → OpenMVS → CloudCompare (plugin *RANSAC Shape Detection*) → FreeCAD (*Reverse Engineering*-workbench + PartDesign). Zo zijn de nauwkeurigheid en de open keten gevalideerd voordat er een app bestaat.
2. **Fase 1.** Desktop-companion (Docker) + Android-app met mat-tracking + de klassieke mesh-to-CAD-route → STEP, CadQuery en FreeCAD. Nul cloudkosten.
3. **Fase 2.** Lerende componenten trainen op een eigen GPU en de gewichten open publiceren (Apache-2.0).
4. **Later, optioneel.** Een gehoste dienst met dezelfde containers. De cloud wordt dan een keuze, geen voorwaarde.

## Bronnen (selectie)

- OpenSplat — AGPL-3.0; NVIDIA-, AMD- en Apple-GPU's en CPU: <https://github.com/pierotofy/OpenSplat>
- Qwen2.5 — alle open modellen behalve 3B en 72B onder Apache-2.0: <https://qwenlm.github.io/blog/qwen2.5/>
- OLMo 2 — gewichten, code en data open (Apache-2.0; data ODC-BY): <https://www.maginative.com/article/ai2-releases-olmo-2-the-most-capable-fully-open-ai-model/>
- gsplat — Apache-2.0, met 2DGS-rasterisatie: <https://github.com/nerfstudio-project/gsplat>
- MapAnything — Apache-2.0-variant van de gewichten: <https://github.com/facebookresearch/map-anything#models>
- COLMAP — BSD-3; componenten in de standaardbuild: <https://github.com/colmap/colmap/blob/main/LICENSE>
- OpenMVS — AGPL-3.0: <https://github.com/cdcseacave/openMVS/blob/master/COPYRIGHT.md>
- OCCT — LGPL-2.1 met exceptie: <https://github.com/Open-Cascade-SAS/OCCT/blob/master/OCCT_LGPL_EXCEPTION.txt>
- ARCore — ondersteunde toestellen en Google Play Services for AR: <https://developers.google.com/ar/devices>
