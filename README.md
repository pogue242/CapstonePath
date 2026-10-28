<div class="IEEEkeywords">

pedestrian networks, OpenStreetMap, map completion, aerial imagery,
pedestrian routing, accessibility

</div>

# Introduction

Pedestrian navigation depends on the quality of the underlying path
network, yet most routing systems are built around street networks
designed with motor vehicles in mind.   Small pedestrian connections
such as stairways, courtyard passages, alleys, arcades, and passages
through buildings are often absent from these networks or are not
classified as walkable. When a connection is missing, a router must send
the pedestrian around it, producing detours that are far longer than the
route a person could ideally walk. In Molischgasse 31, 1140 Wien, in
Vienna, an official passage of about 20m through a building is absent
from OpenStreetMap (OSM), so OSM-based routing sends pedestrians 145m
around the block. The problem is most severe in dense, older cities,
where the street fabric predates the car and much of the walkable
network consists of exactly these small connections. It also affects
accessibility, since whether a stairway is mapped determines whether a
router can warn about it or route around it. OSM is a widely used base
map for pedestrian routing, so gaps in it carry over to any application
built on it. Some governments, however, maintain official transportation
networks that record these small connections in detail. Austria’s
official Graph Integration Platform (GIP)  is one example, and it makes
it possible to use Vienna as a test site to compare what is missing from
OSM compared to a more complete dataset. This proposal is situated at
the intersection of pedestrian network mapping, automated map
completion, and pedestrian routing. The next section reviews existing
work on how pedestrian networks are built and the gaps that remain.

# Literature Review

Rhoads et al. , in *Sidewalk Networks: Review and Outlook*, heavily
inspired this project, and they examine the lack of reliable pedestrian
network data and how it limits analyses of pedestrian connectivity and
accessibility across cities. They explain that transportation mapping
has historically been designed and mapped primarily for motor vehicles,
with pedestrian infrastructure often treated as a secondary component.
As a result, existing maps may not fully represent the paths available
to pedestrians, making it difficult to analyze walking routes and how
different parts of a city are connected.

Cambra et al.  demonstrate the limitations of existing pedestrian
network data. The researchers manually mapped 223.8 km of pedestrian
infrastructure across six neighborhoods in Lisbon, Portugal, finding
that conventional sidewalks accounted for only 57.3% to 81.4% of total
pedestrian space, depending on the study area. The remaining
infrastructure included other pedestrian connections, such as paths
through parks, accessible courtyards, and open spaces. Their findings
show how much pedestrian infrastructure may exist that is
unconventional. It also highlights the need for automated methods of
creating more complete pedestrian networks, as their manual recreation
took hundreds of hours.

Yin et al.  built SAM-Road++, a model trained using satellite imagery
covering over 13,800 square kilometers to identify roads and generate
connected road networks, even when obstructed, such as by trees.
However, their research focuses on general road infrastructure rather
than smaller pedestrian connections, missing some vital narrow
connections. Hosseini et al.  developed Tile2Net, an open-source
framework that uses aerial imagery to identify sidewalks, crosswalks,
and footpaths and generate pedestrian networks. These studies provide a
technical foundation for how computer vision can automate transportation
network mapping. Our project aims to build upon this research by
training a model to identify pedestrian-centric and hard-to-detect
infrastructure. Rather than generating an entirely new network, our
model will analyze existing maps to identify potentially missing
pedestrian connections and evaluate whether they can be incorporated
into walking routes.

PathwayBench, a framework developed by researchers at the University of
Washington and presented in a 2024 research paper, evaluates whether
pedestrian networks made from aerial imagery produce viable pedestrian
routes. Zhang et al.  used pedestrian data totaling 3,000 square
kilometers across eight cities, including Seattle, DC, São Paulo, and
Santiago, to show that current methods lead to discontinuities. While
individual paths can be extracted, small gaps arise, leading to path
connectivity issues and detours.

Ning et al.  address the tree occlusion problem that the paper by Yin et
al.   noted by recovering sidewalk segments hidden in aerial imagery
using street-view imagery, giving precedent in literature to our
strategy of combining data sources, rather than using solely aerial
imagery. Zhang et al.  respond to the connectivity gaps that
PathwayBench  found by also combining aerial imagery with an existing
street network. Their network is also built on by human vetting
processes.

Treccani et al.  extend automated pedestrian mapping to historic
Sabbioneta, Italy, where they note point cloud processing techniques
often lose accuracy. This addresses the unconventional infrastructure
that Cambra et al.  mapped manually in Lisbon, though this paper’s study
used laser scanning rather than widely available data, which will
present challenges for the advisability of implementation for our
project.

Bartzokas-Tsiompras  finds that pedestrian tags in OpenStreetMap are
inconsistent, reinforcing the data gaps described by Rhoads et al. .
This meta-analysis also notes that pedestrian coverage varies widely
across cities, particularly between Southern and Western European
cities.

Wanyan et al.  use self-supervised learning to reduce the labeled data
needed to map footpaths, redressing the difficulties and
unsustainability of the manual efforts by Cambra et al.  and relevant to
our plan to move from Vienna to other cities down the line of
development.

# Motivation and Problem Statement

The gaps described above are not isolated cases. In a first round
estimate with 8m tolerances for OSM matching paths, we found that 21% of
GIP paths of Vienna were missing from OSM. This estimate is preliminary,
and we are repeating the comparison with distance thresholds from 5m to
12m to see how much it depends on our choice. This indicates that the
missing connections are systemic to the map, not just indicative of
sparse errors.

The problem matters in practice for accessibility. Navigation service
users should have access to the most up to date, optimal route for their
journey, and one that is not incomplete, and that shows all accessible
routes for people of all abilities. Currently, there is strong evidence
to suggest that OSM does not have this level of complete data for
pedestrian routes.

Manual correction does not scale to this problem. Official data such as
the GIP is not available everywhere, and the manual mapping effort
reported by   suggests that complete pedestrian networks are very costly
to build by hand. Existing automated methods, as discussed in the
related work, extract sidewalks, crosswalks, and roads from aerial
imagery, but small connections are the ones those methods struggle to
capture. Pedestrian networks built from imagery also tend to contain
gaps that cause detours.

The problem addressed by this project is therefore the automatic
detection of small pedestrian connections (passages, courtyards, and
stairs) that are missing from OSM. We use Vienna’s official GIP network
  as the reference against which detections are measured. We scope the
work to this class of connection, to Vienna as the study area, and to a
candidate-generation and verification pipeline: one model that proposes
and ranks candidate connections from the existing OSM network, and a
second that uses aerial imagery to check them.

# Research Questions

What share of Vienna’s official passages and stairs is missing from OSM,
and how sensitive is this estimate to the matching tolerance?

How effectively can a candidate-generation model, combined with an
aerial-imagery verification model, detect these missing connections when
measured against the official network?

How much do the recovered links effectively change walking routes?

# Proposed Solution and Implementation

We propose a pipeline with four parts: a dataset built from OSM and the
official Austrian network (GIP), a candidate-generation model (Model 1),
an aerial-imagery verification model (Model 2), and a routing
application that uses the recovered connections.

## Data and Reference Dataset

The study uses four sources. The GIP is Austria’s official map of
streets, paths, and staircases and serves as the training and test
dataset. OSM provides the base map and the routing network for the
application. Aerial photographs of Vienna provide the imagery for
training Model 2, and a city and district boundary shapefile is used to
clip the GIP data.

The GIP covers all of Austria (2,583,146 segments), so we first restrict
it to Vienna (161,395 segments) using the boundary shapefile. We keep
pedestrian segments and pedestrian-only paths, identified through the
segment tables and the user-mode column (151,670 segments). We then
select small links by their type codes (passages partly inside
buildings, stairs, and lifts), which gives 8,619 segments. Because a
single passage or staircase is stored as several touching segments, we
join adjoining segments into whole features, giving 1,974 passages and
1,426 staircases. Finally, we remove train stations and their linked
stairs and paths, leaving 2,074 passages and stairs. These comprise
1,040 courtyard passages, 779 outdoor staircases, 178 passages through
buildings, and 77 other passages.

To measure which of these features are missing from OSM, we compared the
two networks. A point on a feature counts as covered if an OSM path lies
within 8m and runs in almost the same direction (within 35°). Features
with at least 80% of their points covered count as present in OSM, those
between 30% and 80% as partly present, and those under 30% as missing.
Under this rule, about 1,465 features (71%) are in OSM, 436 (21%) are
missing, and 173 remain undetermined, so roughly one in five of Vienna’s
official passages and stairs is absent from OSM. To test how much this
result depends on our choice of threshold, we will repeat the comparison
with distances from 5 to 12m. This comparison provides both the labels
for the models below and the answer to the first research question.

## Model 1: Candidate Generation

Model 1 proposes possible missing small paths based on the existing OSM
network and ranks the candidates by how likely each is to be a real
connection. If on some OSM route, the shortest walk between two points
is about 350m around a block, Model 1 may propose candidate links (such
as a short connection between the two points) that may be shorter and
rank them.

## Model 2: Aerial-Imagery Verification

Model 2 uses Tile2Net as a base model and trains it on the identified
small paths using aerial photographs. The model’s input is an aerial
image of Vienna with the known OSM footpaths drawn on top; during
training, the official passages and stairs from the GIP are added as
well. Model 2 is then used to check the candidate routes proposed by
Model 1.

## Routing Application

The application routes on OSM plus the connections that OSM lacks. The
base map is built from OSM data turned into map tiles once and drawn in
the phone browser with MapLibre GL JS. Navigation uses OpenTripPlanner
2, with a routing graph made from Vienna’s OSM streets, paths, steps,
and lifts; the added connections are included as extra paths in this
graph. Transit and bike-on-train options use Wiener Linien GTFS
timetables for trams, buses, the U-Bahn, and the S-Bahn. An
accessibility layer, based on OSM tags and the official GIP map,
supports avoiding steps, using lifts and ramps, and finding step-free
stops. A search component finds addresses and passes the trip to the
router, which returns route options, such as a shorter route via a
passage compared with a streets-only route.

## Relation to the Research Questions

The dataset comparison answers the first research question by
quantifying what OSM is missing and how sensitive that estimate is to
the matching tolerance. The two models together address the second by
producing detections that can be compared with the GIP. The application
demonstrates how the recovered connections can be used in walking
routes, and supports the third research question if routing impact is
evaluated.

# Evaluation Plan and Expected Outcomes

The evaluation tests whether the proposed pipeline recovers connections
that are missing from OSM, using the official GIP network as the
reference. Evaluation covers three aspects: the extent of the gap (RQ1),
detection quality (RQ2), and the effect on routing (RQ3).

## Reference Data and Split

The GIP features identified as missing from OSM serve as the positive
examples. To ensure that test features are not seen during training, we
split the data spatially by Vienna’s district boundaries, training on
some districts and testing on held-out districts. Features labeled as
undetermined will be reported separately.

## Extent of the Gap (RQ1)

We report the share of official passages and stairs that are present,
partly present, or missing in OSM, using the 8 m / 35° matching rule. To
show how much the result depends on this choice, we repeat the
comparison with distance thresholds from 5 to 12 m and report how the
counts change. We also report the results by feature type (courtyard
passages, outdoor staircases, passages through buildings, and other
passages).

## Detection Quality (RQ2)

Detections are compared with the held-out GIP features using the same
matching rule as the dataset construction. A proposed link counts as
correct if it matches a held-out missing GIP feature. We report:

- **Precision:** the fraction of proposed links that match a held-out
  GIP feature.

- **Recall:** the fraction of held-out missing GIP features that are
  recovered.

- **F1 score**, combining precision and recall.

- **Recall at k** for Model 1, since it ranks candidates.

Results are broken down by feature type. To isolate the contribution of
each model, we compare:

- Model 1 alone (ranked candidates, no verification),

- Model 1 followed by Model 2.

Because the GIP may itself be incomplete, proposed links that do not
match any GIP feature are not necessarily wrong. We therefore manually
inspect a sample of these unmatched detections and report how many
correspond to real connections.

## Effect on Routing (RQ3)

We compare walking routes computed on OSM alone with routes computed on
OSM plus the recovered links. For a set of origin–destination pairs, we
report the change in route length, as in the Molischgasse example (20 m
through the passage versus 145 m around the block).

## Application Testing

We test that the application returns routes that include the added
connections and that its accessibility options (avoiding stairs,
step-free routes) behave as intended on selected examples, like the one
discussed in 1140 Wien in the Introduction section.

## Success Criteria

Our target is for the pipeline to recover at least 95% of the official
passages and stairs that are missing from OSM in held-out test
districts, while maintaining a precision of at least 80%. Recall and
precision are computed using the same matching rule as the dataset
construction (8 m distance, 35° direction). This figure is a target, not
an expected result. We report the values actually obtained, broken down
by feature type, together with the comparison against the baselines
described above.

## Expected Outcomes

At the end of the project we expect to deliver: (1) a labeled dataset of
official passages and stairs in Vienna with their status in OSM; (2) a
candidate-generation model and an aerial-imagery verification model; (3)
an evaluation of the pipeline against the official network; and (4) a
Vienna-focused routing application that includes the recovered
connections, with transit and accessibility options. We would also hope
to (5) extend this application to route all street and pedestrian
networks in Austria.

# GitHub Repository Link

The code for this project is available at
<https://github.com/pogue242/CapstonePath>.

<div class="thebibliography">

11

GIP Österreich (gip.gv.at), OGD release 26.04, CC BY 4.0.

D. Rhoads, C. Rames, A. Solé-Ribalta, M. C. González, M. Szell, and J.
Borge-Holthoefer, “Sidewalk networks: Review and outlook,” *Computers,
Environment and Urban Systems*, vol. 106, Art. no. 102031, Dec. 2023.
doi: 10.1016/j.compenvurbsys.2023.102031.

P. J. Cambra, A. B. Gonçalves, and F. Moura, “The digital pedestrian
network in complex urban contexts: A primer discussion on typological
specifications,” *Finisterra*, vol. 54, no. 110, pp. 155–170, 2019. doi:
10.18055/Finis16414.

P. Yin, K. Li, X. Cao, J. Yao, L. Liu, X. Bai, F. Zhou, and D. Meng,
“Towards satellite image road graph extraction: A global-scale dataset
and a novel method,” in *Proc. IEEE/CVF Conf. Computer Vision and
Pattern Recognition (CVPR)*, 2025, pp. 1527–1537. doi:
10.1109/CVPR52734.2025.00150.

M. Hosseini, A. Sevtsuk, F. Miranda, R. M. Cesar Jr., and C. T. Silva,
“Mapping the walk: A scalable computer vision approach for generating
sidewalk network datasets from aerial imagery,” *Computers, Environment
and Urban Systems*, vol. 101, Art. no. 101950, Apr. 2023. doi:
10.1016/j.compenvurbsys.2023.101950.

Y. Zhang, B. Howe, S. Mehta, N.-J. Bolten, and A. Caspi, “PathwayBench:
Assessing routability of pedestrian pathway networks inferred from
multi-city imagery,” arXiv preprint arXiv:2407.16875, Jul. 2024.
Available: <https://arxiv.org/abs/2407.16875>

H. Ning, X. Ye, Z. Chen, T. Liu, and T. Cao, “Sidewalk extraction using
aerial and street view images,” *Environment and Planning B: Urban
Analytics and City Science*, vol. 49, no. 1, pp. 7–22, 2022.

Y. Zhang, B. Howe, and A. Caspi, “Reliable, routable, and reproducible:
Collection of pedestrian pathways at statewide scale,” arXiv preprint
arXiv:2410.19762, 2024. Available: <https://arxiv.org/abs/2410.19762>

D. Treccani, A. Fernández, L. Díaz-Vilariño, and A. Adami, “Automating
the inventory of the navigable space for pedestrians on historical
sites: Towards accurate path planning,” *Int. J. Applied Earth
Observation and Geoinformation*, 2023. Available:
<https://doi.org/10.1016/j.jag.2023.103400>

A. Bartzokas-Tsiompras, “Utilizing OpenStreetMap data to measure and
compare pedestrian street lengths in 992 cities around the world,”
*European Journal of Geography*, vol. 13, no. 2, pp. 127–141, 2022.
Available: <https://doi.org/10.48088/ejg.a.bar.13.2.127.138>

X. Wanyan, S. Seneviratne, K. Nice, J. Thompson, M. White, N.
Langenheim, and M. Stevenson, “Scalable label-efficient footpath network
generation using remote sensing data and self-supervised learning,”
Available: <https://arxiv.org/html/2309.09446v1>

</div>
