# vaccination

## Project Description

Many countries, especially low and middle income countries, are facing limited vaccination supply and fearing the increasingly wide-spread emergence of SARS-CoV-2 virus mutants.

The majority of available vaccines require two immunization doses to provide maximum protection. Yet, the immunological response to the first (''prime'') dose may already provide a substantial degree of protection against infection and severe disease. Thus, it may be more epidemiologically more effective to vaccinate as many people as possible with only one dose, instead of administering a person a second (''booster'') dose. Such a strategic vaccination campaign may help to more effectively slow down the spread of SARS-CoV-2, thereby reducing fatalities and the risk of collapsing health care systems.

To study the conditions which make prime-first vaccination favourable over prime-boost protocols, we combine epidemiological modeling, random sampling techniques, and decision tree learning.

A schematic of prime-first and prime-boost vaccination campaigns and our SEIR model extension is shown below. Panels (A,B) show the evolution of the number of prime and prime-boost vaccianted individuals. Model compartments and transitions are shown in panel (C).

![Image](illustration_final.png)

Vaccination-campaign-preference diagrams can be generated by running the scripts in ``model/vaccination_preference_diagrams``. A simple example that shows the evolution of different model compartments is stored in ``model/example``. If you are interested in generating input files for the decision tree analysis, please check the files in ``model/datasetA`` and ``model/datasetB``.

## Reference
* L. Böttcher, J. Nagler, [Decisive Conditions for Strategic Vaccination against SARS-CoV-2](https://www.medrxiv.org/content/10.1101/2021.03.05.21252962v1), Chaos (2021)

```
@article{bottcher2021decisive,
  title={Decisive Conditions for Strategic Vaccination against SARS-CoV-2},
  author={B{\"o}ttcher, Lucas and Nagler, Jan},
  journal={Chaos},
  year={2021}
}
```
