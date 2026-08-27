# Reference papers

`nextaire_tools` packages the methodology from three peer-reviewed air-quality ML studies
so their datasets and models can be rebuilt with a small, tested, installable
toolkit. Each paper maps onto a runnable recipe in
[`../reproductions/`](../reproductions/).

The PDFs themselves are the publisher versions and are **not** redistributed with
the package (they are git-ignored). Drop your own copies here with these
filenames if you want them locally.

| # | Reproduction script | Paper | Data |
|---|---------------------|-------|------|
| 1 | [`paper1_petric2024_aaqr.py`](../reproductions/paper1_petric2024_aaqr.py) | Petrić et al. (2024), *AAQR* 24:230317 | Zenodo [7959116](https://doi.org/10.5281/zenodo.7959116) (public) |
| 2 | [`paper2_jimenez2024_multitarget.py`](../reproductions/paper2_jimenez2024_multitarget.py) | Jiménez-Navarro et al. (2024), *Results in Engineering* 24:103290 | On request |
| 3 | [`paper3_racic2026_source_apportionment.py`](../reproductions/paper3_racic2026_source_apportionment.py) | Račić et al. (2026), *Atmospheric Environment: X* 29:100413 | On request |

## Citations

**Paper 1** — Petrić, V., Hussain, H., Časni, K., Vučković, M., Schopper, A.,
Ujević Andrijić, Ž., Kecorius, S., Madueño, L., Kern, R., Lovrić, M. (2024).
*Ensemble Machine Learning, Deep Learning, and Time Series Forecasting:
Improving Prediction Accuracy for Hourly Concentrations of Ambient Air
Pollutants.* Aerosol and Air Quality Research 24(12), 230317.
DOI: [10.4209/aaqr.230317](https://doi.org/10.4209/aaqr.230317).
Graz, Austria · 5 stations · hourly 2014–2020 · PM10/NO/NO₂/O₃.

**Paper 2** — Jiménez-Navarro, M. J., Lovrić, M., Kecorius, S., Nyarko, E. K.,
Martínez-Ballesteros, M. (2024). *Explainable deep learning on multi-target time
series forecasting: An air pollution use case.* Results in Engineering 24, 103290.
DOI: [10.1016/j.rineng.2024.103290](https://doi.org/10.1016/j.rineng.2024.103290).
Graz, Austria · 5 stations · hourly 2014–2022 · 17 station×pollutant targets,
24 h-ahead forecasting with a Temporal Selection Layer.

**Paper 3** — Račić, N., Ružičić, S., Petrić, V., Terzić, T., Antunović, M.,
Škaro, I., Pehnec, G., Bešlić, I., Jakovljević, I., Sever Štrukil, Z., Rinkovec,
J., Žužul, S., Lovrić, M. (2026). *Assessment of contributors to airborne PAHs
and heavy metals in PM₁₀ using temporal, spatial, traffic and heating data in
explainable machine learning models.* Atmospheric Environment: X 29, 100413.
DOI: [10.1016/j.aeaoa.2026.100413](https://doi.org/10.1016/j.aeaoa.2026.100413).
Zagreb, Croatia · 4 stations · daily 2017–2020 · PM10-bound PAHs & metals,
NMF + Random Forest + SHAP.
