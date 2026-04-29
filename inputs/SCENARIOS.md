# Simulation Scenarios

The legacy sample inputs and outputs have been replaced with PDF-aligned
rain, fog, and haze scenarios for both active backends:

- `inputs/mie/pdf_fog_radiation.json`
- `inputs/mie/pdf_haze_dust.json`
- `inputs/mie/pdf_rain_moderate.json`
- `inputs/mie/quick_display.json`
- `inputs/iitm/pdf_fog_radiation.json`
- `inputs/iitm/pdf_haze_dust.json`
- `inputs/iitm/pdf_rain_moderate.json`
- `inputs/iitm/quick_display.json`

Shared PDF alignment:

- wavelength: 1550 nm
- nominal range: 1-2000 m for the full scenarios
- lidar gate width: 10 m for the full scenarios
- receiver overlap reaches full response at 200 m
- full scenarios use point-source incidence for backend-to-backend comparison
- quick-display scenarios use a shorter 200 m range and smaller grids
- GUI preview requests `max_grid=48` by default to reduce WebView/Plotly memory
  pressure during repeated field switching

Backend interpretation:

- Mie scenarios model spherical/effective particle distributions.
- IITM scenarios expose the non-spherical T-Matrix path; the haze scenario uses
  a spheroid dust coarse mode with axis ratio 1.6 from the PDF recommendation.
- Full Marshall-Palmer rain over 50-3000 um is still represented by
  `temp/lidar_1d`; the current 3D preview backends use smaller water-drop
  proxies so the scenarios remain runnable and previewable.
