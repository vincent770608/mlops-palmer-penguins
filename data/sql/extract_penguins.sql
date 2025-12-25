SELECT
    species,
    island,
    culmen_length_mm,
    culmen_depth_mm,
    flipper_length_mm,
    body_mass_g
FROM
    `bigquery-public-data.ml_datasets.penguins`
WHERE
    body_mass_g IS NOT NULL