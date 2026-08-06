# Alternatives

There are other tools for serializing sklearn models. Here is when
to reach for them instead of skeights.

## skops

**[skops](https://github.com/skops-dev/skops)** is the actively
maintained, scikit-learn-adjacent option for secure persistence,
referenced in sklearn's own docs. It covers pipelines, XGBoost,
LightGBM, has compression, model inspection, and Hugging Face Hub
integration. Reach for skops if you want the broadest, most
battle-tested secure persistence and do not need to inspect
model config without loading it. Use skeights if you want
human-readable, diffable model configuration.

## sklearn-migrator

**[sklearn-migrator](https://github.com/anvaldes/sklearn-migrator)**
is purpose-built for loading models across different sklearn
versions, with a peer-reviewed paper behind it. Reach for it if
cross-version migration is your problem. skeights does not guarantee
cross-version support. Note that sklearn-migrator does not yet cover
pipelines, XGBoost, or LightGBM, which skeights does.

!!! note
    The feature descriptions of skops and sklearn-migrator
    above reflect their state as of mid-2026. Check their current
    docs for the latest.
