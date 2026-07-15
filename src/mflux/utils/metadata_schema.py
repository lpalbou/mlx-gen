class MetadataSchema:
    # Version of the metadata sidecar / embedded-metadata structure. Evolution is additive-only:
    # bump when a field's meaning changes or a field is removed, never for new optional fields.
    # v2: the 0.21.0 masked-edit key `masked_warm_start_strength` was renamed to `mask_strength`
    # when the warm start became a user option; replay still reads the legacy key.
    VERSION = 2
