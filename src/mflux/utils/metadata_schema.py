class MetadataSchema:
    # Version of the metadata sidecar / embedded-metadata structure. Evolution is additive-only:
    # bump when a field's meaning changes or a field is removed, never for new optional fields.
    VERSION = 1
