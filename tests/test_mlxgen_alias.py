def test_mlxgen_exposes_mflux_api_without_replacing_its_package_identity():
    import mflux
    import mlxgen

    assert mlxgen.__name__ == "mlxgen"
    assert mlxgen.GeneratedOutput is mflux.GeneratedOutput
    assert mlxgen.resolve_generation_plan is mflux.resolve_generation_plan
    assert mlxgen.resolve_generation_runtime is mflux.resolve_generation_runtime


def test_mlxgen_submodule_import_matches_mflux():
    from mlxgen.models.z_image import ZImageTurbo as MlxgenZImageTurbo

    from mflux.models.z_image import ZImageTurbo as MfluxZImageTurbo

    assert MlxgenZImageTurbo is MfluxZImageTurbo


def test_mlxgen_submodule_import_does_not_replace_mflux_parent_package():
    from mlxgen.models.z_image import ZImageTurbo as MlxgenZImageTurbo

    import mflux
    import mflux.models.common
    from mflux.models.z_image import ZImageTurbo as MfluxZImageTurbo

    assert MlxgenZImageTurbo is MfluxZImageTurbo
    assert mflux.models.__name__ == "mflux.models"
    assert hasattr(mflux.models, "common")
    assert mflux.models.common.__name__ == "mflux.models.common"


def test_mlxgen_public_family_exports_cover_qwen_and_fibo():
    from mlxgen.models.fibo import FIBO, FIBOEdit
    from mlxgen.models.qwen import QwenImage, QwenImageControlNet, QwenImageEdit

    from mflux.models.fibo import (
        FIBO as MfluxFIBO,
        FIBOEdit as MfluxFIBOEdit,
    )
    from mflux.models.qwen import (
        QwenImage as MfluxQwenImage,
        QwenImageControlNet as MfluxQwenImageControlNet,
        QwenImageEdit as MfluxQwenImageEdit,
    )

    assert QwenImage is MfluxQwenImage
    assert QwenImageControlNet is MfluxQwenImageControlNet
    assert QwenImageEdit is MfluxQwenImageEdit
    assert FIBO is MfluxFIBO
    assert FIBOEdit is MfluxFIBOEdit
