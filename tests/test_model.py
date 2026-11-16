import pytest

torch = pytest.importorskip("torch")

from sat_change.model import SatelliteEncoder, feature_change, nt_xent


def test_model_shapes_and_loss():
    model = SatelliteEncoder(4, width=8, projection_dim=16)
    a = torch.rand(4, 4, 32, 32); b = a + torch.randn_like(a) * .01
    loss = nt_xent(model(a), model(b))
    assert loss.isfinite() and loss.item() > 0
    score = feature_change(model.eval(), a[:1], b[:1])
    assert score.shape == (1, 32, 32) and score.min() >= -1e-5



def test_feature_change_scale_weights_and_fusion():
    import pytest

    torch.manual_seed(0)
    model = SatelliteEncoder(4, 8, 16).eval()
    a, b = torch.rand(1, 4, 32, 32), torch.rand(1, 4, 32, 32)
    default = feature_change(model, a, b)
    assert torch.allclose(default, feature_change(model, a, b, [1, 1, 1], "mean"), atol=1e-6)
    assert torch.allclose(default, feature_change(model, a, b, [2, 2, 2]), atol=1e-6)  # weights are normalised
    fine_only = feature_change(model, a, b, [1, 0, 0])
    assert not torch.allclose(default, fine_only)
    geo = feature_change(model, a, b, fusion="geomean")
    assert geo.shape == default.shape and (geo <= default + 1e-5).all()  # geometric mean never exceeds arithmetic
    with pytest.raises(ValueError):
        feature_change(model, a, b, [1, 1])
    with pytest.raises(ValueError):
        feature_change(model, a, b, fusion="max")
