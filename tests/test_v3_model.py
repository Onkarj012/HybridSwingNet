import torch

from stockxpert.v3.models import StockXpertModelV3


def test_v3_model_uses_stock_traits_without_embedding():
    model = StockXpertModelV3(
        short_dim=3,
        mid_dim=4,
        long_dim=5,
        context_dim=6,
        hidden_dim=16,
        stock_embed_dim=8,
        stock_traits_dim=7,
        attn_heads=4,
    )

    assert not any(isinstance(module, torch.nn.Embedding) for module in model.modules())

    outputs = model(
        torch.randn(2, 7, 3),
        torch.randn(2, 21, 4),
        torch.randn(2, 60, 5),
        torch.randn(2, 6),
        torch.randn(2, 9),
        torch.randn(2, 7),
    )

    direction_logits, magnitude_pred, confidence, zone_logits, aux_trend, levels, attention = outputs
    assert direction_logits.shape == (2, 5)
    assert magnitude_pred.shape == (2, 5)
    assert confidence.shape == (2, 5)
    assert zone_logits.shape == (2, 5, 7)
    assert aux_trend.shape == (2, 1)
    assert levels.shape == (2, 5, 3)
    assert isinstance(attention, dict)

